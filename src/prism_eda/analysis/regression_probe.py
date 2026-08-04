"""What happens when a model is actually fitted?

Everything in this module is *model-conditional*, and the report says so. A
residual is not a property of a dataset; it is what one estimator left over on
one split. That distinction matters because the diagnostics here are the most
persuasive-looking numbers in the whole recipe, and the easiest to over-read.

Two probes run rather than one, because the disagreement between them is itself
the finding. Ridge minimizes squared error and is dragged toward outliers; Huber
down-weights them. When the robust probe fits well and the conventional one does
not, the data is predictable and a handful of rows are distorting the fit — which
is a completely different problem from "these features do not explain the
target", and it is the influence table, not the feature list, that fixes it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from prism_eda.analysis._regression import (
    MAX_REVIEW_ROWS,
    as_float,
    bin_edges,
    distribution_payload,
    preprocessor,
    probe_folds,
    safe_ratio,
)
from prism_eda.catalog.models import TableCatalog
from prism_eda.config import AnalysisConfig
from prism_eda.evidence.models import Evidence, EvidenceScope

#: Residual bins used by the spread, bias, and coverage diagnostics.
_DIAGNOSTIC_BINS = 10

#: Below this many rows per bin the bin's standard deviation is noise.
_MIN_BIN_ROWS = 5

#: Influence is an in-sample concept and needs a full design matrix, so it is
#: capped independently of the recipe's row budget.
_MAX_INFLUENCE_ROWS = 50_000

# Thresholds for earning a place on the review list.
#
# The familiar 4/n Cook's distance rule is a *screening* convention: by
# construction it flags a few percent of rows in any dataset, clean ones
# included. Using it alone to raise a finding would mean every regression ever
# run reports rows to review, which is precisely the noise this library exists
# to avoid. So 4/n is kept as a reported statistic, and a row reaches the list
# only by clearing a threshold that ordinary data does not: decisive influence,
# a residual far outside the fit's own spread, or both signals together at
# moderate strength.
_COOKS_DECISIVE = 0.5
_RESIDUAL_DECISIVE = 4.0
_RESIDUAL_MODERATE = 3.0

#: Residual spread across fitted bins, as a ratio of widest to narrowest. Below
#: this a single prediction interval is defensible.
_HETEROSCEDASTICITY_RATIO = 3.0

#: Conditional bias as a fraction of residual spread. Half a residual standard
#: deviation of systematic error in one region is large.
_CONDITIONAL_BIAS_RATIO = 0.5

#: Subgroup error relative to the whole, and the share of rows a subgroup needs
#: before its error is worth reporting.
_ERROR_CONCENTRATION_RATIO = 1.5
_MIN_SUBGROUP_RATE = 0.05

#: A probe this close to the baseline has found nothing.
_WEAK_R2 = 0.05
_WEAK_MAE_RATIO = 0.95

#: How much better the robust probe must fit the typical row before the
#: conventional fit is called outlier-sensitive.
_ROBUST_GAIN = 0.20

#: An interior stretch of the target range holding less than this share of rows
#: is a region the model cannot learn.
_SPARSE_BIN_RATE = 0.02
_FEATURE_GAP_FRACTION = 0.2

#: Mass required on *each* side of a thin bin before it counts as a hole.
#:
#: Splitting any bell-shaped target into equal-width bins leaves thin bins out
#: in the tails — that is what a tail is, not a gap in support. Without this
#: guard the check fires on every normally distributed target, which is the
#: opposite of useful. A genuine hole has substantial data on both sides of it.
_SPARSE_BIN_FLANK_RATE = 0.05


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r_squared": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "median_ae": float(median_absolute_error(actual, predicted)),
    }


def probe_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    target_series: pd.Series,
    *,
    config: AnalysisConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    excluded_features: list[str],
) -> tuple[Evidence | None, pd.Series | None]:
    """Cross-validated Ridge and Huber probes, plus the median baseline.

    Returns the evidence and the out-of-fold predictions of whichever probe fit
    best, so the residual diagnostics downstream describe the model a reader
    would actually reach for.
    """
    features = numeric_features + categorical_features
    if not features:
        return None, None
    usable = frame.loc[target_series.notna(), features]
    y = target_series.loc[usable.index]
    folds = probe_folds(len(usable), config.mode)
    if folds is None or y.nunique() < 2:
        return None, None

    splitter = KFold(n_splits=folds, shuffle=True, random_state=config.random_seed)
    actual = y.to_numpy(dtype="float64")

    def run(name: str, estimator: Any) -> tuple[dict[str, Any], pd.Series]:
        pipeline = Pipeline(
            [
                ("preprocess", preprocessor(numeric_features, categorical_features)),
                ("model", estimator),
            ]
        )
        predicted = cross_val_predict(pipeline, usable, y, cv=splitter)
        return (
            {"model": name, **_metrics(actual, predicted)},
            pd.Series(predicted, index=usable.index, dtype="float64"),
        )

    results: list[dict[str, Any]] = []
    predictions: dict[str, pd.Series] = {}
    for name, estimator in (
        ("ridge", Ridge(alpha=1.0)),
        ("huber", HuberRegressor(max_iter=500)),
    ):
        try:
            summary, predicted = run(name, estimator)
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate
            continue
        results.append(summary)
        predictions[name] = predicted
    if not results:
        return None, None

    baseline_predicted = cross_val_predict(
        DummyRegressor(strategy="median"), usable, y, cv=splitter
    )
    baseline = {"model": "median_baseline", **_metrics(actual, baseline_predicted)}

    # "Is this predictable at all" is answered by whichever probe did best, so
    # an outlier-sensitive conventional fit cannot make predictable data look
    # hopeless.
    best_r_squared = max(as_float(item["r_squared"]) for item in results)
    best_mae = min(as_float(item["mae"]) for item in results)
    mae_ratio = safe_ratio(best_mae, as_float(baseline["mae"]))

    # Comparing the two probes on R-squared would be meaningless: Huber
    # optimizes a robust loss and therefore *cannot* win on squared error, so a
    # negative difference would be an artifact of the metric rather than a
    # finding. The informative comparison is on the typical row, where a large
    # robust advantage means a minority of rows is dragging the squared-error
    # fit — a problem the influence table fixes, not the feature list.
    by_name = {item["model"]: item for item in results}
    robust_gain: float | None = None
    if "ridge" in by_name and "huber" in by_name:
        conventional_median = as_float(by_name["ridge"]["median_ae"])
        robust_median = as_float(by_name["huber"]["median_ae"])
        robust_gain = safe_ratio(
            conventional_median - robust_median, conventional_median
        )

    # Residual diagnostics describe the conventional fit. It is the model a
    # reader will reach for by default, so its residuals are the ones worth
    # showing; the robust probe's job here is the comparison above.
    reference = "ridge" if "ridge" in predictions else results[0]["model"]

    evidence = Evidence.create(
        kind="regression_probe",
        scope=EvidenceScope(table=table.name, columns=tuple(features + [target])),
        value={
            "target": target,
            "models": results,
            "baseline": baseline,
            "reference_model": reference,
            "best_r_squared": best_r_squared,
            "mae_vs_baseline_ratio": mae_ratio,
            "robust_median_error_gain": robust_gain,
            "cv_folds": folds,
            "row_count": int(len(usable)),
            "feature_count": len(features),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "excluded_features": excluded_features,
        },
        method="leakage_screened_dual_probe_cv_v1",
        description=f"Cross-validated diagnostic regression probes for {table.name}.",
        confidence=0.74,
        assumptions=(
            "The probes are diagnostic fits, not production models. Every number "
            "derived from them describes these estimators on this data.",
            "Preprocessing is fitted inside each cross-validation fold.",
            "Leakage candidates, identifiers, and over-wide categoricals are "
            "excluded from the feature set.",
        ),
    )
    return evidence, predictions.get(reference)


def _binned_residuals(
    fitted: np.ndarray, residuals: np.ndarray
) -> list[dict[str, Any]]:
    edges = bin_edges(fitted, _DIAGNOSTIC_BINS)
    if edges.size < 3:
        return []
    index = np.clip(np.digitize(fitted, edges[1:-1]), 0, edges.size - 2)
    bins: list[dict[str, Any]] = []
    for position in range(edges.size - 1):
        mask = index == position
        count = int(mask.sum())
        if count < _MIN_BIN_ROWS:
            continue
        window = residuals[mask]
        bins.append(
            {
                "low": float(edges[position]),
                "high": float(edges[position + 1]),
                "count": count,
                "mean_residual": float(window.mean()),
                "std_residual": float(window.std(ddof=1)) if count > 1 else 0.0,
                "mean_absolute_error": float(np.abs(window).mean()),
            }
        )
    return bins


def residual_evidence(
    table: TableCatalog,
    target: str,
    actual: pd.Series,
    predicted: pd.Series,
    probe: Evidence,
) -> list[Evidence]:
    """Residual shape, spread, and conditional bias from the best probe."""
    residuals = (actual - predicted).to_numpy(dtype="float64")
    fitted = predicted.to_numpy(dtype="float64")
    evidence: list[Evidence] = []
    spread = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    model = probe.value["reference_model"]

    standardized = residuals / spread if spread > 0 else residuals
    normality_distance = (
        float(stats.kstest(standardized, "norm").statistic)
        if residuals.size >= 20 and spread > 0
        else None
    )
    evidence.append(
        Evidence.create(
            kind="regression_residual_shape",
            scope=EvidenceScope(table=table.name, columns=(target,)),
            value={
                "target": target,
                "model": model,
                "mean_residual": float(residuals.mean()),
                "std_residual": spread,
                "skewness": as_float(pd.Series(residuals).skew()),
                "kurtosis": as_float(pd.Series(residuals).kurtosis()),
                "normality_distance": normality_distance,
                "distribution": distribution_payload(
                    residuals, column=f"{target} residual"
                ),
            },
            method="probe_residual_shape_v1",
            description=f"Residual shape for {table.name}.{target}.",
            confidence=0.76,
            assumptions=(
                "The normality figure is a Kolmogorov-Smirnov distance, not a "
                "p-value: the comparison distribution is estimated from the same "
                "residuals, so a p-value would be misleading.",
                "Residual shape describes the probe, not the dataset.",
            ),
        )
    )

    bins = _binned_residuals(fitted, residuals)
    if len(bins) >= 3:
        spreads = [item["std_residual"] for item in bins if item["std_residual"] > 0]
        spread_ratio = (
            safe_ratio(max(spreads), min(spreads)) if len(spreads) >= 2 else None
        )
        breusch_pagan: float | None = None
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan

            design = np.column_stack([np.ones(fitted.size), fitted])
            breusch_pagan = float(het_breuschpagan(residuals, design)[0])
        except (ValueError, ImportError):  # pragma: no cover - degenerate design
            breusch_pagan = None
        evidence.append(
            Evidence.create(
                kind="regression_heteroscedasticity",
                scope=EvidenceScope(table=table.name, columns=(target,)),
                value={
                    "target": target,
                    "model": model,
                    "bins": bins,
                    "spread_ratio": spread_ratio,
                    "breusch_pagan_statistic": breusch_pagan,
                    "overall_std_residual": spread,
                },
                method="binned_residual_spread_breusch_pagan_v1",
                description=f"Residual spread across the fitted range for {target}.",
                confidence=0.8,
                assumptions=(
                    "Unequal residual spread does not bias predictions; it makes "
                    "one uniform prediction interval wrong at both ends.",
                ),
            )
        )

        biases = [
            abs(item["mean_residual"]) / spread if spread > 0 else 0.0 for item in bins
        ]
        evidence.append(
            Evidence.create(
                kind="regression_conditional_bias",
                scope=EvidenceScope(table=table.name, columns=(target,)),
                value={
                    "target": target,
                    "model": model,
                    "bins": bins,
                    "max_bias_ratio": float(max(biases)) if biases else 0.0,
                    "overall_std_residual": spread,
                },
                method="fitted_decile_conditional_bias_v1",
                description=(
                    f"Systematic over/under-prediction by fitted range for {target}."
                ),
                confidence=0.8,
                assumptions=(
                    "A linear fit on a censored or skewed target typically "
                    "over-predicts the bottom of the range and under-predicts the "
                    "top; that pattern is the signature, not a coincidence.",
                ),
            )
        )
    return evidence


def error_concentration_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    actual: pd.Series,
    predicted: pd.Series,
    *,
    group_columns: list[str],
) -> Evidence | None:
    """Where the error lives: subgroups the probe serves noticeably worse.

    Two corrections stand between a raw mean absolute error and a statement
    anyone should act on.

    First, error cannot be compared across groups on a raw scale: a group whose
    target is ten times larger carries ten times the absolute error while being
    predicted exactly as well, so ranking on mean absolute error reliably flags
    the highest-magnitude group and tells you nothing. Each group's error is
    therefore divided by that group's own target spread.

    Second — and this is the subtle one — the scaled figure still cannot be
    compared against the *overall* rate. Whenever the grouping column predicts
    the target, the spread within any one level is far smaller than the spread
    across all of them, so every level scores worse than the whole and the
    diagnostic fires on all of them at once. The meaningful comparison is
    between siblings: each level against the median level of its own column,
    which answers the question actually being asked — is this group served worse
    than the others like it?
    """
    if not group_columns:
        return None
    errors = (actual - predicted).abs()
    overall_error = float(errors.mean())
    overall_spread = float(actual.std(ddof=1)) if len(actual) > 1 else 0.0
    if overall_error <= 0 or overall_spread <= 0:
        return None

    rows: list[dict[str, Any]] = []
    for column in group_columns:
        series = frame.loc[errors.index, column]
        if series.nunique(dropna=True) < 2:
            continue
        levels = series.astype("string")
        column_rows: list[dict[str, Any]] = []
        for level, group in errors.groupby(levels, dropna=True):
            rate = len(group) / len(errors)
            if rate < _MIN_SUBGROUP_RATE or len(group) < 2:
                continue
            level_spread = float(actual.loc[group.index].std(ddof=1))
            if not np.isfinite(level_spread) or level_spread <= 0:
                # A constant target inside the group has nothing to explain, so
                # a ratio here would divide by zero rather than mean anything.
                continue
            level_error = float(group.mean())
            column_rows.append(
                {
                    "column": column,
                    "level": str(level),
                    "row_count": int(len(group)),
                    "row_rate": rate,
                    "mean_absolute_error": level_error,
                    "target_std": level_spread,
                    "normalized_error": level_error / level_spread,
                    "raw_error_ratio": level_error / overall_error,
                }
            )
        # Fewer than two comparable levels leaves nothing to compare against.
        if len(column_rows) < 2:
            continue
        peer = float(np.median([item["normalized_error"] for item in column_rows]))
        if peer <= 0:
            continue
        for item in column_rows:
            item["peer_normalized_error"] = peer
            item["error_ratio"] = item["normalized_error"] / peer
        rows.extend(column_rows)

    if not rows:
        return None
    rows.sort(key=lambda item: item["error_ratio"], reverse=True)
    return Evidence.create(
        kind="regression_error_concentration",
        scope=EvidenceScope(table=table.name, columns=tuple(group_columns)),
        value={
            "target": target,
            "overall_mean_absolute_error": overall_error,
            "overall_target_std": overall_spread,
            "groups": rows[:MAX_REVIEW_ROWS],
            "max_error_ratio": rows[0]["error_ratio"],
        },
        method="peer_relative_subgroup_error_v1",
        description=f"Subgroup error concentration for {table.name}.{target}.",
        confidence=0.78,
        assumptions=(
            "Error is scaled by each group's own target spread and compared "
            "against the median level of the same column, so neither a group "
            "with larger values nor a grouping that predicts the target is "
            "flagged for that alone.",
            "Higher relative error can mean the group is genuinely noisier or "
            "that it is under-represented in the features.",
        ),
    )


def _extreme_features(
    frame: pd.DataFrame, index: Any, numeric_features: list[str]
) -> list[dict[str, Any]]:
    """Per-row robust deviations, in the shape ``why_bars_svg`` renders."""
    contributors: list[dict[str, Any]] = []
    for column in numeric_features:
        series = pd.to_numeric(frame[column], errors="coerce")
        value = series.get(index)
        if value is None or pd.isna(value):
            continue
        median = float(series.median())
        deviation = float((series - median).abs().median())
        scale = deviation * 1.4826 if deviation > 0 else float(series.std(ddof=1) or 0)
        if not scale:
            continue
        robust_z = (float(value) - median) / scale
        if abs(robust_z) < 2.0:
            continue
        contributors.append(
            {
                "column": column,
                "value": float(value),
                "baseline": median,
                "robust_z": robust_z,
            }
        )
    contributors.sort(key=lambda item: abs(item["robust_z"]), reverse=True)
    return contributors[:5]


def influence_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    actual: pd.Series,
    predicted: pd.Series,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[Evidence]:
    """Leverage, Cook's distance, and the ranked rows worth opening."""
    features = numeric_features + categorical_features
    usable_index = actual.index
    if not features or len(usable_index) > _MAX_INFLUENCE_ROWS:
        return []
    residuals = actual - predicted
    spread = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0

    leverage: pd.Series | None = None
    cooks: pd.Series | None = None
    try:
        import statsmodels.api as sm

        design = preprocessor(numeric_features, categorical_features).fit_transform(
            frame.loc[usable_index, features]
        )
        matrix = np.column_stack([np.ones(len(usable_index)), np.asarray(design)])
        fitted = sm.OLS(actual.to_numpy(dtype="float64"), matrix).fit()
        influence = fitted.get_influence()
        leverage = pd.Series(influence.hat_matrix_diag, index=usable_index)
        cooks = pd.Series(influence.cooks_distance[0], index=usable_index)
    except Exception:  # pragma: no cover - singular or ill-conditioned design
        leverage = None
        cooks = None

    standardized = residuals / spread if spread > 0 else residuals * 0.0
    # Rank by whichever signal is available: influence combines an extreme
    # position in feature space with a bad fit, and a row only earns review when
    # it is unusual on at least one of them.
    score = standardized.abs()
    if cooks is not None:
        cook_scale = float(cooks.median()) or 1.0
        score = score + (cooks / cook_scale).clip(upper=50.0)
    ranked = score.sort_values(ascending=False).head(MAX_REVIEW_ROWS)

    rows: list[dict[str, Any]] = []
    decisive_rows = 0
    # 4/n is the conventional Cook's distance screen. It is a screening rule for
    # "worth a look", not a significance test, and the report says so.
    cook_threshold = 4.0 / len(usable_index)
    for index in ranked.index:
        row_std = float(standardized.loc[index])
        row_leverage = float(leverage.loc[index]) if leverage is not None else None
        row_cooks = float(cooks.loc[index]) if cooks is not None else None
        magnitude = abs(row_std)

        decisive = magnitude >= _RESIDUAL_DECISIVE or (
            row_cooks is not None and row_cooks >= _COOKS_DECISIVE
        )
        corroborated = (
            row_cooks is not None
            and row_cooks >= cook_threshold
            and magnitude >= _RESIDUAL_MODERATE
        )
        if not (decisive or corroborated):
            continue
        decisive_rows += int(decisive)

        reasons: list[str] = []
        if magnitude >= _RESIDUAL_MODERATE:
            reasons.append("large residual")
        if row_cooks is not None and row_cooks >= cook_threshold:
            reasons.append("high influence")
        if row_leverage is not None and row_leverage >= 3.0 * (
            (len(features) + 1) / len(usable_index)
        ):
            reasons.append("extreme feature values")
        rows.append(
            {
                "row_index": str(index),
                "actual": float(actual.loc[index]),
                "predicted": float(predicted.loc[index]),
                "residual": float(residuals.loc[index]),
                "standardized_residual": row_std,
                "leverage": row_leverage,
                "cooks_distance": row_cooks,
                "reasons": reasons,
                "extreme_features": _extreme_features(
                    frame.loc[usable_index], index, numeric_features
                ),
            }
        )
    # A 3-sigma residual among a few hundred Gaussian draws is *expected*, not
    # anomalous — roughly one row in every clean dataset of this size clears the
    # corroborated bar on chance alone. Publishing that as "1 row worth
    # reviewing" would put a finding on every well-behaved regression ever run.
    # So the list survives only when at least one row is decisive on its own;
    # the corroborated rows then ride along as context for a real problem
    # rather than constituting one.
    if not decisive_rows:
        rows = []

    evidence = [
        Evidence.create(
            kind="regression_influence",
            scope=EvidenceScope(table=table.name, columns=(target,)),
            value={
                "target": target,
                "evaluated_row_count": int(len(usable_index)),
                "cooks_distance_threshold": cook_threshold,
                "high_influence_count": int((cooks >= cook_threshold).sum())
                if cooks is not None
                else 0,
                "large_residual_count": int(
                    (standardized.abs() >= _RESIDUAL_MODERATE).sum()
                ),
                "decisive_row_count": decisive_rows,
                "influence_available": cooks is not None,
            },
            method="ols_leverage_cooks_distance_v1",
            description=f"Influence diagnostics for {table.name}.{target}.",
            confidence=0.8,
            assumptions=(
                "Leverage and Cook's distance come from an ordinary least-squares "
                "fit on the screened feature set, so they inherit its assumptions.",
                "The 4/n Cook's distance threshold is a screening convention for "
                "rows worth inspecting, not a test of significance. A few rows "
                "clear it in any dataset, so it alone does not raise a finding.",
            ),
        )
    ]
    if not rows:
        return evidence
    evidence.append(
        Evidence.create(
            kind="regression_review_rows",
            scope=EvidenceScope(table=table.name, columns=(target,)),
            value={
                "target": target,
                "rows": rows,
                "row_count": len(rows),
                "residual_std": spread,
            },
            method="ranked_regression_review_rows_v1",
            description=f"Rows worth reviewing in {table.name}.{target}.",
            confidence=0.76,
            assumptions=(
                "These rows are review candidates. A large residual can mean a "
                "data error, a missing feature, or a genuinely unusual case.",
            ),
        )
    )
    return evidence


def weak_support_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    target_series: pd.Series,
    *,
    numeric_features: list[str],
) -> Evidence | None:
    """Ranges the model has almost no examples for, so predicting there guesses."""
    finite = target_series.dropna().to_numpy(dtype="float64")
    if finite.size < 50:
        return None
    counts, edges = np.histogram(finite, bins=_DIAGNOSTIC_BINS)
    total = counts.sum() or 1
    bins = [
        {
            "low": float(edges[index]),
            "high": float(edges[index + 1]),
            "count": int(count),
            "rate": float(count / total),
        }
        for index, count in enumerate(counts)
    ]
    # Only a genuine hole matters. A thin bin with almost nothing beyond it is
    # the distribution's tail; a thin bin with real mass on both sides is a
    # range the model has no examples for and will be asked to predict anyway.
    sparse = []
    for position, item in enumerate(bins):
        if item["rate"] >= _SPARSE_BIN_RATE:
            continue
        below = sum(entry["rate"] for entry in bins[:position])
        above = sum(entry["rate"] for entry in bins[position + 1 :])
        if below < _SPARSE_BIN_FLANK_RATE or above < _SPARSE_BIN_FLANK_RATE:
            continue
        sparse.append({**item, "mass_below": below, "mass_above": above})

    gaps: list[dict[str, Any]] = []
    for column in numeric_features:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 20:
            continue
        ordered = np.sort(values.to_numpy(dtype="float64"))
        span = float(ordered[-1] - ordered[0])
        if span <= 0:
            continue
        diffs = np.diff(ordered)
        largest = float(diffs.max())
        if largest / span < _FEATURE_GAP_FRACTION:
            continue
        position = int(np.argmax(diffs))
        gaps.append(
            {
                "feature": column,
                "gap_fraction": largest / span,
                "gap_low": float(ordered[position]),
                "gap_high": float(ordered[position + 1]),
            }
        )
    gaps.sort(key=lambda item: item["gap_fraction"], reverse=True)

    if not sparse and not gaps:
        return None
    return Evidence.create(
        kind="regression_weak_support",
        scope=EvidenceScope(table=table.name, columns=(target,)),
        value={
            "target": target,
            "target_bins": bins,
            "sparse_target_bins": sparse,
            "feature_gaps": gaps[:MAX_REVIEW_ROWS],
            "sparse_row_rate": float(sum(item["rate"] for item in sparse)),
        },
        method="target_coverage_and_feature_gap_scan_v1",
        description=f"Weakly supported ranges for {table.name}.{target}.",
        confidence=0.74,
        assumptions=(
            "A prediction inside a range with almost no training examples is "
            "extrapolation, however confident the model looks.",
        ),
    )


#: Points drawn in the residual scatter before deterministic sampling. Every
#: point costs file size in a report that must stay a single portable file, and
#: past a few thousand the shape stops changing.
_MAX_SCATTER_POINTS = 2_000


def residual_scatter_evidence(
    table: TableCatalog,
    target: str,
    actual: pd.Series,
    predicted: pd.Series,
    *,
    config: AnalysisConfig,
    bins: list[dict[str, Any]],
    flagged: set[str],
) -> Evidence | None:
    """Chart data for the residual plot, banked so the report needs no frame."""
    residuals = actual - predicted
    if residuals.empty:
        return None
    frame = pd.DataFrame({"x": predicted, "y": residuals})
    sampled = False
    if len(frame) > _MAX_SCATTER_POINTS:
        frame = frame.sample(
            n=_MAX_SCATTER_POINTS, random_state=config.random_seed
        ).sort_index()
        sampled = True
    points = [
        {
            "x": as_float(row.x),
            "y": as_float(row.y),
            "flagged": str(index) in flagged,
        }
        for index, row in zip(frame.index, frame.itertuples(), strict=True)
    ]
    return Evidence.create(
        kind="regression_residual_scatter",
        scope=EvidenceScope(table=table.name, columns=(target,)),
        value={
            "target": target,
            "points": points,
            "bins": bins,
            "point_count": len(points),
            "sampled": sampled,
        },
        method="residual_versus_fitted_chart_data_v1",
        description=f"Residual-versus-fitted chart data for {table.name}.{target}.",
        confidence=1.0,
        assumptions=(
            "Points are the cross-validated residuals of the reference probe.",
        ),
    )


def probe_is_weak(probe: Evidence) -> bool:
    """True when neither probe found anything the median baseline did not."""
    ratio = probe.value["mae_vs_baseline_ratio"]
    return bool(
        probe.value["best_r_squared"] < _WEAK_R2
        and (ratio is None or ratio > _WEAK_MAE_RATIO)
    )


def probe_is_outlier_sensitive(probe: Evidence) -> bool:
    """True when the robust probe fits the typical row much better.

    This is the signal that the data *is* predictable and a minority of rows is
    distorting a squared-error fit — a different problem from weak features, and
    one the review rows address directly.
    """
    gain = probe.value["robust_median_error_gain"]
    return bool(gain is not None and gain >= _ROBUST_GAIN)
