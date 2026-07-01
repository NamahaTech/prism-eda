"""Goal-aware deterministic diagnostics for anomaly-detection EDA."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog, TableCatalog
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import (
    Evidence,
    EvidenceScope,
    Finding,
    sort_findings,
)
from prism_eda.results import (
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)
from prism_eda.transformations.models import TransformationPlan, TransformationStep

_ROW_BUDGETS = {
    AnalysisMode.QUICK: 25_000,
    AnalysisMode.STANDARD: 100_000,
    AnalysisMode.DEEP: 250_000,
}

_DEFAULT_REVIEW_RATE = 0.02
_MAX_REVIEW_ROWS = 25

# Every Normal column has ~0.7% of rows beyond 1.5x IQR fences, so a non-zero
# IQR tail count is expected, not anomalous. A univariate tail is only worth a
# top-level finding when at least one value sits far out (large robust z) or the
# tail is genuinely heavy (a high candidate rate). Lower-signal tails still
# produce evidence for the candidate-signal table, just not a headline finding.
_UNIVARIATE_FINDING_MIN_ROBUST_Z = 8.0
_UNIVARIATE_FINDING_MIN_RATE = 0.05

# Conditional-anomaly checks run over ordered feature pairs, which grows as
# k*(k-1). Keep the evidence but cap how many become headline findings so the
# report does not drown in pairwise combinations.
_MAX_CONDITIONAL_FINDINGS = 3

# Row-centric consensus. Instead of one finding per detector (the same rows
# described six ways), we collect every flagged row and lead with the rows
# themselves: what they are, how unusual, and which checks agreed. A row only
# earns a place on the review list if a threshold-based detector caught it
# (univariate/multivariate/conditional) — the ranked detectors (Isolation
# Forest, LOF) always emit their top-k, so they corroborate but never originate.
_CONSENSUS_MAX_ROWS = 15
_CONTRIBUTOR_MIN_Z = 2.0
_MAX_CONTRIBUTORS = 3
# A single conditional hit this strong is a genuine in-context anomaly worth the
# review list on its own; weaker single conditional hits only corroborate.
_CONSENSUS_STRONG_CONDITIONAL = 5.0
# Even when two detectors agree, require some real extremity so the ordinary tails
# of a clean distribution (a few points just past 3.5σ, which any normal column
# has) do not manufacture a review list out of noise.
_CONSENSUS_MIN_AGREE_Z = 4.5
_THRESHOLD_SIGNAL_KINDS = {
    "anomaly_univariate_outlier",
    "anomaly_multivariate_outlier",
    "anomaly_conditional_outlier",
}
_CORROBORATING_SIGNAL_KINDS = {
    "anomaly_isolation_forest",
    "anomaly_local_density_outlier",
}

# Distribution-shape diagnostics. A column with two separated clusters is not a
# clean distribution with a few tail outliers — it is two populations, and that
# reframing is usually the single most useful thing to tell an analyst. We detect
# it with a robust largest-gap split (on a log axis for heavily skewed positive
# columns) and only call it when both sides hold a real share of the rows.
_HIST_BINS = 24
_BIMODAL_MIN_FRACTION = 0.15
_BIMODAL_MIN_GROUP = 4
_BIMODAL_GAP_RATIO = 2.5
# The separating gap must also span a real share of the column's spread, so that
# the small ±1 gaps in ordinary discrete/uniform data (e.g. integer tenures) are
# not mistaken for a true two-population split.
_BIMODAL_MIN_RANGE_FRACTION = 0.10
_SCATTER_MAX_POINTS = 2000


def _row_budget(mode: AnalysisMode | str) -> int:
    return _ROW_BUDGETS[AnalysisMode(mode)]


def _sample_frame(
    frame: pd.DataFrame,
    *,
    table: str,
    config: AnalysisConfig,
    warnings: list[AnalysisWarning],
    sampling: list[SamplingRecord],
) -> pd.DataFrame:
    budget = _row_budget(config.mode)
    if config.sampling == "disabled" or len(frame) <= budget:
        return frame
    sampled = frame.sample(n=budget, random_state=config.random_seed).sort_index()
    warnings.append(
        AnalysisWarning(
            code="sampled_anomaly_detection",
            message=(
                f"{table} has {len(frame):,} rows; anomaly diagnostics were run on "
                f"a deterministic {budget:,}-row sample."
            ),
            table=table,
        )
    )
    sampling.append(
        SamplingRecord(
            operation="anomaly_detection",
            source_rows=len(frame),
            sampled_rows=budget,
            strategy="deterministic_pandas_sample",
            seed=config.random_seed,
            reason="row_count_exceeds_mode_budget",
            limitations=(
                "Rare anomalies may be missed when they are absent from the sample.",
            ),
        )
    )
    return sampled


def _numeric_columns(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    exclude: set[str] | None = None,
    allow_identifiers: bool = True,
) -> list[str]:
    excluded = exclude or set()
    columns: list[str] = []
    catalog_by_name = {column.name: column for column in table.columns}
    for column in frame.columns:
        name = str(column)
        if name in excluded or name not in catalog_by_name:
            continue
        series = frame[column]
        profile = catalog_by_name[name]
        if not ptypes.is_numeric_dtype(series.dtype) or series.notna().sum() < 8:
            continue
        if not allow_identifiers and (
            "identifier_candidate" in profile.roles
            or name.lower() in {"id", "index"}
            or name.lower().endswith("_id")
        ):
            continue
        columns.append(name)
    return columns


def _categorical_columns(frame: pd.DataFrame, table: TableCatalog) -> list[str]:
    catalog_by_name = {column.name: column for column in table.columns}
    columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        profile = catalog_by_name.get(name)
        if profile is None:
            continue
        if profile.semantic_type in {"categorical", "boolean", "text"}:
            if profile.unique_count is None or profile.unique_count <= 100:
                columns.append(name)
    return columns


def _robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    deviations = (numeric - median).abs()
    mad = deviations.median()
    if pd.notna(mad) and mad > 0:
        return 0.6745 * (numeric - median) / mad
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if pd.notna(iqr) and iqr > 0:
        return (numeric - median) / (iqr / 1.349)
    std = numeric.std(ddof=0)
    if pd.notna(std) and std > 0:
        return (numeric - numeric.mean()) / std
    return pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")


def _top_index_values(scores: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    sorted_scores = scores.dropna().abs().sort_values(ascending=False).head(limit)
    return [
        {"row_index": str(index), "score": float(score)}
        for index, score in sorted_scores.items()
    ]


def _review_count(row_count: int, expected_contamination: float | None) -> int:
    if row_count <= 0:
        return 0
    rate = (
        expected_contamination
        if expected_contamination is not None
        else _DEFAULT_REVIEW_RATE
    )
    return max(1, min(_MAX_REVIEW_ROWS, math.ceil(row_count * rate)))


def _validate_expected_contamination(value: float | None) -> None:
    if value is None:
        return
    if not 0 < value < 0.5:
        raise ValueError(
            "expected_contamination must be greater than 0 and less than 0.5"
        )


def _numeric_model_matrix(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    columns = _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=False
    )[:12]
    if len(columns) < 2:
        return columns, pd.DataFrame(), pd.DataFrame()
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.dropna()
    if len(usable) < 20:
        return columns, pd.DataFrame(), pd.DataFrame()
    z_frame = usable.apply(_robust_z)
    return columns, usable, z_frame


def _model_top_records(
    scores: pd.Series,
    z_frame: pd.DataFrame,
    *,
    count: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, score in scores.sort_values(ascending=False).head(count).items():
        row_z = z_frame.loc[index].abs().sort_values(ascending=False).head(3)
        records.append(
            {
                "row_index": str(index),
                "score": float(score),
                "top_contributors": [
                    {"column": str(column), "abs_robust_z": float(value)}
                    for column, value in row_z.items()
                ],
            }
        )
    return records


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _category_value(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.floating) and bool(np.isnan(value)):
        return None
    return str(value)


def _univariate_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for column in _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=True
    ):
        series = pd.to_numeric(frame[column], errors="coerce")
        non_null = series.dropna()
        if len(non_null) < 8:
            continue
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        robust_z = _robust_z(series)
        mask = ((series < lower) | (series > upper) | (robust_z.abs() >= 3.5)) & (
            series.notna()
        )
        count = int(mask.sum())
        evidence.append(
            Evidence.create(
                kind="anomaly_univariate_outlier",
                scope=EvidenceScope(table=table.name, columns=(column,)),
                value={
                    "evaluated_row_count": int(series.notna().sum()),
                    "candidate_count": count,
                    "candidate_rate": count / len(non_null) if len(non_null) else 0.0,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "max_abs_robust_z": float(robust_z.abs().max(skipna=True) or 0.0),
                    "examples": _top_index_values(robust_z[mask]),
                },
                method="iqr_or_modified_z_score_v1",
                description=f"Univariate tail candidates for {table.name}.{column}.",
                confidence=0.82 if count else 0.72,
                assumptions=(
                    "Tail candidates are statistical review targets, not confirmed "
                    "anomalies.",
                ),
            )
        )
    return evidence


def _multivariate_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> Evidence | None:
    columns = _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=False
    )[:12]
    if len(columns) < 2 or len(frame) < 8:
        return None
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.dropna()
    if len(usable) < 8:
        return None
    z_frame = usable.apply(_robust_z)
    scores = np.sqrt((z_frame**2).sum(axis=1))
    threshold = max(4.5, math.sqrt(len(columns)) * 3.0)
    candidate_scores = scores[scores >= threshold]
    top_records: list[dict[str, Any]] = []
    for index, score in candidate_scores.sort_values(ascending=False).head(10).items():
        row_z = z_frame.loc[index].abs().sort_values(ascending=False).head(3)
        top_records.append(
            {
                "row_index": str(index),
                "score": float(score),
                "top_contributors": [
                    {"column": str(column), "abs_robust_z": float(value)}
                    for column, value in row_z.items()
                ],
            }
        )
    return Evidence.create(
        kind="anomaly_multivariate_outlier",
        scope=EvidenceScope(table=table.name, columns=tuple(columns)),
        value={
            "evaluated_row_count": int(len(usable)),
            "feature_count": len(columns),
            "threshold": threshold,
            "candidate_count": int(len(candidate_scores)),
            "candidate_rate": len(candidate_scores) / len(usable),
            "max_score": float(scores.max() if len(scores) else 0.0),
            "top_records": top_records,
        },
        method="robust_scaled_euclidean_score_v1",
        description=f"Multivariate robust-score candidates for {table.name}.",
        confidence=0.74 if len(candidate_scores) else 0.62,
        assumptions=(
            "Features are robust-scaled independently; this is a lightweight "
            "diagnostic, not a fitted anomaly detector.",
        ),
    )


def _isolation_forest_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
    random_seed: int,
    expected_contamination: float | None,
) -> Evidence | None:
    columns, usable, z_frame = _numeric_model_matrix(frame, table, target=target)
    if len(columns) < 2 or usable.empty:
        return None
    review_count = _review_count(len(usable), expected_contamination)
    contamination: float | str = (
        expected_contamination if expected_contamination is not None else "auto"
    )
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_seed,
        n_jobs=1,
    )
    model.fit(z_frame)
    scores = pd.Series(-model.decision_function(z_frame), index=usable.index)
    top_records = _model_top_records(scores, z_frame, count=review_count)
    top_rows = {record["row_index"] for record in top_records}

    stability_sets: list[set[str]] = []
    for seed_offset in (1, 2):
        stability_model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_seed + seed_offset,
            n_jobs=1,
        )
        stability_model.fit(z_frame)
        stability_scores = pd.Series(
            -stability_model.decision_function(z_frame), index=usable.index
        )
        stability_sets.append(
            {
                str(index)
                for index in stability_scores.sort_values(ascending=False)
                .head(review_count)
                .index
            }
        )
    mean_jaccard = (
        sum(_jaccard(top_rows, item) for item in stability_sets) / len(stability_sets)
        if stability_sets
        else 1.0
    )
    return Evidence.create(
        kind="anomaly_isolation_forest",
        scope=EvidenceScope(table=table.name, columns=tuple(columns)),
        value={
            "evaluated_row_count": int(len(usable)),
            "feature_count": len(columns),
            "candidate_count": len(top_records),
            "candidate_rate": len(top_records) / len(usable),
            "expected_contamination": expected_contamination,
            "threshold_policy": (
                "expected_contamination"
                if expected_contamination is not None
                else "ranked_review_default"
            ),
            "max_score": float(scores.max() if len(scores) else 0.0),
            "top_records": top_records,
            "stability": {
                "seed_count": 3,
                "mean_top_set_jaccard": float(mean_jaccard),
            },
        },
        method="isolation_forest_ranked_review_v1",
        description=f"Isolation Forest ranked anomaly candidates for {table.name}.",
        confidence=0.78 if mean_jaccard >= 0.5 else 0.66,
        assumptions=(
            "Isolation Forest is a diagnostic review instrument; ranked candidates "
            "are not confirmed anomalies.",
            "Numeric features are robust-scaled and rows with missing modeled "
            "features are skipped.",
        ),
    )


def _local_density_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
    expected_contamination: float | None,
) -> Evidence | None:
    columns, usable, z_frame = _numeric_model_matrix(frame, table, target=target)
    if len(columns) < 2 or usable.empty:
        return None
    if len(usable) < 30 or len(columns) > 12:
        return None
    review_count = _review_count(len(usable), expected_contamination)
    neighbors = min(35, max(10, int(math.sqrt(len(usable)))))
    if neighbors >= len(usable):
        return None
    contamination: float | str = (
        expected_contamination if expected_contamination is not None else "auto"
    )
    model = LocalOutlierFactor(
        n_neighbors=neighbors,
        contamination=contamination,
        metric="minkowski",
        n_jobs=1,
    )
    model.fit_predict(z_frame)
    scores = pd.Series(-model.negative_outlier_factor_, index=usable.index)
    top_records = _model_top_records(scores, z_frame, count=review_count)
    return Evidence.create(
        kind="anomaly_local_density_outlier",
        scope=EvidenceScope(table=table.name, columns=tuple(columns)),
        value={
            "evaluated_row_count": int(len(usable)),
            "feature_count": len(columns),
            "candidate_count": len(top_records),
            "candidate_rate": len(top_records) / len(usable),
            "expected_contamination": expected_contamination,
            "n_neighbors": neighbors,
            "threshold_policy": (
                "expected_contamination"
                if expected_contamination is not None
                else "ranked_review_default"
            ),
            "max_score": float(scores.max() if len(scores) else 0.0),
            "top_records": top_records,
        },
        method="local_outlier_factor_ranked_review_v1",
        description=f"Local-density anomaly candidates for {table.name}.",
        confidence=0.72,
        assumptions=(
            "Local Outlier Factor is sensitive to feature scaling, sparse regions, "
            "and high dimensionality.",
            "Numeric features are robust-scaled and rows with missing modeled "
            "features are skipped.",
        ),
    )


def _candidate_rows(item: Evidence) -> list[str]:
    value = item.value
    if item.kind in {
        "anomaly_univariate_outlier",
        "anomaly_conditional_outlier",
    }:
        return [str(record["row_index"]) for record in value.get("examples", ())]
    if item.kind in {
        "anomaly_multivariate_outlier",
        "anomaly_isolation_forest",
        "anomaly_local_density_outlier",
    }:
        return [str(record["row_index"]) for record in value.get("top_records", ())]
    return []


# --------------------------------------------------------------------------- #
# Row-centric consensus, distribution shape, and relationship context.
#
# These power the analyst-facing report: instead of restating the same rows from
# six detector angles, we name the rows, show their values against the baseline,
# explain why each is unusual, and surface distribution shape so an analyst can
# *see* the data behind every claim.
# --------------------------------------------------------------------------- #


def _identifier_column(table: TableCatalog) -> str | None:
    """The column best suited to label a row in the report (an id if present)."""
    for column in table.columns:
        if "identifier_candidate" in column.roles:
            return column.name
    for column in table.columns:
        if column.name.lower() in {"id", "index"} or column.name.lower().endswith(
            "_id"
        ):
            return column.name
    return None


def _numeric_robust_z(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> tuple[list[str], dict[str, pd.Series], dict[str, float]]:
    """Robust z-scores and medians for meaningful (non-identifier) numeric columns."""
    columns = _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=False
    )
    z_by_column: dict[str, pd.Series] = {}
    median_by_column: dict[str, float] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() < 8:
            continue
        z_by_column[column] = _robust_z(series)
        median_by_column[column] = float(series.median())
    return list(z_by_column), z_by_column, median_by_column


def _as_float(value: Any) -> float:
    """Coerce a pandas/numpy scalar to ``float`` (keeps the type-checker happy)."""
    return float(value)


def _format_number(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.4g}"


def _row_contributors(
    label: Any,
    frame: pd.DataFrame,
    z_by_column: dict[str, pd.Series],
    median_by_column: dict[str, float],
    *,
    min_z: float = _CONTRIBUTOR_MIN_Z,
    limit: int | None = _MAX_CONTRIBUTORS,
) -> list[dict[str, Any]]:
    """The columns that make this row unusual, most extreme first.

    With the defaults this returns only the notable spikes (used for the
    univariate "why" bars). Called with ``min_z=0.0`` and ``limit=None`` it
    returns every numeric column's deviation, sorted by magnitude — the full
    profile behind a multivariate (joint-distance) flag, so the analyst can see
    whether the row is broadly unusual or dominated by a single column.
    """
    scored: list[tuple[float, str, float, float]] = []
    for column, z_series in z_by_column.items():
        if label not in z_series.index:
            continue
        z = z_series.loc[label]
        cell = pd.to_numeric(
            pd.Series([frame.at[label, column]]), errors="coerce"
        ).iloc[0]
        if pd.isna(z) or pd.isna(cell):
            continue
        scored.append((abs(float(z)), column, float(z), float(cell)))
    scored.sort(key=lambda item: item[0], reverse=True)
    contributors: list[dict[str, Any]] = []
    for abs_z, column, z, cell in scored:
        if abs_z < min_z and contributors:
            break
        contributors.append(
            {
                "column": column,
                "value": cell,
                "baseline": median_by_column[column],
                "robust_z": z,
                "direction": "high" if z >= 0 else "low",
            }
        )
        if limit is not None and len(contributors) >= limit:
            break
    return contributors


def _why_text(
    contributors: list[dict[str, Any]], method_count: int, total_methods: int
) -> str:
    agreement = (
        f" Surfaced by {method_count} of {total_methods} checks."
        if total_methods
        else ""
    )
    if not contributors:
        return f"Flagged by {method_count} check(s).{agreement}".strip()
    top = contributors[0]
    column = top["column"]
    value = top["value"]
    baseline = top["baseline"]
    direction = top["direction"]
    value_text = _format_number(value)
    baseline_text = _format_number(baseline)
    # Only use the "N× the typical" phrasing when the value is genuinely extreme;
    # otherwise a small-integer ratio (1 vs 6) reads as alarming when it is not.
    if baseline != 0 and (value / baseline) > 0 and abs(top["robust_z"]) >= 3.0:
        ratio = value / baseline
        magnitude = ratio if ratio >= 1 else 1 / ratio
        if magnitude >= 2:
            rel = "above" if direction == "high" else "below"
            return (
                f"{column} {value_text} is {magnitude:.0f}× the typical "
                f"{baseline_text} ({rel}).{agreement}"
            )
    rel = "above" if direction == "high" else "below"
    return (
        f"{column} {value_text} sits {abs(top['robust_z']):.1f}σ {rel} the "
        f"typical {baseline_text}.{agreement}"
    )


def _conditional_why(
    condition_column: str,
    value_column: str,
    value: float,
    condition_value: float,
    method_count: int,
    total_methods: int,
) -> str:
    agreement = (
        f" Surfaced by {method_count} of {total_methods} checks."
        if total_methods
        else ""
    )
    return (
        f"{value_column} {_format_number(value)} is unusual for its "
        f"{condition_column} peer group (around {condition_column} "
        f"{_format_number(condition_value)}).{agreement}"
    )


def _row_explanations(
    row_label: Any,
    row_str: str,
    frame: pd.DataFrame,
    *,
    contributors: list[dict[str, Any]],
    methods: set[str],
    z_by_column: dict[str, pd.Series],
    median_by_column: dict[str, float],
    multivariate_score: dict[str, float],
    multivariate_threshold: float,
    conditional_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Per-detector evidence behind a review row, so each tag is actually shown.

    Every block is present only when that detector flagged the row, so a
    "multivariate outlier" tag is backed by a visible joint-deviation profile and
    a "conditional outlier" tag by the peer group it stands out from — instead of
    all three tags collapsing to a single univariate sigma bar.
    """
    explanations: dict[str, Any] = {}
    if contributors:
        explanations["univariate"] = {"contributors": contributors}
    if "multivariate outlier" in methods and row_str in multivariate_score:
        # The full profile (no min-z cutoff) shows whether the joint distance is
        # broadly driven or dominated by one column.
        profile = _row_contributors(
            row_label,
            frame,
            z_by_column,
            median_by_column,
            min_z=0.0,
            limit=None,
        )
        explanations["multivariate"] = {
            "score": multivariate_score[row_str],
            "threshold": multivariate_threshold,
            "columns": profile,
        }
    if "conditional outlier" in methods and row_str in conditional_context:
        explanations["conditional"] = conditional_context[row_str]
    return explanations


def _consensus_evidence(
    evidence: Sequence[Evidence],
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> Evidence | None:
    """Collapse all detectors into one ranked, row-centric review list."""
    methods_by_row: dict[str, set[str]] = {}
    threshold_by_row: dict[str, set[str]] = {}
    conditional_strength: dict[str, float] = {}
    conditional_context: dict[str, dict[str, Any]] = {}
    multivariate_score: dict[str, float] = {}
    multivariate_threshold = 0.0
    for item in evidence:
        if item.scope.table != table.name:
            continue
        if (
            item.kind not in _THRESHOLD_SIGNAL_KINDS
            and item.kind not in _CORROBORATING_SIGNAL_KINDS
        ):
            continue
        label = item.kind.replace("anomaly_", "").replace("_", " ")
        for row in _candidate_rows(item):
            methods_by_row.setdefault(row, set()).add(label)
            if item.kind in _THRESHOLD_SIGNAL_KINDS:
                threshold_by_row.setdefault(row, set()).add(label)
        if item.kind == "anomaly_conditional_outlier" and len(item.scope.columns) == 2:
            condition_column, value_column = item.scope.columns
            for example in item.value.get("examples", ()):
                row = str(example["row_index"])
                score = abs(float(example.get("score", 0.0)))
                # Keep the peer group the row stands out from most strongly.
                if score >= conditional_strength.get(row, 0.0):
                    conditional_strength[row] = score
                    conditional_context[row] = {
                        "condition_column": condition_column,
                        "value_column": value_column,
                        **example,
                    }
        elif item.kind == "anomaly_multivariate_outlier":
            multivariate_threshold = float(
                item.value.get("threshold", multivariate_threshold)
            )
            for record in item.value.get("top_records", ()):
                multivariate_score[str(record["row_index"])] = float(
                    record.get("score", 0.0)
                )
    if not methods_by_row:
        return None

    _, z_by_column, median_by_column = _numeric_robust_z(frame, table, target=target)
    index_by_str: dict[str, Any] = {str(label): label for label in frame.index}
    total_methods = len(
        {method for methods in methods_by_row.values() for method in methods}
    )

    rows_payload: list[dict[str, Any]] = []
    for row_str, methods in methods_by_row.items():
        row_label = index_by_str.get(row_str)
        if row_label is None:
            continue
        contributors = _row_contributors(
            row_label, frame, z_by_column, median_by_column
        )
        max_z = max((abs(item["robust_z"]) for item in contributors), default=0.0)
        threshold_methods = threshold_by_row.get(row_str, set())
        corroborating = methods - threshold_methods
        qualifies = (
            (len(threshold_methods) >= 2 and max_z >= _CONSENSUS_MIN_AGREE_Z)
            or max_z >= _UNIVARIATE_FINDING_MIN_ROBUST_Z
            or ("multivariate outlier" in methods and len(corroborating) >= 1)
            or conditional_strength.get(row_str, 0.0) >= _CONSENSUS_STRONG_CONDITIONAL
        )
        if not qualifies:
            continue
        # If the row is globally extreme, explain it column-wise; if it was driven
        # by a conditional (in-context) check, explain the peer-group reason
        # instead of a weak global deviation.
        context = conditional_context.get(row_str)
        if max_z >= 3.0 or context is None:
            why = _why_text(contributors, len(methods), total_methods)
        else:
            why = _conditional_why(
                context["condition_column"],
                context["value_column"],
                _as_float(frame.at[row_label, context["value_column"]]),
                _as_float(frame.at[row_label, context["condition_column"]]),
                len(methods),
                total_methods,
            )
        rows_payload.append(
            {
                "row_index": row_str,
                "method_count": len(methods),
                "methods": sorted(methods),
                "max_abs_robust_z": max_z,
                "values": {
                    str(col): frame.at[row_label, col] for col in frame.columns
                },
                "contributors": contributors,
                "why": why,
                "explanations": _row_explanations(
                    row_label,
                    row_str,
                    frame,
                    contributors=contributors,
                    methods=methods,
                    z_by_column=z_by_column,
                    median_by_column=median_by_column,
                    multivariate_score=multivariate_score,
                    multivariate_threshold=multivariate_threshold,
                    conditional_context=conditional_context,
                ),
            }
        )
    if not rows_payload:
        return None
    rows_payload.sort(key=lambda row: (-row["method_count"], -row["max_abs_robust_z"]))
    rows_payload = rows_payload[:_CONSENSUS_MAX_ROWS]
    confidence = 0.85 if any(row["method_count"] >= 3 for row in rows_payload) else 0.78
    return Evidence.create(
        kind="anomaly_consensus_review",
        scope=EvidenceScope(table=table.name),
        value={
            "evaluated_row_count": int(len(frame)),
            "review_row_count": len(rows_payload),
            "total_detectors": total_methods,
            "id_column": _identifier_column(table),
            "columns": [str(column) for column in frame.columns],
            "rows": rows_payload,
        },
        method="cross_detector_consensus_v1",
        description=f"Cross-detector review rows for {table.name}.",
        confidence=confidence,
        assumptions=(
            "Review rows are prioritized by detector agreement and extremity; they "
            "are candidates for human review, not confirmed anomalies.",
        ),
    )


def _detect_modality(series: pd.Series) -> dict[str, Any]:
    """Robust largest-gap test for a two-population split."""
    values = np.sort(series.to_numpy(dtype="float64"))
    n = len(values)
    result: dict[str, Any] = {"is_multimodal": False, "clusters": []}
    if n < 20:
        return result
    work = values
    log_space = False
    if float(values.min()) > 0:
        skew = _as_float(series.skew()) if n > 2 else 0.0
        spread = values.max() / max(values.min(), 1e-9)
        if abs(skew) >= 1.0 or spread >= 50:
            work = np.log10(values)
            log_space = True
    gaps = np.diff(work)
    positive = gaps[gaps > 0]
    if gaps.size == 0 or positive.size == 0:
        return result
    median_gap = float(np.median(positive))
    work_range = float(work[-1] - work[0])
    max_idx = int(np.argmax(gaps))
    max_gap = float(gaps[max_idx])
    lower_n = max_idx + 1
    upper_n = n - lower_n
    if (
        median_gap > 0
        and max_gap >= _BIMODAL_GAP_RATIO * median_gap
        and work_range > 0
        and max_gap >= _BIMODAL_MIN_RANGE_FRACTION * work_range
        and lower_n >= _BIMODAL_MIN_GROUP
        and upper_n >= _BIMODAL_MIN_GROUP
        and lower_n / n >= _BIMODAL_MIN_FRACTION
        and upper_n / n >= _BIMODAL_MIN_FRACTION
    ):
        boundary_low = float(values[max_idx])
        boundary_high = float(values[max_idx + 1])
        result.update(
            {
                "is_multimodal": True,
                "log_space": log_space,
                "gap": {"low": boundary_low, "high": boundary_high},
                "clusters": [
                    {
                        "min": float(values[0]),
                        "max": boundary_low,
                        "count": lower_n,
                        "fraction": lower_n / n,
                    },
                    {
                        "min": boundary_high,
                        "max": float(values[-1]),
                        "count": upper_n,
                        "fraction": upper_n / n,
                    },
                ],
            }
        )
    return result


def _distribution_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
    flagged: set[str],
) -> list[Evidence]:
    """Per-column distribution shape: histogram, box stats, modality, flagged values."""
    columns, _, _ = _numeric_robust_z(frame, table, target=target)
    index_by_str: dict[str, Any] = {str(label): label for label in frame.index}
    flagged_labels = {index_by_str[item] for item in flagged if item in index_by_str}
    evidence: list[Evidence] = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(series) < 12:
            continue
        values = series.to_numpy(dtype="float64")
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        box = {
            "min": float(series.min()),
            "q1": q1,
            "median": float(series.median()),
            "q3": q3,
            "max": float(series.max()),
            "mean": float(series.mean()),
            "lower_fence": q1 - 1.5 * iqr,
            "upper_fence": q3 + 1.5 * iqr,
        }
        bin_count = min(_HIST_BINS, max(8, len(series) // 2))
        counts, edges = np.histogram(values, bins=bin_count)
        flagged_values = [
            _as_float(frame.at[label, column])
            for label in flagged_labels
            if label in frame.index and pd.notna(frame.at[label, column])
        ]
        modality = _detect_modality(series)
        evidence.append(
            Evidence.create(
                kind="anomaly_distribution_shape",
                scope=EvidenceScope(table=table.name, columns=(column,)),
                value={
                    "column": column,
                    "evaluated_row_count": int(len(series)),
                    "box": box,
                    "histogram": {
                        "counts": [int(count) for count in counts],
                        "edges": [float(edge) for edge in edges],
                    },
                    "flagged_values": flagged_values,
                    "modality": modality,
                },
                method="largest_gap_modality_v1",
                description=f"Distribution shape for {table.name}.{column}.",
                confidence=0.83 if modality["is_multimodal"] else 0.7,
                assumptions=(
                    "Modality is detected with a robust gap split; it is a "
                    "descriptive summary, not a fitted mixture model.",
                ),
            )
        )
    return evidence


def _scatter_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
    flagged: set[str],
) -> Evidence | None:
    """A scatter of the most-associated numeric pair, with flagged rows marked."""
    columns, z_by_column, _ = _numeric_robust_z(frame, table, target=target)
    if len(columns) < 2:
        return None
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.dropna()
    if len(usable) < 8:
        return None
    index_by_str: dict[str, Any] = {str(label): label for label in frame.index}
    flagged_labels = {index_by_str[item] for item in flagged if item in index_by_str}
    ranks = usable.rank()
    correlation = ranks.corr(method="pearson")

    def _best_pair_within(candidates: list[str]) -> tuple[float, str, str] | None:
        best: tuple[float, str, str] | None = None
        for left, right in itertools.combinations(candidates, 2):
            coefficient = abs(float(correlation.loc[left, right]))
            if math.isnan(coefficient):
                continue
            if best is None or coefficient > best[0]:
                best = (coefficient, left, right)
        return best

    # Prefer a scatter that explains the flagged rows: anchor on the column that
    # drives them most, paired with its strongest numeric associate. Fall back to
    # the most-associated pair overall when there are no flagged rows.
    focus: str | None = None
    if flagged_labels:
        scored = {
            column: float(
                sum(
                    abs(z_by_column[column].loc[label])
                    for label in flagged_labels
                    if label in z_by_column[column].index
                    and pd.notna(z_by_column[column].loc[label])
                )
            )
            for column in columns
        }
        focus = max(scored, key=lambda column: scored[column]) if any(
            scored.values()
        ) else None
    if focus is not None:
        partner = max(
            (column for column in columns if column != focus),
            key=lambda column: abs(float(correlation.loc[focus, column]))
            if not math.isnan(correlation.loc[focus, column])
            else -1.0,
            default=None,
        )
        best = (
            (abs(float(correlation.loc[focus, partner])), partner, focus)
            if partner is not None
            else None
        )
    else:
        best = _best_pair_within(columns)
    if best is None:
        return None
    _, x_column, y_column = best
    sample = usable
    if len(sample) > _SCATTER_MAX_POINTS:
        sample = sample.sample(n=_SCATTER_MAX_POINTS, random_state=42).sort_index()
    points = [
        {
            "x": float(sample.at[label, x_column]),
            "y": float(sample.at[label, y_column]),
            "flagged": label in flagged_labels,
        }
        for label in sample.index
    ]
    return Evidence.create(
        kind="anomaly_scatter_pair",
        scope=EvidenceScope(table=table.name, columns=(x_column, y_column)),
        value={
            "x_column": x_column,
            "y_column": y_column,
            "association": best[0],
            "point_count": len(points),
            "flagged_count": sum(1 for point in points if point["flagged"]),
            "points": points,
        },
        method="spearman_top_pair_scatter_v1",
        description=f"Scatter of {x_column} vs {y_column} for {table.name}.",
        confidence=0.7,
    )


def _agreement_evidence(evidence: Sequence[Evidence], *, table: str) -> Evidence | None:
    signal_rows: dict[str, set[str]] = {}
    for item in evidence:
        if item.scope.table != table:
            continue
        rows = set(_candidate_rows(item))
        if rows:
            signal_rows[item.id] = rows
    if len(signal_rows) < 2:
        return None

    row_signals: dict[str, list[str]] = {}
    evidence_by_id = {item.id: item for item in evidence}
    for evidence_id, rows in signal_rows.items():
        signal = evidence_by_id[evidence_id].kind.replace("anomaly_", "")
        for row in rows:
            row_signals.setdefault(row, []).append(signal)
    agreed = {
        row: sorted(signals)
        for row, signals in row_signals.items()
        if len(set(signals)) >= 2
    }
    pairwise_scores: list[float] = []
    ids = sorted(signal_rows)
    for left, right in itertools.combinations(ids, 2):
        pairwise_scores.append(_jaccard(signal_rows[left], signal_rows[right]))
    mean_pairwise = (
        sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0
    )
    return Evidence.create(
        kind="anomaly_detector_agreement",
        scope=EvidenceScope(table=table),
        value={
            "signal_count": len(signal_rows),
            "candidate_count": len(agreed),
            "mean_pairwise_top_set_jaccard": float(mean_pairwise),
            "top_rows": [
                {
                    "row_index": row,
                    "detector_count": len(signals),
                    "signals": signals,
                }
                for row, signals in sorted(
                    agreed.items(), key=lambda item: (-len(item[1]), item[0])
                )[:20]
            ],
        },
        method="ranked_detector_top_set_agreement_v1",
        description=f"Detector agreement across anomaly signals for {table}.",
        confidence=0.8 if agreed else 0.64,
        assumptions=(
            "Agreement is based on ranked review sets, not confirmed anomaly labels.",
        ),
        metadata={"source_evidence_ids": tuple(signal_rows)},
    )


def _conditional_examples(
    pair: pd.DataFrame,
    bins: pd.Series,
    scores: pd.Series,
    candidates: pd.Series,
    *,
    condition_column: str,
    value_column: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Enrich each conditional candidate with the peer-group band it stands out from.

    A conditional (in-context) outlier only makes sense against its peers: the
    rows whose ``condition_column`` falls in the same quantile bin. We carry that
    peer band (median and middle 50%) plus the row's own value so the report can
    *show* where the row sits relative to its context, not just assert a score.
    """
    examples: list[dict[str, Any]] = []
    bin_by_index = dict(bins.items())
    ranked = candidates.abs().sort_values(ascending=False).head(limit)
    for index, score in ranked.items():
        bin_label = bin_by_index[index]
        peers = pair.loc[bins == bin_label, value_column]
        peers = peers[peers.index != index]
        if peers.empty:
            continue
        interval = bin_label if isinstance(bin_label, pd.Interval) else None
        examples.append(
            {
                "row_index": str(index),
                "score": float(score),
                "value": _as_float(pair.at[index, value_column]),
                "condition_value": _as_float(pair.at[index, condition_column]),
                "bin_low": float(interval.left) if interval is not None else None,
                "bin_high": float(interval.right) if interval is not None else None,
                "peer_count": int(len(peers)),
                "peer_median": float(peers.median()),
                "peer_q1": float(peers.quantile(0.25)),
                "peer_q3": float(peers.quantile(0.75)),
                "peer_min": float(peers.min()),
                "peer_max": float(peers.max()),
            }
        )
    return examples


def _conditional_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> list[Evidence]:
    columns = _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=False
    )[:8]
    if len(columns) < 2 or len(frame) < 20:
        return []
    evidence: list[Evidence] = []
    for condition_column, value_column in itertools.permutations(columns, 2):
        pair = frame[[condition_column, value_column]].apply(
            pd.to_numeric, errors="coerce"
        )
        pair = pair.dropna()
        if len(pair) < 20 or pair[condition_column].nunique() < 5:
            continue
        try:
            bins = pd.qcut(
                pair[condition_column],
                q=min(8, max(3, len(pair) // 10)),
                duplicates="drop",
            )
        except ValueError:
            continue
        if bins.nunique() < 3:
            continue
        scores = pd.Series(np.nan, index=pair.index, dtype="float64")
        for _, group in pair.groupby(bins, observed=False):
            if len(group) < 5:
                continue
            scores.loc[group.index] = _robust_z(group[value_column]).abs()
        candidates = scores[scores >= 3.5].dropna()
        if candidates.empty:
            continue
        evidence.append(
            Evidence.create(
                kind="anomaly_conditional_outlier",
                scope=EvidenceScope(
                    table=table.name, columns=(condition_column, value_column)
                ),
                value={
                    "condition_column": condition_column,
                    "value_column": value_column,
                    "evaluated_row_count": int(scores.notna().sum()),
                    "candidate_count": int(len(candidates)),
                    "candidate_rate": len(candidates)
                    / max(1, int(scores.notna().sum())),
                    "max_conditional_score": float(candidates.max()),
                    "examples": _conditional_examples(
                        pair,
                        bins,
                        scores,
                        candidates,
                        condition_column=condition_column,
                        value_column=value_column,
                    ),
                },
                method="quantile_bin_conditional_modified_z_score_v1",
                description=(
                    f"Conditional candidates for {table.name}.{value_column} given "
                    f"{condition_column}."
                ),
                confidence=0.72,
                assumptions=(
                    "Quantile bins approximate local context and may miss nonlinear "
                    "or sparse subgroup effects.",
                ),
            )
        )
    return evidence


def _rare_category_evidence(frame: pd.DataFrame, table: TableCatalog) -> list[Evidence]:
    evidence: list[Evidence] = []
    row_count = len(frame)
    if row_count < 20:
        return evidence
    for column in _categorical_columns(frame, table):
        counts = frame[column].value_counts(dropna=False)
        rare = counts[counts <= max(1, math.floor(row_count * 0.01))]
        if rare.empty:
            continue
        evidence.append(
            Evidence.create(
                kind="anomaly_rare_category",
                scope=EvidenceScope(table=table.name, columns=(column,)),
                value={
                    "evaluated_row_count": row_count,
                    "rare_value_count": int(len(rare)),
                    "rare_row_count": int(rare.sum()),
                    "examples": [
                        {"value": _category_value(value), "count": int(count)}
                        for value, count in rare.head(10).items()
                    ],
                },
                method="low_frequency_category_scan_v1",
                description=f"Rare category candidates for {table.name}.{column}.",
                confidence=0.68,
                assumptions=(
                    "Rare values can be valid long-tail cases; domain review is "
                    "required.",
                ),
            )
        )
    return evidence


def _rare_category_combination_evidence(
    frame: pd.DataFrame, table: TableCatalog
) -> list[Evidence]:
    evidence: list[Evidence] = []
    row_count = len(frame)
    if row_count < 50:
        return evidence
    columns = _categorical_columns(frame, table)[:8]
    if len(columns) < 2:
        return evidence
    threshold = max(1, math.floor(row_count * 0.005))
    for left, right in itertools.combinations(columns, 2):
        pair = frame[[left, right]].copy()
        if (
            pair[left].nunique(dropna=False) < 2
            or pair[right].nunique(dropna=False) < 2
        ):
            continue
        counts = pair.value_counts(dropna=False)
        rare = counts[counts <= threshold]
        if rare.empty:
            continue
        examples: list[dict[str, Any]] = []
        for values, count in rare.head(10).items():
            left_value, right_value = (
                values if isinstance(values, tuple) else (values, None)
            )
            examples.append(
                {
                    left: _category_value(left_value),
                    right: _category_value(right_value),
                    "count": int(count),
                }
            )
        evidence.append(
            Evidence.create(
                kind="anomaly_rare_category_combination",
                scope=EvidenceScope(table=table.name, columns=(left, right)),
                value={
                    "evaluated_row_count": row_count,
                    "rare_combination_count": int(len(rare)),
                    "rare_row_count": int(rare.sum()),
                    "frequency_threshold": threshold,
                    "examples": examples,
                },
                method="low_frequency_category_pair_scan_v1",
                description=(
                    f"Rare category-pair candidates for {table.name}.{left} + {right}."
                ),
                confidence=0.66,
                assumptions=(
                    "Rare category combinations can be valid sparse segments; "
                    "domain review is required.",
                ),
            )
        )
    return evidence


def _label_evidence(frame: pd.DataFrame, table: str, target: str) -> Evidence | None:
    if target not in frame.columns:
        return None
    counts = frame[target].value_counts(dropna=False)
    if counts.empty:
        return None
    rare_count = int(counts.min())
    rare_rate = rare_count / len(frame) if len(frame) else 0.0
    return Evidence.create(
        kind="anomaly_label_summary",
        scope=EvidenceScope(table=table, columns=(target,)),
        value={
            "class_count": int(len(counts)),
            "counts": {str(value): int(count) for value, count in counts.items()},
            "minority_count": rare_count,
            "minority_rate": rare_rate,
        },
        method="target_frequency_summary_v1",
        description=f"Optional anomaly label summary for {table}.{target}.",
        confidence=1.0,
    )


def _findings_and_steps(
    evidence: Sequence[Evidence],
) -> tuple[list[Finding], list[TransformationStep]]:
    findings: list[Finding] = []
    steps: list[TransformationStep] = []
    # Row-centric reporting: the consensus list and distribution-shape findings
    # lead, and per-detector results (multivariate / isolation forest / LOF /
    # conditional / agreement) feed the consensus instead of becoming their own
    # findings, so the same rows are not restated six ways.
    for item in evidence:
        value = item.value
        if item.kind == "anomaly_consensus_review" and value["rows"]:
            count = value["review_row_count"]
            findings.append(
                Finding.create(
                    title=f"{count} row(s) to review in {item.scope.table}",
                    summary=(
                        f"{count} row(s) stand out across multiple checks — each is "
                        "listed below with its values, the typical baseline, and why "
                        "it was flagged."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Open the flagged rows below and decide whether each is a data "
                        "error, a rare valid case, or a separate regime."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_flagged_rows",
                    table=item.scope.table or "",
                    columns=(),
                    parameters={
                        "review_row_count": count,
                        "method": item.method,
                    },
                    rationale=(
                        "Rows surfaced by multiple independent checks are the highest "
                        "priority for human review."
                    ),
                    evidence_ids=(item.id,),
                    risk="medium",
                )
            )
        elif (
            item.kind == "anomaly_distribution_shape"
            and value["modality"]["is_multimodal"]
        ):
            column = value["column"]
            lower, upper = value["modality"]["clusters"]
            findings.append(
                Finding.create(
                    title=f"{item.scope.table}.{column} looks like two populations",
                    summary=(
                        "Values split into two separated groups — a lower cluster of "
                        f"{lower['count']:,} row(s) and an upper cluster of "
                        f"{upper['count']:,} row(s), with a clear gap between. That is "
                        "two regimes, not one distribution with a few tail outliers; "
                        "see the distribution chart for the exact split."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Check whether the two groups differ by unit, source, or "
                        "definition before treating either as outliers."
                    ),
                )
            )
        elif (
            item.kind == "anomaly_univariate_outlier"
            and value["candidate_count"]
            and (
                value["max_abs_robust_z"] >= _UNIVARIATE_FINDING_MIN_ROBUST_Z
                or value["candidate_rate"] >= _UNIVARIATE_FINDING_MIN_RATE
            )
        ):
            column = item.scope.columns[0]
            findings.append(
                Finding.create(
                    title=f"Univariate tail candidates in {item.scope.table}.{column}",
                    summary=(
                        f"{value['candidate_count']:,} row(s) "
                        f"({value['candidate_rate']:.1%}) sit outside robust tail "
                        "thresholds."
                    ),
                    severity="high" if value["candidate_rate"] >= 0.05 else "medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Review the flagged rows before capping, filtering, or "
                        "modeling them separately."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_outlier_candidates",
                    table=item.scope.table or "",
                    columns=item.scope.columns,
                    parameters={
                        "candidate_count": value["candidate_count"],
                        "method": item.method,
                    },
                    rationale=(
                        "Tail candidates may represent errors, rare valid cases, or "
                        "a separate regime."
                    ),
                    evidence_ids=(item.id,),
                    risk="medium",
                )
            )
        elif item.kind == "anomaly_rare_category":
            column = item.scope.columns[0]
            findings.append(
                Finding.create(
                    title=f"Rare categories in {item.scope.table}.{column}",
                    summary=(
                        f"{value['rare_value_count']:,} rare value(s) cover "
                        f"{value['rare_row_count']:,} row(s)."
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm rare categories are valid before grouping or "
                        "treating them as anomalies."
                    ),
                )
            )
        elif item.kind == "anomaly_rare_category_combination":
            left, right = item.scope.columns
            findings.append(
                Finding.create(
                    title=(
                        f"Rare category combinations in {item.scope.table}: "
                        f"{left} + {right}"
                    ),
                    summary=(
                        f"{value['rare_combination_count']:,} rare combination(s) "
                        f"cover {value['rare_row_count']:,} row(s)."
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm whether these sparse category pairs are valid "
                        "segments before grouping or treating them as anomalies."
                    ),
                )
            )
        elif item.kind == "anomaly_label_summary" and value["minority_rate"] <= 0.05:
            findings.append(
                Finding.create(
                    title=f"Rare labeled anomaly class in {item.scope.table}",
                    summary=(
                        f"The minority label has {value['minority_count']:,} row(s) "
                        f"({value['minority_rate']:.1%})."
                    ),
                    severity="medium",
                    confidence=1.0,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Use stratified, group-aware review and avoid accuracy as "
                        "the main metric."
                    ),
                )
            )
    return sort_findings(findings), steps


def _artifact(evidence: Sequence[Evidence]) -> Artifact | None:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        value = item.value
        if item.kind in {
            "anomaly_univariate_outlier",
            "anomaly_multivariate_outlier",
            "anomaly_conditional_outlier",
            "anomaly_isolation_forest",
            "anomaly_local_density_outlier",
            "anomaly_detector_agreement",
        }:
            rows.append(
                {
                    "signal": item.kind.replace("anomaly_", "").replace("_", " "),
                    "table": item.scope.table,
                    "columns": " + ".join(item.scope.columns),
                    "candidate_rows": value.get("candidate_count", 0),
                    "rate": f"{value.get('candidate_rate', 0.0):.1%}",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "anomaly_rare_category":
            rows.append(
                {
                    "signal": "rare category",
                    "table": item.scope.table,
                    "columns": " + ".join(item.scope.columns),
                    "candidate_rows": value["rare_row_count"],
                    "rate": "low frequency",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "anomaly_rare_category_combination":
            rows.append(
                {
                    "signal": "rare category combination",
                    "table": item.scope.table,
                    "columns": " + ".join(item.scope.columns),
                    "candidate_rows": value["rare_row_count"],
                    "rate": "low frequency",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
    rows = [row for row in rows if row["candidate_rows"]]
    if not rows:
        return None
    return Artifact.create(
        kind="metric_table",
        title="Anomaly candidate signals",
        data={
            "columns": [
                {"key": "signal", "label": "Signal"},
                {"key": "table", "label": "Table"},
                {"key": "columns", "label": "Columns"},
                {"key": "candidate_rows", "label": "Candidate rows"},
                {"key": "rate", "label": "Rate"},
                {"key": "confidence", "label": "Confidence"},
            ],
            "rows": rows[:30],
        },
        evidence_ids=tuple(item.id for item in evidence),
        metadata={
            "description": (
                "How the review rows were found: each detector's contribution. "
                "Candidates are not confirmed anomalies."
            )
        },
    )


def _verdict(evidence: Sequence[Evidence]) -> str | None:
    """One plain-language headline: the single most useful thing to say first.

    Signal over noise — lead with the strongest reframing (two populations beats
    a pile of "outliers"), else the row-review count with a concrete example,
    else nothing so the hero can fall back to the neutral summary.
    """
    multimodal = [
        str(item.value["column"])
        for item in evidence
        if item.kind == "anomaly_distribution_shape"
        and item.value["modality"]["is_multimodal"]
    ]
    if multimodal:
        if len(multimodal) == 1:
            subject, verb = multimodal[0], "splits"
        else:
            subject = f"{', '.join(multimodal[:-1])} and {multimodal[-1]}"
            verb = "each split"
        return (
            f"{subject} {verb} into two distinct populations — two regimes to "
            "separate, not scattered outliers."
        )
    review = next(
        (
            item
            for item in evidence
            if item.kind == "anomaly_consensus_review" and item.value["rows"]
        ),
        None,
    )
    if review is not None:
        count = review.value["review_row_count"]
        total = review.value["evaluated_row_count"]
        top_why = review.value["rows"][0]["why"].split(" Surfaced by")[0].strip()
        return (
            f"{count} of {total:,} row(s) warrant review across multiple "
            f"independent checks — the clearest: {top_why}"
        )
    return None


def _anomaly_summary(
    table_count: int,
    findings: list[Finding],
    *,
    has_warnings: bool,
) -> str:
    """Lead with the strongest candidate signal instead of a raw count."""
    suffix = " Sampling or recoverable caveats apply." if has_warnings else ""
    if not findings:
        return (
            f"No anomaly review candidates cleared the diagnostic thresholds "
            f"across {table_count} table(s).{suffix}"
        )
    top = findings[0]
    return (
        f"Anomaly review across {table_count} table(s): top signal — "
        f"{top.title}. {len(findings)} prioritized candidate signal(s).{suffix}"
    )


def anomaly_detection_dataset(
    tables: Mapping[str, pd.DataFrame],
    catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
    table: str | None = None,
    target: str | None = None,
    expected_contamination: float | None = None,
    callbacks: tuple[EventCallback, ...] = (),
) -> AnalysisResult:
    """Run deterministic anomaly-detection diagnostics."""
    _validate_expected_contamination(expected_contamination)
    emit(
        callbacks,
        Event(EventKind.RUN_STARTED, "Anomaly diagnostics started.", stage="anomaly"),
    )
    warnings: list[AnalysisWarning] = []
    sampling: list[SamplingRecord] = []
    selected_tables = [catalog.table(table)] if table else list(catalog.tables)
    evidence: list[Evidence] = []

    for table_catalog in selected_tables:
        frame = tables[table_catalog.name]
        sampled = _sample_frame(
            frame,
            table=table_catalog.name,
            config=config,
            warnings=warnings,
            sampling=sampling,
        )
        active_target = target or context.target
        if active_target and active_target in sampled.columns:
            label = _label_evidence(sampled, table_catalog.name, active_target)
            if label is not None:
                evidence.append(label)
        evidence.extend(
            _univariate_evidence(sampled, table_catalog, target=active_target)
        )
        multivariate = _multivariate_evidence(
            sampled, table_catalog, target=active_target
        )
        if multivariate is not None:
            evidence.append(multivariate)
        isolation_forest = _isolation_forest_evidence(
            sampled,
            table_catalog,
            target=active_target,
            random_seed=config.random_seed,
            expected_contamination=expected_contamination,
        )
        if isolation_forest is not None:
            evidence.append(isolation_forest)
        local_density = _local_density_evidence(
            sampled,
            table_catalog,
            target=active_target,
            expected_contamination=expected_contamination,
        )
        if local_density is not None:
            evidence.append(local_density)
        evidence.extend(
            _conditional_evidence(sampled, table_catalog, target=active_target)
        )
        evidence.extend(_rare_category_evidence(sampled, table_catalog))
        evidence.extend(_rare_category_combination_evidence(sampled, table_catalog))
        agreement = _agreement_evidence(evidence, table=table_catalog.name)
        if agreement is not None:
            evidence.append(agreement)
        # Consolidate every detector into one ranked, row-centric review list,
        # then describe distribution shape and the strongest pair so the report
        # can show the data behind each claim.
        consensus = _consensus_evidence(
            evidence, sampled, table_catalog, target=active_target
        )
        flagged_rows: set[str] = set()
        if consensus is not None:
            evidence.append(consensus)
            flagged_rows = {row["row_index"] for row in consensus.value["rows"]}
        evidence.extend(
            _distribution_evidence(
                sampled, table_catalog, target=active_target, flagged=flagged_rows
            )
        )
        scatter = _scatter_evidence(
            sampled, table_catalog, target=active_target, flagged=flagged_rows
        )
        if scatter is not None:
            evidence.append(scatter)

    for item in evidence:
        emit(
            callbacks,
            Event(
                EventKind.EVIDENCE_CREATED,
                item.description,
                stage="evidence",
                data={"evidence_id": item.id, "kind": item.kind},
            ),
        )
    findings, steps = _findings_and_steps(evidence)
    artifacts = tuple(item for item in (_artifact(evidence),) if item is not None)

    if not selected_tables or catalog.row_count == 0:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = "There is insufficient data for anomaly diagnostics."
    elif not evidence:
        status = AnalysisStatus.NO_MEANINGFUL_STRUCTURE
        summary = (
            "No usable anomaly-detection signals were available in the selected data."
        )
    elif warnings:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        summary = _anomaly_summary(len(selected_tables), findings, has_warnings=True)
    else:
        status = AnalysisStatus.COMPLETED
        summary = _anomaly_summary(len(selected_tables), findings, has_warnings=False)

    result = AnalysisResult(
        goal="anomaly_detection",
        status=status,
        summary=summary,
        catalog=catalog,
        findings=tuple(findings),
        evidence=tuple(evidence),
        artifacts=artifacts,
        assumptions=(
            *context.assumptions,
            "Unsupervised anomaly diagnostics identify review candidates, not "
            "confirmed anomalies.",
        ),
        warnings=tuple(warnings),
        sampling=tuple(sampling),
        transformation_plan=TransformationPlan(tuple(steps)),
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
            "selected_table": table,
            "target": target or context.target,
            "expected_contamination": expected_contamination,
            "candidate_signals": len(findings),
            "verdict": _verdict(evidence),
        },
    )
    emit(
        callbacks,
        Event(
            EventKind.RUN_COMPLETED,
            result.summary,
            stage="anomaly",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
