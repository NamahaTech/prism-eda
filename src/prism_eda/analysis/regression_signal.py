"""What carries signal, and what is leaking or merely duplicated?

Three questions that look similar and are not. A feature can be *associated*
with the target (useful), *redundant* with another feature (not wrong, but it
makes coefficients unreadable), or *leaking* the target (fatal, and the thing
this module exists to catch).

Two deliberate choices here. Associations are measured three ways — linear,
monotone, and binned — because a feature that a Pearson correlation calls
useless can carry most of the signal in a curve. And redundancy is reported as
concrete pairs plus the VIF values, never as a universal ``VIF > 10`` verdict:
that cutoff has no theoretical basis and firing on it would be exactly the kind
of number-at-your-face this library exists to avoid.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from prism_eda.analysis._regression import bin_edges, safe_ratio
from prism_eda.catalog.models import TableCatalog
from prism_eda.evidence.models import Evidence, EvidenceScope

# A feature that explains essentially all of the target is not a good feature.
# R-squared is already measured against the mean baseline, so unlike the
# classification screen this needs no separate lift guard.
_LEAKAGE_R2 = 0.98
_NEAR_PERFECT_R2 = 0.999

# Redundancy is a claim about two named columns, so the bar is high enough that
# the pair really is interchangeable.
_REDUNDANCY_CORRELATION = 0.95
_MAX_REDUNDANCY_PAIRS = 10

# A curved relationship is worth naming when the binned fit finds real signal
# that the straight line missed.
_NONLINEARITY_GAP = 0.10
_MIN_BINNED_SIGNAL = 0.05

_ASSOCIATION_BINS = 10
_MIN_ASSOCIATION_ROWS = 20

#: Shortest name fragment allowed to imply a leak. Matching raw substrings makes
#: a one- or two-letter target match almost anything — a target called ``y``
#: appears inside ``x1_copy`` — so name evidence is taken from whole tokens and
#: only from tokens long enough to mean something.
_MIN_NAME_TOKEN = 3


def _name_tokens(name: str) -> set[str]:
    """Lowercase word tokens of a column name, split on separators and case."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return {token for token in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if token}


def _shares_target_name(target: str, feature: str) -> bool:
    """True when a feature's name contains a meaningful token of the target's."""
    target_tokens = {
        token for token in _name_tokens(target) if len(token) >= _MIN_NAME_TOKEN
    }
    return bool(target_tokens & _name_tokens(feature))


def _correlation_ratio(categories: pd.Series, target: pd.Series) -> float | None:
    """Eta-squared: how much of the target's variance the grouping explains."""
    data = pd.DataFrame({"group": categories.astype("string"), "target": target})
    data = data.dropna()
    if len(data) < _MIN_ASSOCIATION_ROWS or data["group"].nunique() < 2:
        return None
    overall = data["target"].mean()
    total_ss = float(((data["target"] - overall) ** 2).sum())
    if total_ss <= 0:
        return None
    between_ss = sum(
        len(group) * float((group["target"].mean() - overall) ** 2)
        for _, group in data.groupby("group", dropna=False)
    )
    return max(0.0, min(1.0, between_ss / total_ss))


def _binned_eta_squared(feature: pd.Series, target: pd.Series) -> float | None:
    """Eta-squared over feature deciles — catches curves a correlation misses."""
    data = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(data) < _MIN_ASSOCIATION_ROWS:
        return None
    values = data["feature"].to_numpy(dtype="float64")
    edges = bin_edges(values, _ASSOCIATION_BINS)
    if edges.size < 3:
        return None
    bins = pd.Series(
        np.clip(np.digitize(values, edges[1:-1]), 0, edges.size - 2),
        index=data.index,
    )
    return _correlation_ratio(bins, data["target"])


def _numeric_association(
    feature: pd.Series, target: pd.Series
) -> dict[str, Any] | None:
    data = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(data) < _MIN_ASSOCIATION_ROWS or data["feature"].nunique() < 2:
        return None
    pearson = data["feature"].corr(data["target"])
    spearman = data["feature"].corr(data["target"], method="spearman")
    pearson = float(pearson) if pd.notna(pearson) else 0.0
    spearman = float(spearman) if pd.notna(spearman) else 0.0
    binned = _binned_eta_squared(data["feature"], data["target"])
    linear_r2 = pearson**2
    gap = (binned - linear_r2) if binned is not None else None
    return {
        "pearson": pearson,
        "spearman": spearman,
        "linear_r_squared": linear_r2,
        "binned_eta_squared": binned,
        "nonlinearity_gap": gap,
        "is_nonlinear": bool(
            gap is not None
            and gap >= _NONLINEARITY_GAP
            and binned is not None
            and binned >= _MIN_BINNED_SIGNAL
        ),
        "evaluated_row_count": int(len(data)),
    }


def identifier_and_leakage_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    target_series: pd.Series,
) -> list[Evidence]:
    """Columns that must not become features, and why.

    Runs before every other feature-side stage so that the probe, the redundancy
    scan, and the influence design all inherit the same exclusions. A probe
    trained on a leak reports a near-perfect fit and buries the one finding that
    actually matters.
    """
    evidence: list[Evidence] = []
    profile_by_name = {column.name: column for column in table.columns}
    for column in frame.columns:
        name = str(column)
        if name == target:
            continue
        profile = profile_by_name.get(name)
        if profile is None:
            continue
        series = frame[column]

        if "identifier_candidate" in profile.roles:
            evidence.append(
                Evidence.create(
                    kind="regression_identifier_feature",
                    scope=EvidenceScope(table=table.name, columns=(name,)),
                    value={
                        "feature": name,
                        "unique_count": profile.unique_count,
                        "unique_rate": profile.unique_rate,
                    },
                    method="identifier_role_exclusion_v1",
                    description=(
                        f"Identifier-like feature {table.name}.{name} is "
                        "near-unique per row."
                    ),
                    confidence=0.9,
                    assumptions=(
                        "Identifier-like columns label rows rather than explain a "
                        "numeric outcome and should be excluded from features.",
                    ),
                )
            )
            continue

        explained: float | None = None
        metric = ""
        if ptypes.is_numeric_dtype(series.dtype) and not ptypes.is_bool_dtype(
            series.dtype
        ):
            association = _numeric_association(series, target_series)
            if association is not None:
                explained = max(
                    association["linear_r_squared"],
                    association["binned_eta_squared"] or 0.0,
                )
                metric = "univariate_r_squared"
        elif profile.semantic_type in {"categorical", "boolean", "text"}:
            explained = _correlation_ratio(series, target_series)
            metric = "correlation_ratio"

        name_overlap = _shares_target_name(target, name)
        suspicious = explained is not None and explained >= _LEAKAGE_R2
        if not (suspicious or name_overlap):
            continue
        near_perfect = explained is not None and explained >= _NEAR_PERFECT_R2
        evidence.append(
            Evidence.create(
                kind="regression_leakage_candidate",
                scope=EvidenceScope(table=table.name, columns=(name, target)),
                value={
                    "feature": name,
                    "target": target,
                    "explained_variance": explained,
                    "metric": metric or "name_match",
                    "name_contains_target": name_overlap,
                    "near_perfect": near_perfect,
                },
                method="deterministic_regression_leakage_screen_v1",
                description=(
                    f"Potential target leakage candidate {table.name}.{name}."
                ),
                confidence=0.92 if suspicious else 0.64,
                assumptions=(
                    "A feature that reproduces the target almost exactly is "
                    "either derived from it or recorded after it. Confirm the "
                    "value is available at prediction time before keeping it.",
                ),
            )
        )
    return evidence


def association_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    target_series: pd.Series,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[Evidence]:
    """Per-feature association, measured linearly, monotonically, and binned."""
    evidence: list[Evidence] = []
    for name in numeric_features:
        association = _numeric_association(frame[name], target_series)
        if association is None:
            continue
        evidence.append(
            Evidence.create(
                kind="regression_feature_association",
                scope=EvidenceScope(table=table.name, columns=(name, target)),
                value={"feature": name, "target": target, **association},
                method="linear_monotone_binned_association_v1",
                description=(
                    f"Numeric feature-target association for {table.name}.{name}."
                ),
                confidence=0.8,
            )
        )
    for name in categorical_features:
        score = _correlation_ratio(frame[name], target_series)
        if score is None:
            continue
        evidence.append(
            Evidence.create(
                kind="regression_categorical_association",
                scope=EvidenceScope(table=table.name, columns=(name, target)),
                value={
                    "feature": name,
                    "target": target,
                    "correlation_ratio": score,
                    "metric": "correlation_ratio",
                },
                method="correlation_ratio_association_v1",
                description=(
                    f"Categorical feature-target association for {table.name}.{name}."
                ),
                confidence=0.78,
            )
        )
    return evidence


def redundancy_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    numeric_features: list[str],
) -> Evidence | None:
    """Interchangeable feature pairs, with VIF reported but never thresholded."""
    if len(numeric_features) < 2:
        return None
    design = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
    design = design.loc[:, design.nunique(dropna=True) > 1]
    if design.shape[1] < 2:
        return None
    design = design.fillna(design.median(numeric_only=True))
    correlations = design.corr(method="pearson").abs()

    pairs: list[dict[str, Any]] = []
    columns = list(correlations.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = correlations.loc[left, right]
            if pd.isna(value) or float(value) < _REDUNDANCY_CORRELATION:
                continue
            pairs.append(
                {
                    "left": str(left),
                    "right": str(right),
                    "abs_correlation": float(value),
                }
            )
    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    pairs = pairs[:_MAX_REDUNDANCY_PAIRS]

    vif: list[dict[str, Any]] = []
    condition_number: float | None = None
    if design.shape[0] > design.shape[1] + 1:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        matrix = np.column_stack(
            [np.ones(len(design)), design.to_numpy(dtype="float64")]
        )
        for index, name in enumerate(design.columns, start=1):
            try:
                value = float(variance_inflation_factor(matrix, index))
            except (ValueError, ZeroDivisionError):  # pragma: no cover - singular
                continue
            if np.isfinite(value):
                vif.append({"feature": str(name), "vif": value})
        vif.sort(key=lambda item: item["vif"], reverse=True)
        scaled = design.to_numpy(dtype="float64")
        spread = scaled.std(axis=0)
        spread[spread == 0] = 1.0
        singular = np.linalg.svd(
            (scaled - scaled.mean(axis=0)) / spread, compute_uv=False
        )
        condition_number = safe_ratio(float(singular[0]), float(singular[-1]))

    if not pairs and condition_number is None:
        return None
    return Evidence.create(
        kind="regression_redundancy",
        scope=EvidenceScope(
            table=table.name, columns=tuple(design.columns.astype(str))
        ),
        value={
            "redundant_pairs": pairs,
            "variance_inflation": vif,
            "condition_number": condition_number,
            "evaluated_feature_count": int(design.shape[1]),
        },
        method="pairwise_redundancy_and_vif_v1",
        description=f"Feature redundancy and multicollinearity in {table.name}.",
        confidence=0.84,
        assumptions=(
            "Redundant features are not an error; they make individual "
            "coefficients unstable and unreadable, while leaving predictions "
            "largely intact.",
            "VIF and the condition number are reported as measurements. There is "
            "no universal cutoff at which either becomes a defect.",
        ),
    )
