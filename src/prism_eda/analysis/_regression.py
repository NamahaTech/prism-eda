"""Shared plumbing for the regression recipe.

The recipe is split by question the way the baseline profile is:
:mod:`regression_target` asks what shape the thing being predicted is in,
:mod:`regression_signal` asks what carries signal and what is leaking or
redundant, and :mod:`regression_probe` asks what happens when a model is
actually fitted. This module holds what all three need, so that the feature set
the leakage screen excludes is provably the same feature set the probe trains
on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prism_eda.analysis._numeric import HIST_BINS
from prism_eda.catalog.models import DatasetCatalog, TableCatalog
from prism_eda.config import AnalysisConfig, AnalysisMode
from prism_eda.evidence.models import Evidence
from prism_eda.results import AnalysisWarning, SamplingRecord

_ROW_BUDGETS = {
    AnalysisMode.QUICK: 25_000,
    AnalysisMode.STANDARD: 100_000,
    AnalysisMode.DEEP: 250_000,
}

#: Features admitted to the probe and to the influence design matrix. Leverage
#: and Cook's distance need an invertible design, and both cost grows with
#: width, so the cap is shared rather than set per stage.
MAX_PROBE_FEATURES = 30

#: Rows carried into the review table. This is a list a person reads, not an
#: export, so it stops at the length someone will actually work through.
MAX_REVIEW_ROWS = 20

#: A numeric target with fewer distinct values than this is probably a class
#: label, a rating, or a count. Regression still runs — a 5-point rating is a
#: legitimate regression target — but the report says so rather than silently
#: treating it as continuous.
MIN_CONTINUOUS_DISTINCT = 10

#: Below this many usable rows the residual, influence, and heteroscedasticity
#: diagnostics are reading noise rather than structure.
MIN_PROBE_ROWS = 30


def row_budget(mode: AnalysisMode | str) -> int:
    return _ROW_BUDGETS[AnalysisMode(mode)]


def sample_frame(
    frame: pd.DataFrame,
    *,
    table: str,
    config: AnalysisConfig,
    warnings: list[AnalysisWarning],
    sampling: list[SamplingRecord],
) -> pd.DataFrame:
    """Cut the frame to the mode's row budget, recording that it happened."""
    budget = row_budget(config.mode)
    if config.sampling == "disabled" or len(frame) <= budget:
        return frame
    sampled = frame.sample(n=budget, random_state=config.random_seed).sort_index()
    warnings.append(
        AnalysisWarning(
            code="sampled_regression_analysis",
            message=(
                f"{table} has {len(frame):,} rows; regression diagnostics were run "
                f"on a deterministic {budget:,}-row sample."
            ),
            table=table,
        )
    )
    sampling.append(
        SamplingRecord(
            operation="regression_analysis",
            source_rows=len(frame),
            sampled_rows=budget,
            strategy="deterministic_pandas_sample",
            seed=config.random_seed,
            reason="row_count_exceeds_mode_budget",
            limitations=(
                "Rare extreme target values and sparse feature regions may be "
                "absent from the sample, so influence and extrapolation "
                "diagnostics describe the sample rather than every row.",
            ),
        )
    )
    return sampled


def resolve_table(
    catalog: DatasetCatalog,
    tables: Mapping[str, pd.DataFrame],
    *,
    table: str | None,
    target: str | None,
    warnings: list[AnalysisWarning],
) -> tuple[TableCatalog | None, str | None]:
    """Find the one table holding a usable numeric target, or explain why not."""
    if target is None:
        warnings.append(
            AnalysisWarning(
                code="regression_target_required",
                message="Regression analysis requires a numeric target column.",
            )
        )
        return None, None

    if table is not None:
        if table not in tables:
            warnings.append(
                AnalysisWarning(
                    code="regression_table_not_found",
                    message=f"Table {table!r} was not found.",
                    table=table,
                )
            )
            return None, target
        if target not in tables[table].columns:
            warnings.append(
                AnalysisWarning(
                    code="regression_target_not_found",
                    message=f"Target column {target!r} was not found in {table!r}.",
                    table=table,
                    column=target,
                )
            )
            return None, target
        resolved = catalog.table(table)
    else:
        matches = [
            item for item in catalog.tables if target in tables[item.name].columns
        ]
        if not matches:
            warnings.append(
                AnalysisWarning(
                    code="regression_target_not_found",
                    message=f"Target column {target!r} was not found in any table.",
                    column=target,
                )
            )
            return None, target
        if len(matches) > 1:
            warnings.append(
                AnalysisWarning(
                    code="regression_target_ambiguous",
                    message=(
                        f"Target column {target!r} appears in multiple tables; "
                        "pass table=."
                    ),
                    column=target,
                )
            )
            return None, target
        resolved = matches[0]

    series = tables[resolved.name][target]
    if not ptypes.is_numeric_dtype(series.dtype) or ptypes.is_bool_dtype(series.dtype):
        # Coercing a text column would invent a target. Say what it is instead.
        warnings.append(
            AnalysisWarning(
                code="regression_target_not_numeric",
                message=(
                    f"Target column {target!r} is {series.dtype}, not numeric. "
                    "Regression needs a continuous target; use classification() "
                    "for a label."
                ),
                table=resolved.name,
                column=target,
            )
        )
        return None, target
    return resolved, target


def numeric_target(frame: pd.DataFrame, target: str) -> pd.Series:
    """Return the target as float, without touching the caller's frame."""
    return pd.to_numeric(frame[target], errors="coerce").astype("float64")


def leakage_feature_names(evidence: list[Evidence]) -> set[str]:
    return {
        item.scope.columns[0]
        for item in evidence
        if item.kind == "regression_leakage_candidate" and item.scope.columns
    }


def identifier_feature_names(evidence: list[Evidence]) -> set[str]:
    return {
        item.scope.columns[0]
        for item in evidence
        if item.kind == "regression_identifier_feature" and item.scope.columns
    }


def feature_groups(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    *,
    max_categories: int,
    excluded_features: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split usable features into (numeric, categorical, excluded).

    Anything already identified as a leak or an identifier is excluded here, so
    the probe never sees it. A probe that trains on the leak would report a
    near-perfect fit and bury the very finding that matters.
    """
    profile_by_name = {column.name: column for column in table.columns}
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[str] = []
    for column in frame.columns:
        name = str(column)
        profile = profile_by_name.get(name)
        if name == target or profile is None:
            continue
        if name in excluded_features or "identifier_candidate" in profile.roles:
            excluded.append(name)
            continue
        if ptypes.is_bool_dtype(frame[column].dtype):
            categorical.append(name)
            continue
        if ptypes.is_numeric_dtype(frame[column].dtype):
            numeric.append(name)
            continue
        if profile.semantic_type in {"categorical", "boolean", "text"}:
            if profile.unique_count is None or profile.unique_count <= max_categories:
                categorical.append(name)
            else:
                excluded.append(name)

    selected_numeric = numeric[:MAX_PROBE_FEATURES]
    remaining = MAX_PROBE_FEATURES - len(selected_numeric)
    selected_categorical = categorical[: max(0, remaining)]
    excluded.extend(numeric[len(selected_numeric) :])
    excluded.extend(categorical[len(selected_categorical) :])
    return selected_numeric, selected_categorical, sorted(set(excluded))


def preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Impute, scale, and encode — fitted inside each fold by the caller."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                drop="first",
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers)


def probe_folds(row_count: int, mode: AnalysisMode | str) -> int | None:
    if row_count < MIN_PROBE_ROWS:
        return None
    max_folds = 3 if AnalysisMode(mode) == AnalysisMode.QUICK else 5
    return min(max_folds, max(2, row_count // 10))


def bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Quantile edges that stay strictly increasing on heaped/discrete data."""
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size < 2:
        return np.array([])
    return edges


def distribution_payload(
    values: np.ndarray,
    *,
    column: str,
    flagged: Sequence[float] = (),
) -> dict[str, Any]:
    """Chart data in the exact shape ``reporting.charts.histogram_svg`` reads.

    Reports render from banked evidence with no DataFrame in scope, so the bins
    and the box five-number summary are computed here, once, and travel with the
    evidence.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return {}
    q1 = float(np.quantile(finite, 0.25))
    q3 = float(np.quantile(finite, 0.75))
    iqr = q3 - q1
    bin_count = min(HIST_BINS, max(6, finite.size // 2))
    counts, edges = np.histogram(finite, bins=bin_count)
    return {
        "column": column,
        "evaluated_row_count": int(finite.size),
        "box": {
            "min": float(finite.min()),
            "q1": q1,
            "median": float(np.median(finite)),
            "q3": q3,
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "lower_fence": q1 - 1.5 * iqr,
            "upper_fence": q3 + 1.5 * iqr,
        },
        "histogram": {
            "counts": [int(count) for count in counts],
            "edges": [float(edge) for edge in edges],
        },
        "flagged_values": [float(value) for value in flagged],
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0 or not np.isfinite(denominator):
        return None
    ratio = numerator / denominator
    return float(ratio) if np.isfinite(ratio) else None


def as_float(value: Any) -> float:
    return float(value)
