"""Shared plumbing for the clustering recipe.

One decision shapes this whole module: **categorical columns do not enter the
distance.**

Euclidean distance over one-hot columns is not a meaningful measure of
similarity — it silently asserts that every pair of categories is exactly
sqrt(2) apart, that a five-category column deserves five times the weight of a
two-category one, and that "west versus east" is commensurable with "spent 400
more". Tools that quietly one-hot everything and hand the result to k-means
produce clusters that are real numbers computed from a meaningless geometry.

So the distance is built from numeric features only, and the categorical columns
are put to the use they are actually good for: describing the groups afterwards.
"Segment 2 is 80% west" is a genuinely useful sentence; "west is 1.41 units from
east" is not. When categorical columns are the real signal, the report says so
and points at Gower distance or k-prototypes rather than pretending.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from prism_eda.catalog.models import DatasetCatalog, TableCatalog
from prism_eda.config import AnalysisConfig, AnalysisMode
from prism_eda.results import AnalysisWarning, SamplingRecord

_ROW_BUDGETS = {
    AnalysisMode.QUICK: 5_000,
    AnalysisMode.STANDARD: 20_000,
    AnalysisMode.DEEP: 50_000,
}

#: Rows needed before any clustering claim is worth making.
MIN_ROWS = 30

#: Numeric features admitted to the distance. Past this the distance is
#: dominated by dimensionality itself rather than by any of the features.
MAX_FEATURES = 30

#: Distinct values above which a categorical column is not worth profiling.
MAX_PROFILE_CATEGORIES = 30

#: Rows sampled for the pairwise-distance diagnostics, which are quadratic.
MAX_DISTANCE_ROWS = 1_500

#: Cluster counts searched, and the smallest cluster worth having.
MIN_K = 2
MAX_K = 10
MIN_ROWS_PER_K = 20

#: Segments listed with a full profile before the list stops being read.
MAX_SEGMENTS_PROFILED = 12

#: Representative rows shown per segment.
MAX_REPRESENTATIVES = 3


@dataclass(slots=True)
class FeatureSet:
    """Which columns became the distance, which describe it, and which were cut."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)

    def exclude(self, name: str, reason: str, detail: str) -> None:
        self.excluded.append({"feature": name, "reason": reason, "detail": detail})


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
            code="sampled_clustering_analysis",
            message=(
                f"{table} has {len(frame):,} rows; clustering diagnostics were run "
                f"on a deterministic {budget:,}-row sample."
            ),
            table=table,
        )
    )
    sampling.append(
        SamplingRecord(
            operation="clustering_analysis",
            source_rows=len(frame),
            sampled_rows=budget,
            strategy="deterministic_pandas_sample",
            seed=config.random_seed,
            reason="row_count_exceeds_mode_budget",
            limitations=(
                "A small group present in the full data may be absent from the "
                "sample, and a group found in the sample may be a sampling "
                "artifact.",
            ),
        )
    )
    return sampled


def resolve_table(
    catalog: DatasetCatalog,
    tables: Mapping[str, pd.DataFrame],
    *,
    table: str | None,
    warnings: list[AnalysisWarning],
) -> TableCatalog | None:
    """Pick the table to cluster, or explain why none could be chosen."""
    if table is not None:
        if table not in tables:
            warnings.append(
                AnalysisWarning(
                    code="clustering_table_not_found",
                    message=f"Table {table!r} was not found.",
                    table=table,
                )
            )
            return None
        return catalog.table(table)
    if len(catalog.tables) == 1:
        return catalog.tables[0]
    warnings.append(
        AnalysisWarning(
            code="clustering_table_ambiguous",
            message=(
                f"The dataset has {len(catalog.tables)} tables; pass table= to "
                "choose which one to cluster."
            ),
        )
    )
    return None


def admit_features(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    features: Sequence[str] | None,
    warnings: list[AnalysisWarning],
) -> FeatureSet:
    """Decide which columns build the distance and which merely describe it.

    Every exclusion carries a reason, because a silently dropped column is
    indistinguishable from a column that contributed nothing.
    """
    selected = FeatureSet()
    profile_by_name = {column.name: column for column in table.columns}
    requested = list(features) if features is not None else None
    if requested is not None:
        missing = [name for name in requested if name not in frame.columns]
        for name in missing:
            warnings.append(
                AnalysisWarning(
                    code="clustering_feature_not_found",
                    message=f"Feature {name!r} was not found in {table.name!r}.",
                    table=table.name,
                    column=name,
                )
            )
        requested = [name for name in requested if name in frame.columns]

    candidates = (
        requested
        if requested is not None
        else [str(column) for column in frame.columns]
    )
    explicit = requested is not None

    for name in candidates:
        column = frame[name]
        profile = profile_by_name.get(name)
        distinct = int(column.nunique(dropna=True))

        if distinct <= 1:
            selected.exclude(
                name,
                "constant",
                "Every row holds the same value, so it adds no distance.",
            )
            continue
        if column.isna().all():
            selected.exclude(name, "all_missing", "Every value is missing.")
            continue
        # An explicitly requested identifier is still an identifier. Clustering
        # on a unique key gives every point the same distance to every other.
        if profile is not None and "identifier_candidate" in profile.roles:
            selected.exclude(
                name,
                "identifier",
                "Near-unique per row; it would add a dimension in which every "
                "point is equidistant from every other.",
            )
            continue

        if ptypes.is_bool_dtype(column.dtype):
            selected.categorical.append(name)
            continue
        if ptypes.is_numeric_dtype(column.dtype):
            selected.numeric.append(name)
            continue
        if ptypes.is_datetime64_any_dtype(column.dtype):
            selected.exclude(
                name,
                "datetime",
                "A timestamp needs an explicit derived feature (age, recency) "
                "before it means anything as a coordinate.",
            )
            continue
        if distinct > MAX_PROFILE_CATEGORIES:
            selected.exclude(
                name,
                "high_cardinality",
                f"{distinct:,} distinct values is too many to profile per group.",
            )
            continue
        selected.categorical.append(name)

    if len(selected.numeric) > MAX_FEATURES:
        for name in selected.numeric[MAX_FEATURES:]:
            selected.exclude(
                name,
                "feature_cap",
                f"Only the first {MAX_FEATURES} numeric features enter the "
                "distance; beyond that the geometry is dominated by "
                "dimensionality itself.",
            )
        selected.numeric = selected.numeric[:MAX_FEATURES]

    if explicit and not selected.numeric:
        warnings.append(
            AnalysisWarning(
                code="clustering_no_usable_features",
                message=("None of the requested features are usable numeric columns."),
                table=table.name,
            )
        )
    return selected


def build_matrix(
    frame: pd.DataFrame,
    numeric: list[str],
    *,
    scale: bool = True,
) -> tuple[np.ndarray, pd.Index]:
    """Impute and (optionally) standardize the numeric features.

    Returns the matrix and the row index it corresponds to. Scaling is optional
    because whether the answer *changes* without it is itself a diagnostic.
    """
    if not numeric:
        return np.empty((0, 0)), frame.index[:0]
    block = frame[numeric].apply(pd.to_numeric, errors="coerce")
    usable = block.dropna(how="all")
    if usable.empty:
        return np.empty((0, 0)), frame.index[:0]
    imputed = SimpleImputer(strategy="median").fit_transform(usable)
    matrix = StandardScaler().fit_transform(imputed) if scale else np.asarray(imputed)
    return np.asarray(matrix, dtype="float64"), usable.index


def distance_sample(
    matrix: np.ndarray, *, seed: int, limit: int = MAX_DISTANCE_ROWS
) -> np.ndarray:
    """A deterministic row subsample, because pairwise work is quadratic."""
    if matrix.shape[0] <= limit:
        return matrix
    rng = np.random.default_rng(seed)
    picked = rng.choice(matrix.shape[0], size=limit, replace=False)
    return matrix[np.sort(picked)]


def candidate_k_values(rows: int) -> list[int]:
    """Cluster counts worth evaluating for this many rows.

    Capped so that every candidate could in principle hold a group large enough
    to mean something; searching k=10 on 60 rows measures noise.
    """
    highest = min(MAX_K, max(MIN_K, rows // MIN_ROWS_PER_K))
    if highest < MIN_K:
        return []
    return list(range(MIN_K, highest + 1))


def as_float(value: Any) -> float:
    return float(value)


#: Significant digits kept on a metric before it is banked as evidence.
#:
#: scikit-learn's k-means reduces in parallel, so the order of the floating-point
#: additions varies between runs and results can differ in the last bit or two —
#: 571.870907495445 one run, 571.8709074954448 the next, from identical labels.
#: Evidence IDs hash their values, so that meaningless difference would produce a
#: different ID every run and break the reproducibility contract for a quantity
#: that did not actually change. Twelve significant digits is far beyond any
#: meaningful precision for a clustering score and absorbs the noise entirely.
_STABLE_DIGITS = 12


def stable(value: float | None) -> float | None:
    """Round a metric to a precision that survives parallel reduction order."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return number
    return float(f"{number:.{_STABLE_DIGITS}g}")
