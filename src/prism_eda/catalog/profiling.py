"""Deterministic baseline catalog generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from prism_eda._serialization import to_jsonable
from prism_eda.catalog.fingerprints import (
    FINGERPRINT_METHOD,
    dataframe_fingerprint,
    dataset_fingerprint,
)
from prism_eda.catalog.models import (
    ColumnCatalog,
    DatasetCatalog,
    SourceInfo,
    TableCatalog,
)

# A low unique *ratio* alone is not enough to call a column categorical: on a
# million-row table 5% is still tens of thousands of distinct values (names,
# free text). The absolute cap bounds how many distinct values a column may
# have and still be treated as an enumerable category set.
CATEGORICAL_UNIQUE_CAP = 200
# Categorical columns above this distinct count get an analyst-facing warning:
# downstream category-based checks stop using them around this cardinality.
HIGH_CARDINALITY_WARNING_THRESHOLD = 100


def _semantic_type(series: pd.Series, unique_count: int | None) -> str:
    if ptypes.is_bool_dtype(series.dtype):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if ptypes.is_numeric_dtype(series.dtype):
        return "numeric"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    non_null_count = int(series.notna().sum())
    if unique_count is not None and non_null_count:
        if unique_count <= 50 or (
            unique_count <= CATEGORICAL_UNIQUE_CAP
            and unique_count / non_null_count <= 0.05
        ):
            return "categorical"
    return "text"


def _roles(
    name: str,
    semantic_type: str,
    unique_rate: float | None,
    missing_rate: float,
) -> tuple[str, ...]:
    normalized = name.lower().strip()
    roles: list[str] = []
    if semantic_type == "datetime" or any(
        token in normalized for token in ("date", "time", "timestamp")
    ):
        roles.append("timestamp_candidate")
    if unique_rate is not None and unique_rate >= 0.98 and missing_rate == 0:
        if (
            normalized == "id"
            or normalized.endswith("_id")
            or "identifier" in normalized
        ):
            roles.append("identifier_candidate")
    if semantic_type == "numeric":
        roles.append("measure_candidate")
    elif semantic_type in {"categorical", "boolean"}:
        roles.append("dimension_candidate")
    elif semantic_type == "text":
        roles.append("free_text_candidate")
    return tuple(roles)


def _unique_count(series: pd.Series) -> tuple[int | None, str | None]:
    try:
        return int(series.nunique(dropna=True)), None
    except (TypeError, ValueError) as error:
        return None, f"Distinct count failed: {error}"


def _top_values(series: pd.Series, limit: int = 5) -> tuple[dict[str, Any], ...]:
    try:
        counts = series.value_counts(dropna=False).head(limit)
    except (TypeError, ValueError):
        counts = series.astype(str).value_counts(dropna=False).head(limit)
    return tuple(
        {"value": to_jsonable(value), "count": int(count)}
        for value, count in counts.items()
    )


def _statistics(series: pd.Series, semantic_type: str) -> dict[str, Any]:
    non_null = series.dropna()
    if non_null.empty:
        return {}
    if semantic_type == "numeric":
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if numeric.empty:
            return {}
        return to_jsonable(
            {
                "min": numeric.min(),
                "max": numeric.max(),
                "mean": numeric.mean(),
                "median": numeric.median(),
                "std": numeric.std(ddof=1) if len(numeric) > 1 else None,
                "q1": numeric.quantile(0.25),
                "q3": numeric.quantile(0.75),
            }
        )
    if semantic_type == "datetime":
        return to_jsonable({"min": non_null.min(), "max": non_null.max()})
    if semantic_type == "text":
        lengths = non_null.astype(str).str.len()
        return to_jsonable(
            {
                "min_length": lengths.min(),
                "max_length": lengths.max(),
                "mean_length": lengths.mean(),
            }
        )
    return {}


def profile_column(name: str, series: pd.Series) -> ColumnCatalog:
    row_count = len(series)
    non_null_count = int(series.notna().sum())
    missing_count = row_count - non_null_count
    missing_rate = missing_count / row_count if row_count else 0.0
    unique_count, unique_warning = _unique_count(series)
    unique_rate = (
        unique_count / non_null_count
        if unique_count is not None and non_null_count
        else None
    )
    semantic_type = _semantic_type(series, unique_count)
    warnings = [unique_warning] if unique_warning else []
    if missing_rate >= 0.5:
        warnings.append("At least half of the values are missing.")
    if unique_count == 1 and non_null_count:
        warnings.append("Column is constant among non-null values.")
    if (
        semantic_type == "categorical"
        and unique_count is not None
        and unique_count > HIGH_CARDINALITY_WARNING_THRESHOLD
    ):
        rate_note = f" ({unique_rate:.0%} unique)" if unique_rate is not None else ""
        warnings.append(
            f"High cardinality for a categorical column: {unique_count:,} "
            f"distinct values{rate_note} — likely not a true category."
        )

    return ColumnCatalog(
        name=name,
        physical_type=str(series.dtype),
        semantic_type=semantic_type,
        roles=_roles(name, semantic_type, unique_rate, missing_rate),
        row_count=row_count,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_rate=missing_rate,
        unique_count=unique_count,
        unique_rate=unique_rate,
        statistics=_statistics(series, semantic_type),
        top_values=_top_values(series),
        warnings=tuple(warnings),
    )


def profile_table(
    name: str,
    frame: pd.DataFrame,
    source: SourceInfo,
) -> TableCatalog:
    warnings: list[str] = []
    try:
        duplicate_row_count = int(frame.duplicated().sum())
    except (TypeError, ValueError) as error:
        duplicate_row_count = None
        warnings.append(f"Duplicate-row detection failed: {error}")

    columns = tuple(
        profile_column(str(column), frame[column]) for column in frame.columns
    )
    return TableCatalog(
        name=name,
        row_count=len(frame),
        column_count=len(frame.columns),
        memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
        duplicate_row_count=duplicate_row_count,
        fingerprint=dataframe_fingerprint(frame),
        fingerprint_method=FINGERPRINT_METHOD,
        source=source,
        columns=columns,
        warnings=tuple(warnings),
    )


def build_catalog(
    tables: Mapping[str, pd.DataFrame],
    sources: Mapping[str, SourceInfo],
) -> DatasetCatalog:
    table_catalogs = tuple(
        profile_table(name, frame, sources[name]) for name, frame in tables.items()
    )
    fingerprints = {table.name: table.fingerprint for table in table_catalogs}
    return DatasetCatalog(
        fingerprint=dataset_fingerprint(fingerprints),
        fingerprint_method=FINGERPRINT_METHOD,
        table_count=len(table_catalogs),
        row_count=sum(table.row_count for table in table_catalogs),
        column_count=sum(table.column_count for table in table_catalogs),
        tables=table_catalogs,
    )
