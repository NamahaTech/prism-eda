"""A source backed by pandas DataFrames already in memory.

It answers the same questions a database source does, exactly, so everything
exploration can find in a warehouse it can also find in a folder of CSV files.
It is also the reference implementation the SQL sources are tested against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from prism_eda.sources.base import (
    ColumnInfo,
    ColumnStats,
    Source,
    TableInfo,
    check_rate,
    in_hash_sample,
    normalise_value,
    safe_name,
)

_GRAINS = {"hour": "h", "day": "D", "week": "W", "month": "M", "year": "Y"}


def frame_kind(series: pd.Series) -> str:
    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "datetime"
    if ptypes.is_integer_dtype(dtype):
        return "integer"
    if ptypes.is_float_dtype(dtype):
        return "float"
    if ptypes.is_numeric_dtype(dtype):
        return "float"
    if isinstance(dtype, pd.CategoricalDtype) or ptypes.is_string_dtype(dtype):
        return "string"
    if dtype == np.dtype("O"):
        return "string"
    return "other"


def canonical(series: pd.Series) -> pd.Series:
    """Canonical text per value (``None`` for null and blank)."""
    return series.map(normalise_value)


class FrameSource(Source):
    """Named DataFrames presented through the source contract."""

    kind = "dataframe"

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        name: str = "data",
        locations: Mapping[str, str | None] | None = None,
    ) -> None:
        self.name = safe_name(name)
        self._frames = dict(frames)
        self._locations = dict(locations or {})
        self._canonical: dict[tuple[str, str], pd.Series] = {}

    # -- helpers -----------------------------------------------------------

    def _frame(self, table: TableInfo) -> pd.DataFrame:
        return self._frames[table.name]

    def _canon(self, table: TableInfo, column: str) -> pd.Series:
        key = (table.name, column)
        if key not in self._canonical:
            self._canonical[key] = canonical(self._frame(table)[column])
        return self._canonical[key]

    # -- inventory ---------------------------------------------------------

    def tables(self) -> list[TableInfo]:
        return [
            TableInfo(
                source=self.name,
                schema=None,
                name=name,
                kind="dataframe",
                approx_rows=len(frame),
                bytes=int(frame.memory_usage(index=False, deep=True).sum()),
                comment=self._locations.get(name),
            )
            for name, frame in self._frames.items()
        ]

    def columns(self, table: TableInfo) -> list[ColumnInfo]:
        frame = self._frame(table)
        return [
            ColumnInfo(
                name=str(column),
                native_type=str(frame[column].dtype),
                kind=frame_kind(frame[column]),
                nullable=True,
            )
            for column in frame.columns
            if isinstance(column, str)
        ]

    def row_count(self, table: TableInfo) -> int:
        return len(self._frame(table))

    # -- column profile ----------------------------------------------------

    def column_stats(
        self, table: TableInfo, columns: Sequence[ColumnInfo]
    ) -> dict[str, ColumnStats]:
        frame = self._frame(table)
        rows = len(frame)
        stats: dict[str, ColumnStats] = {}
        for column in columns:
            series = frame[column.name]
            present = series.dropna()
            try:
                distinct: int | None = int(present.nunique())
            except TypeError:
                distinct = int(present.astype(str).nunique())
            extra: dict[str, Any] = {}
            if column.kind in {"integer", "float"} and len(present):
                numeric = pd.to_numeric(present, errors="coerce").astype("float64")
                finite = numeric[np.isfinite(numeric)]
                if len(finite):
                    extra = {
                        "minimum": float(finite.min()),
                        "maximum": float(finite.max()),
                        "mean": float(finite.mean()),
                        "std": float(finite.std(ddof=0)) if len(finite) > 1 else 0.0,
                        "quantiles": {
                            label: float(finite.quantile(q))
                            for label, q in (
                                ("p5", 0.05),
                                ("q1", 0.25),
                                ("median", 0.5),
                                ("q3", 0.75),
                                ("p95", 0.95),
                                ("p99", 0.99),
                            )
                        },
                        "zeros": int((finite == 0).sum()),
                        "negatives": int((finite < 0).sum()),
                    }
            elif column.kind in {"datetime", "date"} and len(present):
                extra = {"minimum": present.min(), "maximum": present.max()}
            elif column.kind == "string" and len(present):
                text = present.astype(str)
                lengths = text.str.len()
                extra = {
                    "blank": int((text.str.strip(" ") == "").sum()),
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "mean_length": float(lengths.mean()),
                }
            stats[column.name] = ColumnStats(
                name=column.name,
                rows=rows,
                non_null=int(len(present)),
                distinct=distinct,
                distinct_exact=True,
                **extra,
            )
        return stats

    def top_values(
        self, table: TableInfo, columns: Sequence[ColumnInfo], limit: int
    ) -> dict[str, list[tuple[Any, int]]]:
        frame = self._frame(table)
        answer: dict[str, list[tuple[Any, int]]] = {}
        for column in columns:
            series = frame[column.name].dropna()
            try:
                counts = series.value_counts().head(limit)
            except TypeError:
                counts = series.astype(str).value_counts().head(limit)
            answer[column.name] = [
                (value, int(count)) for value, count in counts.items()
            ]
        return answer

    def time_histogram(
        self, table: TableInfo, column: ColumnInfo, grain: str
    ) -> list[tuple[str, int]]:
        series = pd.to_datetime(self._frame(table)[column.name], errors="coerce")
        series = series.dropna()
        if series.empty:
            return []
        if getattr(series.dt, "tz", None) is not None:
            series = series.dt.tz_localize(None)
        periods = series.dt.to_period(_GRAINS[grain])
        counts = periods.value_counts().sort_index()
        starts = pd.PeriodIndex(counts.index).start_time
        return [
            (start.isoformat(), int(count))
            for start, count in zip(starts, counts.to_numpy(), strict=True)
        ]

    # -- rows --------------------------------------------------------------

    def sample(
        self,
        table: TableInfo,
        columns: Sequence[ColumnInfo],
        rows: int,
        *,
        total_rows: int,
        seed: int,
        hash_columns: Sequence[ColumnInfo] = (),
        where: tuple[ColumnInfo, Sequence[Any]] | None = None,
    ) -> pd.DataFrame:
        frame = self._frame(table)[[column.name for column in columns]]
        if where is not None:
            column, values = where
            wanted = {normalise_value(value) for value in values}
            frame = frame[self._canon(table, column.name).isin(wanted).to_numpy()]
        if len(frame) > rows:
            frame = frame.sample(n=rows, random_state=seed).sort_index()
        return frame.reset_index(drop=True).copy()

    # -- values ------------------------------------------------------------

    def distinct_sample(
        self,
        table: TableInfo,
        requests: Mapping[str, tuple[ColumnInfo, int]],
        limit: int,
    ) -> dict[str, list[str]]:
        answer: dict[str, list[str]] = {}
        for name, (column, rate) in requests.items():
            check_rate(rate)
            series = self._frame(table)[column.name]
            canon = self._canon(table, column.name)
            keep = canon.notna()
            raw = series[keep].astype(str)
            canon = canon[keep]
            if rate > 1:
                chosen = canon.map(lambda value: in_hash_sample(value, rate))
                raw = raw[chosen.to_numpy()]
            answer[name] = sorted(set(raw))[:limit]
        return answer

    def probe(
        self,
        table: TableInfo,
        requests: Mapping[str, tuple[ColumnInfo, Sequence[str]]],
    ) -> dict[str, set[str]]:
        answer: dict[str, set[str]] = {}
        for name, (column, values) in requests.items():
            present = set(self._canon(table, column.name).dropna().unique())
            answer[name] = present.intersection(values)
        return answer

    # -- links -------------------------------------------------------------

    def _pairs(
        self, table: TableInfo, left: ColumnInfo, right: ColumnInfo
    ) -> pd.DataFrame:
        pairs = pd.DataFrame(
            {
                "left": self._canon(table, left.name),
                "right": self._canon(table, right.name),
            }
        )
        return pairs.dropna()

    def pair_profile(
        self, table: TableInfo, left: ColumnInfo, right: ColumnInfo
    ) -> dict[str, Any]:
        pairs = self._pairs(table, left, right)
        return {
            "rows": int(len(pairs)),
            "equal_rows": int((pairs["left"] == pairs["right"]).sum()),
            "distinct_pairs": int(len(pairs.drop_duplicates())),
            "distinct_pairs_exact": True,
        }

    def fanout_sample(
        self,
        table: TableInfo,
        key: ColumnInfo,
        others: Sequence[ColumnInfo],
        rate: int,
        limit: int,
    ) -> pd.DataFrame:
        check_rate(rate)
        data = {"key": self._canon(table, key.name)}
        for other in others:
            data[other.name] = self._canon(table, other.name)
        frame = pd.DataFrame(data)
        frame = frame[frame["key"].notna()]
        if rate > 1:
            keys = pd.Series(frame["key"].unique())
            chosen = set(keys[keys.map(lambda value: in_hash_sample(value, rate))])
            frame = frame[frame["key"].isin(chosen)]
        grouped = frame.groupby("key", sort=True)
        answer = grouped.size().rename("rows").to_frame()
        for other in others:
            answer[other.name] = grouped[other.name].nunique()
        return answer.reset_index().head(limit)

    def neighbours(
        self,
        table: TableInfo,
        left: ColumnInfo,
        right: ColumnInfo,
        values: Sequence[str],
        limit: int,
    ) -> list[tuple[str, str]]:
        pairs = self._pairs(table, left, right)
        pairs = pairs[pairs["left"].isin(set(values))].drop_duplicates()
        pairs = pairs.sort_values(["left", "right"]).head(limit)
        return list(zip(pairs["left"], pairs["right"], strict=True))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "tables": len(self._frames)}
