"""Shared plumbing for the time-series recipe.

Two ideas hold this module together.

The first is that **a time series has two representations and they are not
interchangeable**. The raw rows are what the data actually contains — duplicated
timestamps, absent days, uneven spacing and all — and every hygiene claim must be
computed there, because those claims are exact statements about defects. The
regular grid is a reconstruction: gaps filled, duplicates collapsed, spacing
forced even. Decomposition, autocorrelation, and stationarity can only run on the
grid, so everything derived from it inherits an assumption and says so.

The second is that frequency inference has to survive real data. ``pd.infer_freq``
gives up entirely on an index with one missing day, which is most real indexes, so
the dominant spacing is taken from the modal gap between observations and calendar
frequencies (monthly, quarterly, yearly) are recognized by range rather than by an
exact delta they never have.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from prism_eda.catalog.models import DatasetCatalog, TableCatalog
from prism_eda.evidence.models import Evidence
from prism_eda.results import AnalysisWarning, SamplingRecord

#: Observations needed before any structural claim is worth making.
MIN_OBSERVATIONS = 20

#: Complete seasonal cycles STL needs before a seasonal component means anything.
MIN_SEASONAL_CYCLES = 2

#: Points banked into a chart series before deterministic thinning.
MAX_CHART_POINTS = 1_500

#: Entities profiled individually in a panel before the list stops being read.
MAX_PANEL_ENTITIES = 40


@dataclass(frozen=True, slots=True)
class Frequency:
    """The inferred spacing of a series, and what it implies for seasonality."""

    #: pandas offset alias usable with ``date_range`` / ``reindex``.
    alias: str
    #: Human label for the report.
    label: str
    #: Fixed spacing, or ``None`` for calendar frequencies whose length varies.
    delta: pd.Timedelta | None
    #: Observations in the dominant seasonal cycle (7 for daily/weekly, 12 for
    #: monthly/yearly, 24 for hourly/daily). ``None`` when none is natural.
    seasonal_period: int | None
    #: Name of that cycle, for prose.
    seasonal_label: str | None
    #: Share of observed gaps that match the dominant spacing exactly.
    regularity: float


# Calendar frequencies never have a constant delta, so they are matched by the
# range their gaps actually fall in rather than by an exact value.
_CALENDAR_RANGES = (
    (pd.Timedelta(days=28), pd.Timedelta(days=31), "MS", "monthly", 12, "yearly"),
    (pd.Timedelta(days=89), pd.Timedelta(days=92), "QS", "quarterly", 4, "yearly"),
    (pd.Timedelta(days=365), pd.Timedelta(days=366), "YS", "yearly", None, None),
)

# Fixed frequencies, matched exactly against the modal gap.
_FIXED = (
    (pd.Timedelta(seconds=1), "s", "per-second", 60, "per-minute"),
    (pd.Timedelta(minutes=1), "min", "per-minute", 60, "hourly"),
    (pd.Timedelta(minutes=15), "15min", "15-minute", 96, "daily"),
    (pd.Timedelta(minutes=30), "30min", "half-hourly", 48, "daily"),
    (pd.Timedelta(hours=1), "h", "hourly", 24, "daily"),
    (pd.Timedelta(days=1), "D", "daily", 7, "weekly"),
    (pd.Timedelta(days=7), "W", "weekly", 52, "yearly"),
)


def infer_frequency(timestamps: pd.Series) -> Frequency | None:
    """Dominant spacing of a timestamp column, tolerant of gaps and duplicates.

    ``pd.infer_freq`` requires a perfectly regular index and returns ``None`` the
    moment one observation is missing, which is the normal case rather than the
    exceptional one. The modal gap between distinct sorted timestamps survives
    both gaps and duplicates.
    """
    ordered = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna()
    unique = pd.Series(ordered.unique()).sort_values()
    if len(unique) < 3:
        return None
    gaps = unique.diff().dropna()
    if gaps.empty:
        return None
    modal = gaps.mode()
    if modal.empty:
        return None
    dominant = pd.Timedelta(modal.iloc[0])
    if dominant <= pd.Timedelta(0):
        return None
    regularity = float((gaps == dominant).mean())

    for low, high, alias, label, period, season in _CALENDAR_RANGES:
        # A calendar frequency is recognized when most gaps land in its band,
        # not when the modal gap happens to equal one particular month length.
        if float(((gaps >= low) & (gaps <= high)).mean()) >= 0.6:
            return Frequency(
                alias=alias,
                label=label,
                delta=None,
                seasonal_period=period,
                seasonal_label=season,
                regularity=float(((gaps >= low) & (gaps <= high)).mean()),
            )

    for delta, alias, label, period, season in _FIXED:
        if dominant == delta:
            return Frequency(
                alias=alias,
                label=label,
                delta=delta,
                seasonal_period=period,
                seasonal_label=season,
                regularity=regularity,
            )

    # An unrecognized but consistent spacing is still usable as a grid; it just
    # carries no assumption about which cycle would be seasonal.
    return Frequency(
        alias=None,  # type: ignore[arg-type]
        label=f"every {_describe_delta(dominant)}",
        delta=dominant,
        seasonal_period=None,
        seasonal_label=None,
        regularity=regularity,
    )


def _describe_delta(delta: pd.Timedelta) -> str:
    seconds = delta.total_seconds()
    if seconds >= 86_400:
        return f"{seconds / 86_400:.4g} day(s)"
    if seconds >= 3_600:
        return f"{seconds / 3_600:.4g} hour(s)"
    if seconds >= 60:
        return f"{seconds / 60:.4g} minute(s)"
    return f"{seconds:.4g} second(s)"


def frequency_offset(frequency: Frequency) -> Any:
    """The offset ``date_range`` should step by for this frequency."""
    return frequency.alias if frequency.alias else frequency.delta


def resolve_series(
    catalog: DatasetCatalog,
    tables: Mapping[str, pd.DataFrame],
    *,
    table: str | None,
    timestamp: str | None,
    value: str | None,
    entity_id: str | None,
    warnings: list[AnalysisWarning],
) -> tuple[TableCatalog | None, str | None, str | None, str | None]:
    """Pick the table, the time column, the value column, and the entity column.

    The time column is inferred when the table holds exactly one datetime
    column. Anything else is ambiguous, and guessing between two date columns —
    ``ordered_at`` and ``shipped_at``, say — would silently change every number
    in the report, so ambiguity is reported rather than resolved.
    """
    if value is None:
        warnings.append(
            AnalysisWarning(
                code="time_series_value_required",
                message="Time-series analysis requires a numeric value column.",
            )
        )
        return None, timestamp, None, entity_id

    if table is not None:
        if table not in tables:
            warnings.append(
                AnalysisWarning(
                    code="time_series_table_not_found",
                    message=f"Table {table!r} was not found.",
                    table=table,
                )
            )
            return None, timestamp, value, entity_id
        resolved = catalog.table(table)
    else:
        matches = [
            item for item in catalog.tables if value in tables[item.name].columns
        ]
        if not matches:
            warnings.append(
                AnalysisWarning(
                    code="time_series_value_not_found",
                    message=f"Value column {value!r} was not found in any table.",
                    column=value,
                )
            )
            return None, timestamp, value, entity_id
        if len(matches) > 1:
            warnings.append(
                AnalysisWarning(
                    code="time_series_value_ambiguous",
                    message=(
                        f"Value column {value!r} appears in multiple tables; "
                        "pass table=."
                    ),
                    column=value,
                )
            )
            return None, timestamp, value, entity_id
        resolved = matches[0]

    frame = tables[resolved.name]
    if value not in frame.columns:
        warnings.append(
            AnalysisWarning(
                code="time_series_value_not_found",
                message=f"Value column {value!r} was not found in {resolved.name!r}.",
                table=resolved.name,
                column=value,
            )
        )
        return None, timestamp, value, entity_id
    if not ptypes.is_numeric_dtype(frame[value].dtype) or ptypes.is_bool_dtype(
        frame[value].dtype
    ):
        warnings.append(
            AnalysisWarning(
                code="time_series_value_not_numeric",
                message=(
                    f"Value column {value!r} is {frame[value].dtype}, not numeric."
                ),
                table=resolved.name,
                column=value,
            )
        )
        return None, timestamp, value, entity_id

    resolved_timestamp = timestamp
    if resolved_timestamp is None:
        candidates = [
            column.name
            for column in resolved.columns
            if column.semantic_type == "datetime" and column.name in frame.columns
        ]
        if len(candidates) == 1:
            resolved_timestamp = candidates[0]
        elif not candidates:
            warnings.append(
                AnalysisWarning(
                    code="time_series_timestamp_not_found",
                    message=(
                        f"{resolved.name} has no datetime column; pass timestamp= "
                        "naming the time axis."
                    ),
                    table=resolved.name,
                )
            )
            return None, None, value, entity_id
        else:
            warnings.append(
                AnalysisWarning(
                    code="time_series_timestamp_ambiguous",
                    message=(
                        f"{resolved.name} has several datetime columns "
                        f"({', '.join(sorted(candidates))}); pass timestamp= to "
                        "choose the time axis."
                    ),
                    table=resolved.name,
                )
            )
            return None, None, value, entity_id
    elif resolved_timestamp not in frame.columns:
        warnings.append(
            AnalysisWarning(
                code="time_series_timestamp_not_found",
                message=(
                    f"Timestamp column {resolved_timestamp!r} was not found in "
                    f"{resolved.name!r}."
                ),
                table=resolved.name,
                column=resolved_timestamp,
            )
        )
        return None, resolved_timestamp, value, entity_id

    resolved_entity = entity_id
    if resolved_entity is not None and resolved_entity not in frame.columns:
        warnings.append(
            AnalysisWarning(
                code="time_series_entity_not_found",
                message=(
                    f"Entity column {resolved_entity!r} was not found in "
                    f"{resolved.name!r}; the rows were analyzed as one series."
                ),
                table=resolved.name,
                column=resolved_entity,
            )
        )
        resolved_entity = None
    return resolved, resolved_timestamp, value, resolved_entity


def observed_series(frame: pd.DataFrame, timestamp: str, value: str) -> pd.Series:
    """The raw observations as a time-indexed series, sorted, nothing filled.

    Duplicated timestamps are kept. Every hygiene check reads this, because a
    duplicate that has already been collapsed cannot be counted.
    """
    times = pd.to_datetime(frame[timestamp], errors="coerce")
    values = pd.to_numeric(frame[value], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(times), name=value)
    return series[series.index.notna()].sort_index()


@dataclass(frozen=True, slots=True)
class RegularSeries:
    """A series forced onto an even grid, and the record of what that cost."""

    values: pd.Series
    frequency: Frequency
    expected_periods: int
    observed_periods: int
    missing_periods: int
    duplicate_timestamps: int
    filled_periods: int


def regular_series(
    observed: pd.Series,
    frequency: Frequency,
    *,
    aggregation: str = "mean",
) -> RegularSeries | None:
    """Collapse duplicates and reindex onto the inferred grid.

    Duplicates are averaged rather than summed. For a rate or a level the mean
    is correct and the sum is nonsense; for a count the reverse is arguably
    true, but averaging cannot invent a spike that was not there, and the
    duplicate finding tells the reader to resolve the ambiguity at source
    rather than leaving Prism to guess it.
    """
    if observed.empty:
        return None
    duplicates = int(observed.index.duplicated().sum())
    collapsed = observed.groupby(level=0).agg(aggregation) if duplicates else observed

    offset = frequency_offset(frequency)
    if offset is None:
        return None
    try:
        grid = pd.date_range(
            start=collapsed.index.min(), end=collapsed.index.max(), freq=offset
        )
    except (ValueError, TypeError):  # pragma: no cover - unusable offset
        return None
    if len(grid) == 0:
        return None

    gridded = collapsed.reindex(grid)
    # Observations that do not land on the grid at all (an irregular series
    # snapped to a dominant spacing) are counted as missing periods rather than
    # silently dropped.
    missing = int(gridded.isna().sum())
    return RegularSeries(
        values=gridded,
        frequency=frequency,
        expected_periods=len(grid),
        observed_periods=int(gridded.notna().sum()),
        missing_periods=missing,
        duplicate_timestamps=duplicates,
        filled_periods=0,
    )


def analysis_series(
    regular: RegularSeries,
    *,
    warnings: list[AnalysisWarning],
    sampling: list[SamplingRecord],
    table: str,
    column: str,
) -> pd.Series | None:
    """Fill the grid so decomposition and autocorrelation can run, and disclose it.

    STL, ACF, and the stationarity tests all require a complete series. Filling
    is therefore unavoidable, but it manufactures observations that were never
    recorded, so it is recorded as sampling and stated on the page. Every
    structural finding downstream carries the same caveat.
    """
    values = regular.values
    if values.notna().sum() < MIN_OBSERVATIONS:
        return None
    if not values.isna().any():
        return values

    filled = values.interpolate(method="time", limit_direction="both")
    count = int(values.isna().sum())
    warnings.append(
        AnalysisWarning(
            code="time_series_interpolated_for_analysis",
            message=(
                f"{count:,} of {regular.expected_periods:,} {regular.frequency.label} "
                f"period(s) had no observation and were interpolated so trend, "
                f"seasonality, and stationarity could be computed. The gap and "
                f"coverage findings describe the real data; the structural "
                f"findings describe this reconstruction."
            ),
            table=table,
            column=column,
        )
    )
    sampling.append(
        SamplingRecord(
            operation="time_series_regularization",
            source_rows=regular.observed_periods,
            sampled_rows=regular.expected_periods,
            strategy="time_weighted_interpolation_onto_inferred_grid",
            seed=0,
            reason="structural_analysis_requires_a_complete_series",
            limitations=(
                "Interpolated periods are reconstructions, not observations.",
                "A long interpolated run flattens variance and can hide a change "
                "point that occurred inside it.",
            ),
        )
    )
    return filled


def thin_for_chart(series: pd.Series, limit: int = MAX_CHART_POINTS) -> pd.Series:
    """Evenly thin a long series so a report stays a portable single file."""
    if len(series) <= limit:
        return series
    step = int(np.ceil(len(series) / limit))
    return series.iloc[::step]


def to_timestamp(value: Any) -> pd.Timestamp:
    """Narrow a pandas index label to a Timestamp.

    Iterating a time-indexed Series yields keys typed only as ``Hashable``, so
    the conversion is done in one place rather than ignored at each call site.
    """
    return pd.Timestamp(value)


def series_points(series: pd.Series) -> list[dict[str, Any]]:
    """Chart-ready points, banked into evidence so the report needs no frame."""
    thinned = thin_for_chart(series)
    index = pd.DatetimeIndex(thinned.index)
    return [
        {
            "t": index[position].isoformat(),
            "y": None if pd.isna(value) else float(value),
        }
        for position, value in enumerate(thinned.to_numpy())
    ]


def evidence_by_kind(evidence: list[Evidence], kind: str) -> Evidence | None:
    for item in evidence:
        if item.kind == kind:
            return item
    return None


def as_float(value: Any) -> float:
    return float(value)
