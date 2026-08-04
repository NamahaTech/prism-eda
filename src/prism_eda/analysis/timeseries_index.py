"""Is the time axis trustworthy?

Everything a forecast depends on is downstream of this question, and none of it
is visible in a row count. A series can have no missing values and still be
unusable because nine consecutive days are absent as *rows* rather than as
nulls; a panel can look complete because one busy entity fills every date while
another has two months of history.

So this module reports absence in the two forms it actually takes — a day that
was never recorded, and a day recorded blank — and never as a single
"missingness" percentage that averages the two into something meaningless. Gaps
are reported as contiguous blocks, because a forecaster cares that the outage
lasted nine days, not that 1.2% of rows are absent.

Every check here reads the raw observations, never the reconstructed grid. These
are exact claims about defects, and a duplicate that has already been collapsed
cannot be counted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prism_eda.analysis._timeseries import (
    MAX_PANEL_ENTITIES,
    Frequency,
    RegularSeries,
    frequency_offset,
    to_timestamp,
)
from prism_eda.evidence.models import Evidence, EvidenceScope

#: Blocks listed individually before the list becomes a wall.
MAX_REPORTED_BLOCKS = 12

#: Below this share of gaps matching the dominant spacing, the series is
#: irregular rather than merely gappy, and grid-based analysis is a stretch.
IRREGULAR_BELOW = 0.9

#: An entity with fewer than this many complete seasonal cycles cannot support a
#: seasonal forecast, however long the panel as a whole looks.
MIN_ENTITY_CYCLES = 2

#: History below this fraction of the longest entity's makes a panel unbalanced.
#:
#: The absolute seasonal floor above and this relative bar catch different
#: failures, and a panel needs both. A store with two months of history clears
#: the floor for a weekly cycle while having eight percent of the history its
#: peers have — enough to fit, not enough to be fitted *with*, because a model
#: pooled across the panel will be dominated by the long entities.
IMBALANCED_BELOW = 0.2


def index_evidence(
    observed: pd.Series,
    frequency: Frequency | None,
    *,
    table: str,
    timestamp: str,
    value: str,
) -> Evidence:
    """Span, spacing, ordering, and timezone of the time axis itself."""
    index = pd.DatetimeIndex(observed.index)
    distinct = index.unique()
    span_days = (
        float((index.max() - index.min()).total_seconds() / 86_400)
        if len(index)
        else 0.0
    )
    now = pd.Timestamp.now("UTC")
    if index.tz is None:
        # A naive index cannot be compared against an aware timestamp, and
        # assuming a timezone for it would invent information.
        now = now.tz_localize(None)
    future = int((index > now).sum())

    return Evidence.create(
        kind="time_series_index",
        scope=EvidenceScope(table=table, columns=(timestamp, value)),
        value={
            "timestamp": timestamp,
            "value": value,
            "row_count": int(len(observed)),
            "distinct_timestamp_count": int(len(distinct)),
            "start": index.min().isoformat() if len(index) else None,
            "end": index.max().isoformat() if len(index) else None,
            "span_days": span_days,
            "timezone": str(index.tz) if index.tz is not None else None,
            "rows_in_file_order_are_sorted": bool(
                pd.Series(index).is_monotonic_increasing
            ),
            "future_timestamp_count": future,
            "frequency": (
                {
                    "alias": frequency.alias,
                    "label": frequency.label,
                    "regularity": frequency.regularity,
                    "seasonal_period": frequency.seasonal_period,
                    "seasonal_label": frequency.seasonal_label,
                }
                if frequency
                else None
            ),
        },
        method="time_axis_summary_v1",
        description=f"Time axis summary for {table}.{timestamp}.",
        confidence=1.0,
        assumptions=(
            "Frequency is the dominant spacing between distinct observations, "
            "not a declared schedule.",
        ),
    )


def duplicate_evidence(
    frame: pd.DataFrame,
    *,
    table: str,
    timestamp: str,
    value: str,
    entity_id: str | None,
) -> Evidence | None:
    """Timestamps recorded more than once for the same series.

    Entity-aware by necessity. In a panel every date legitimately appears once
    per entity, so a naive duplicate count on the timestamp column alone reports
    that essentially every row is duplicated — a number that is both true and
    completely useless.
    """
    keys = [timestamp] if entity_id is None else [timestamp, entity_id]
    times = pd.to_datetime(frame[timestamp], errors="coerce")
    subset = frame.assign(**{timestamp: times}).dropna(subset=[timestamp])
    if subset.empty:
        return None
    duplicated = subset.duplicated(subset=keys, keep=False)
    affected = int(duplicated.sum())
    if not affected:
        return None

    groups = subset[duplicated].groupby(keys, dropna=False)
    examples: list[dict[str, Any]] = []
    for key, group in list(groups)[:MAX_REPORTED_BLOCKS]:
        # Grouping by a one-element list still yields tuple keys, so normalize
        # rather than indexing differently depending on the entity argument.
        parts = key if isinstance(key, tuple) else (key,)
        stamp = parts[0]
        entity = parts[1] if len(parts) > 1 else None
        spread = pd.to_numeric(group[value], errors="coerce")
        examples.append(
            {
                "timestamp": to_timestamp(stamp).isoformat(),
                "entity": None if entity is None else str(entity),
                "row_count": int(len(group)),
                "distinct_values": int(spread.nunique(dropna=True)),
                "min": None if spread.isna().all() else float(spread.min()),
                "max": None if spread.isna().all() else float(spread.max()),
            }
        )
    return Evidence.create(
        kind="time_series_duplicate_timestamps",
        scope=EvidenceScope(table=table, columns=(timestamp,)),
        value={
            "timestamp": timestamp,
            "entity_id": entity_id,
            "duplicated_row_count": affected,
            "duplicated_timestamp_count": int(groups.ngroups),
            "duplicated_row_rate": affected / len(subset),
            "examples": examples,
            "conflicting_value_count": sum(
                1 for item in examples if item["distinct_values"] > 1
            ),
        },
        method="entity_aware_duplicate_timestamp_scan_v1",
        description=f"Duplicate timestamps in {table}.{timestamp}.",
        confidence=1.0,
        assumptions=(
            "Duplicates are counted per series: per entity when an entity column "
            "is supplied, otherwise across the whole table.",
        ),
    )


def _runs(mask: pd.Series) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as (start position, length)."""
    values = mask.to_numpy()
    if not values.any():
        return []
    padded = np.concatenate([[False], values, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(end - start))
        for start, end in zip(edges[::2], edges[1::2], strict=True)
    ]


def gap_evidence(
    observed: pd.Series,
    regular: RegularSeries,
    *,
    table: str,
    timestamp: str,
    value: str,
) -> Evidence | None:
    """Absent periods, reported as blocks and split by what kind of absent.

    Two different failures wear the same total. A row that was never written is
    a collection failure; a row written with a blank value is a measurement
    failure. They have different causes and different fixes, and a single
    "missing" percentage hides both.
    """
    grid = pd.DatetimeIndex(regular.values.index)
    recorded = pd.DatetimeIndex(observed.index).unique()
    absent = pd.Series(~grid.isin(recorded), index=grid)

    # Present in the data, but the value itself is blank.
    blank_stamps = pd.DatetimeIndex(observed[observed.isna()].index).unique()
    blank = pd.Series(grid.isin(blank_stamps), index=grid)

    def blocks(mask: pd.Series) -> list[dict[str, Any]]:
        return [
            {
                "start": grid[start].isoformat(),
                "end": grid[start + length - 1].isoformat(),
                "periods": length,
            }
            for start, length in _runs(mask)
        ]

    absent_blocks = sorted(
        blocks(absent), key=lambda item: item["periods"], reverse=True
    )
    blank_blocks = sorted(blocks(blank), key=lambda item: item["periods"], reverse=True)
    if not absent_blocks and not blank_blocks:
        return None

    total_absent = int(absent.sum())
    total_blank = int(blank.sum())
    return Evidence.create(
        kind="time_series_gaps",
        scope=EvidenceScope(table=table, columns=(timestamp, value)),
        value={
            "frequency_label": regular.frequency.label,
            "expected_periods": regular.expected_periods,
            "unrecorded_period_count": total_absent,
            "unrecorded_period_rate": total_absent / regular.expected_periods
            if regular.expected_periods
            else 0.0,
            "unrecorded_blocks": absent_blocks[:MAX_REPORTED_BLOCKS],
            "unrecorded_block_count": len(absent_blocks),
            "longest_unrecorded_block": absent_blocks[0]["periods"]
            if absent_blocks
            else 0,
            "blank_period_count": total_blank,
            "blank_blocks": blank_blocks[:MAX_REPORTED_BLOCKS],
            "blank_block_count": len(blank_blocks),
            "longest_blank_block": blank_blocks[0]["periods"] if blank_blocks else 0,
        },
        method="gap_block_scan_v1",
        description=f"Absent periods in {table}.{value}.",
        confidence=1.0,
        assumptions=(
            "A period with no row is a collection failure; a period with a blank "
            "value is a measurement failure. They are counted separately.",
            "Blocks are measured against the inferred grid, so they move if the "
            "inferred frequency is wrong.",
        ),
    )


def irregularity_evidence(
    observed: pd.Series,
    frequency: Frequency,
    *,
    table: str,
    timestamp: str,
) -> Evidence | None:
    """How far the real spacing departs from a single dominant interval."""
    unique = pd.Series(pd.DatetimeIndex(observed.index).unique()).sort_values()
    if len(unique) < 3:
        return None
    gaps = unique.diff().dropna()
    if gaps.empty:
        return None
    seconds = gaps.dt.total_seconds()
    dominant = float(seconds.mode().iloc[0]) if not seconds.mode().empty else 0.0
    on_grid = float((seconds == dominant).mean()) if dominant else 0.0
    if on_grid >= IRREGULAR_BELOW:
        return None
    return Evidence.create(
        kind="time_series_irregular_spacing",
        scope=EvidenceScope(table=table, columns=(timestamp,)),
        value={
            "timestamp": timestamp,
            "dominant_gap_seconds": dominant,
            "frequency_label": frequency.label,
            "on_grid_rate": on_grid,
            "distinct_gap_count": int(seconds.nunique()),
            "median_gap_seconds": float(seconds.median()),
            "min_gap_seconds": float(seconds.min()),
            "max_gap_seconds": float(seconds.max()),
        },
        method="inter_arrival_spacing_scan_v1",
        description=f"Irregular observation spacing in {table}.{timestamp}.",
        confidence=0.9,
        assumptions=(
            "Irregular spacing does not prevent analysis, but every structural "
            "result is computed on a regularized reconstruction of the series.",
        ),
    )


def composition_evidence(
    frame: pd.DataFrame,
    *,
    table: str,
    timestamp: str,
    value: str,
    entity_id: str,
) -> Evidence | None:
    """Whether the aggregate is made of the same series throughout.

    This is the trap in every panel. Totalling an unbalanced panel produces a
    level shift on the day an entity joins or leaves — not because demand
    changed, but because the *thing being measured* changed. Every downstream
    check reads that as a regime change, and a forecaster who believes it
    re-baselines onto an event that never happened.

    So the count of contributing series is measured per period and reported
    whenever it moves.
    """
    times = pd.to_datetime(frame[timestamp], errors="coerce")
    work = frame.assign(**{timestamp: times}).dropna(subset=[timestamp])
    if work.empty:
        return None
    active = work.groupby(timestamp)[entity_id].nunique()
    if active.empty or int(active.min()) == int(active.max()):
        return None

    changes: list[dict[str, Any]] = []
    previous = int(active.iloc[0])
    for stamp, count in active.items():
        current = int(count)
        if current != previous:
            changes.append(
                {
                    "timestamp": to_timestamp(stamp).isoformat(),
                    "from": previous,
                    "to": current,
                }
            )
            previous = current
    return Evidence.create(
        kind="time_series_panel_composition",
        scope=EvidenceScope(table=table, columns=(entity_id, timestamp, value)),
        value={
            "entity_id": entity_id,
            "min_active_entities": int(active.min()),
            "max_active_entities": int(active.max()),
            "changes": changes[:MAX_REPORTED_BLOCKS],
            "change_count": len(changes),
            "first_full_coverage": (
                active[active == int(active.max())].index[0].isoformat()
                if (active == int(active.max())).any()
                else None
            ),
        },
        method="panel_composition_scan_v1",
        description=f"Series contributing to the {table} aggregate over time.",
        confidence=1.0,
        assumptions=(
            "A change in the number of contributing series moves the aggregate "
            "without anything changing in any individual series.",
        ),
    )


def panel_evidence(
    frame: pd.DataFrame,
    frequency: Frequency,
    *,
    table: str,
    timestamp: str,
    value: str,
    entity_id: str,
) -> Evidence | None:
    """Per-entity history, because a panel is almost never balanced.

    The aggregate can span two years while half the entities have two months.
    Forecasting per entity is then impossible for those entities regardless of
    what the total looks like, so coverage is reported per entity and the short
    ones are named.
    """
    times = pd.to_datetime(frame[timestamp], errors="coerce")
    work = frame.assign(**{timestamp: times}).dropna(subset=[timestamp])
    if work.empty or entity_id not in work.columns:
        return None
    offset = frequency_offset(frequency)

    rows: list[dict[str, Any]] = []
    for name, group in work.groupby(entity_id, dropna=False):
        stamps = pd.DatetimeIndex(group[timestamp]).unique()
        if len(stamps) == 0:
            continue
        try:
            expected = len(
                pd.date_range(start=stamps.min(), end=stamps.max(), freq=offset)
            )
        except (ValueError, TypeError):  # pragma: no cover - unusable offset
            expected = len(stamps)
        observed_values = pd.to_numeric(group[value], errors="coerce")
        rows.append(
            {
                "entity": str(name),
                "row_count": int(len(group)),
                "period_count": int(len(stamps)),
                "expected_periods": int(expected),
                "coverage": float(len(stamps) / expected) if expected else 0.0,
                "start": stamps.min().isoformat(),
                "end": stamps.max().isoformat(),
                "span_days": float(
                    (stamps.max() - stamps.min()).total_seconds() / 86_400
                ),
                "non_null_values": int(observed_values.notna().sum()),
            }
        )
    if not rows:
        return None
    rows.sort(key=lambda item: item["period_count"])

    period = frequency.seasonal_period
    required = (period * MIN_ENTITY_CYCLES) if period else 0
    short = [item for item in rows if required and item["period_count"] < required]
    longest = max(item["period_count"] for item in rows)
    imbalanced = [
        item
        for item in rows
        if longest and item["period_count"] / longest < IMBALANCED_BELOW
    ]
    for item in rows:
        item["history_share"] = item["period_count"] / longest if longest else 0.0

    return Evidence.create(
        kind="time_series_panel_coverage",
        scope=EvidenceScope(table=table, columns=(entity_id, timestamp, value)),
        value={
            "entity_id": entity_id,
            "entity_count": len(rows),
            "entities": rows[:MAX_PANEL_ENTITIES],
            "listed_entity_count": min(len(rows), MAX_PANEL_ENTITIES),
            "shortest_period_count": rows[0]["period_count"],
            "longest_period_count": longest,
            "required_periods_for_seasonality": required or None,
            "short_history_entities": [item["entity"] for item in short][
                :MAX_PANEL_ENTITIES
            ],
            "short_history_count": len(short),
            "imbalanced_entities": [
                {
                    "entity": item["entity"],
                    "period_count": item["period_count"],
                    "history_share": item["history_share"],
                }
                for item in imbalanced[:MAX_PANEL_ENTITIES]
            ],
            "imbalanced_count": len(imbalanced),
            "seasonal_label": frequency.seasonal_label,
        },
        method="per_entity_coverage_scan_v1",
        description=f"Panel coverage for {table} by {entity_id}.",
        confidence=1.0,
        assumptions=(
            "Coverage is measured against each entity's own first and last "
            "observation, so a late-joining entity is not penalized for the "
            "history it could not have.",
            "History length and coverage are different questions: an entity can "
            "have complete coverage of a very short window.",
        ),
    )
