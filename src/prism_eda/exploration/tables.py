"""Profile one table: full-table statistics plus a sampled closer look.

Two bases are kept strictly apart, because they license different claims:

- **Full table** — counts, nulls, distinct values, ranges, top values and row
  volume over time are computed by the source over every row.
- **Sample** — distribution shapes, correlations, format checks and example
  rows come from a deterministic row sample. Everything derived from it is
  marked ``basis: sample`` and the sampling is recorded.

A step that fails (a timeout, an unreadable column) is recorded and the rest of
the table is still profiled; one bad column never costs the whole table.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from prism_eda._serialization import to_jsonable
from prism_eda.analysis._limits import ProfileLimits
from prism_eda.analysis.associations import (
    association_evidence,
    missingness_evidence,
    sample_evidence,
)
from prism_eda.analysis.distributions import (
    category_frequency_evidence,
    numeric_distribution_evidence,
)
from prism_eda.analysis.quality_checks import build_quality_findings
from prism_eda.catalog.models import ColumnCatalog, SourceInfo, TableCatalog
from prism_eda.catalog.profiling import profile_table
from prism_eda.evidence.models import OBSERVATION, Evidence, EvidenceScope, Finding
from prism_eda.exploration.config import ExploreConfig
from prism_eda.results import AnalysisFailure, AnalysisWarning, SamplingRecord
from prism_eda.sources.base import (
    ColumnInfo,
    ColumnStats,
    Source,
    SourceError,
    TableInfo,
    normalise_value,
    sampling_rate,
    value_hash,
)

FINGERPRINT_METHOD = "exploration_shape_v1"
_DATE_TEXT = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
_SAMPLE_LIMITS = dataclasses.replace(
    ProfileLimits.for_detail("standard"), sample_rows=8, sample_columns=14
)
#: Charts per table. Past this a table's section stops being readable; the
#: columns left out still have their full statistics in the column table.
MAX_CHARTED_COLUMNS = 24


@dataclass(slots=True)
class KeyColumn:
    """An identifier-like column, with the value samples relationships need."""

    name: str
    kind: str
    distinct: int
    non_null: int
    unique: bool
    rate: int
    #: Canonical distinct values in the hash sample, in hash order.
    values: list[str] = field(default_factory=list)
    #: Canonical values of sampled *rows* (with repeats), for row coverage.
    row_values: list[str] = field(default_factory=list)
    #: A few raw values, to show how the column actually writes them.
    examples: list[str] = field(default_factory=list)
    #: Share of sampled values made only of digits, and their median length.
    digit_share: float = 0.0
    median_length: float = 0.0
    max_length: int | None = None


@dataclass(slots=True)
class TableState:
    """Everything learned about one table; also what a checkpoint stores."""

    key: str
    display: str
    info: TableInfo
    columns: list[ColumnInfo] = field(default_factory=list)
    rows: int = 0
    rows_exact: bool = True
    status: str = "profiled"
    message: str | None = None
    catalog: TableCatalog | None = None
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    warnings: list[AnalysisWarning] = field(default_factory=list)
    failures: list[AnalysisFailure] = field(default_factory=list)
    sampling: list[SamplingRecord] = field(default_factory=list)
    key_columns: list[KeyColumn] = field(default_factory=list)
    time_ranges: list[dict[str, Any]] = field(default_factory=list)
    sample_rows: int = 0

    def column(self, name: str) -> ColumnInfo:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def key_column(self, name: str) -> KeyColumn | None:
        for column in self.key_columns:
            if column.name == name:
                return column
        return None


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def choose_grain(start: Any, end: Any, kind: str) -> str:
    try:
        span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86_400
    except (TypeError, ValueError):
        return "month"
    if span <= 3 and kind == "datetime":
        return "hour"
    if span <= 200:
        return "day"
    if span <= 1_100:
        return "week"
    if span <= 22_000:
        return "month"
    return "year"


def _semantic_type(column: ColumnInfo, stats: ColumnStats | None) -> str:
    if column.kind == "boolean":
        return "boolean"
    if column.kind in {"datetime", "date"}:
        return "datetime"
    if column.kind in {"integer", "float"}:
        return "numeric"
    if column.kind == "string":
        distinct = stats.distinct if stats else None
        non_null = stats.non_null if stats else 0
        if distinct is not None and non_null:
            if distinct <= 50 or (distinct <= 200 and distinct / non_null <= 0.05):
                return "categorical"
        return "text"
    return "other"


def _is_unique(distinct: int, non_null: int, exact: bool) -> bool:
    """Whether every value appears once.

    An estimated distinct count is typically within a percent or two of the
    truth, so an estimate is given that much room.
    """
    if not non_null:
        return False
    return distinct >= (0.999 if exact else 0.97) * non_null


def _clip_histogram(item: Evidence, series: pd.Series) -> Evidence:
    """Redraw a long-tailed histogram over the 1st-99th percentile range.

    One extreme value otherwise squeezes every other row into the first bar.
    The values outside the range are counted into the end bars, and the
    evidence records that it was clipped.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) < 50:
        return item
    low, high = np.quantile(values, [0.01, 0.99])
    spread = high - low
    if spread <= 0 or (values.max() - values.min()) <= 3 * spread:
        return item
    clipped = np.clip(values, low, high)
    # Positive values spanning orders of magnitude (amounts, balances) only
    # show their shape on a log scale.
    log_scale = bool(low > 0 and high / low >= 100)
    bins: Any = np.geomspace(low, high, 31) if log_scale else 30
    counts, edges = np.histogram(clipped, bins=bins)
    return dataclasses.replace(
        item,
        value={
            **item.value,
            "histogram": {
                "counts": [int(count) for count in counts],
                "edges": [float(edge) for edge in edges],
            },
            "histogram_clipped": True,
            "histogram_log_scale": log_scale,
        },
    )


def _stat_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return to_jsonable(value)


def _shape_fingerprint(
    info: TableInfo, rows: int, columns: Sequence[ColumnInfo]
) -> str:
    text = "|".join(
        [
            info.key,
            str(rows),
            *[f"{column.name}:{column.native_type}" for column in columns],
        ]
    )
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _tag_sample(item: Evidence, rows: int, total: int) -> Evidence:
    """Mark evidence computed on the row sample so no reader takes it as exact."""
    return dataclasses.replace(
        item,
        metadata={
            **item.metadata,
            "basis": "sample",
            "sample_rows": rows,
            "table_rows": total,
        },
    )


def _guard(
    state: TableState, stage: str, work: Callable[[], Any], default: Any = None
) -> Any:
    try:
        return work()
    except SourceError as error:
        state.failures.append(
            AnalysisFailure(
                stage=stage, message=str(error), recoverable=True, table=state.display
            )
        )
    except Exception as error:  # noqa: BLE001 - one step never costs the table
        state.failures.append(
            AnalysisFailure(
                stage=stage,
                message=f"{type(error).__name__}: {error}",
                recoverable=True,
                table=state.display,
            )
        )
    return default


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def _column_stats(
    source: Source, state: TableState, columns: Sequence[ColumnInfo]
) -> dict[str, ColumnStats]:
    """All columns in one pass; column by column if that pass fails."""
    try:
        return source.column_stats(state.info, columns)
    except SourceError as error:
        first_error = str(error)
    stats: dict[str, ColumnStats] = {}
    failed = []
    for column in columns:
        try:
            stats.update(source.column_stats(state.info, [column]))
        except SourceError:
            failed.append(column.name)
    if failed:
        shown = ", ".join(failed[:8]) + ("…" if len(failed) > 8 else "")
        state.failures.append(
            AnalysisFailure(
                stage="column_statistics",
                message=(
                    f"Statistics could not be computed for {len(failed)} column(s) "
                    f"({shown}): {first_error}"
                ),
                recoverable=True,
                table=state.display,
            )
        )
    return stats


def _timelines(
    source: Source,
    state: TableState,
    stats: dict[str, ColumnStats],
    config: ExploreConfig,
) -> None:
    candidates = [
        column
        for column in state.columns
        if column.kind in {"datetime", "date"}
        and column.name in stats
        and stats[column.name].non_null
        and stats[column.name].minimum is not None
    ]
    candidates.sort(key=lambda column: -stats[column.name].non_null)
    for column in candidates:
        found = stats[column.name]
        state.time_ranges.append(
            {
                "table": state.display,
                "column": column.name,
                "min": _stat_value(found.minimum),
                "max": _stat_value(found.maximum),
                "rows": found.non_null,
            }
        )
    for column in candidates[: config.timelines_per_table]:
        found = stats[column.name]
        grain = choose_grain(found.minimum, found.maximum, column.kind)
        buckets = _guard(
            state,
            "timeline",
            functools.partial(source.time_histogram, state.info, column, grain),
        )
        if not buckets:
            continue
        gaps = find_gaps(buckets, grain)
        item = Evidence.create(
            kind="explore_timeline",
            scope=EvidenceScope(table=state.display, columns=(column.name,)),
            value={
                "column": column.name,
                "granularity": grain,
                "min": _stat_value(found.minimum),
                "max": _stat_value(found.maximum),
                "rows": found.non_null,
                "buckets": [
                    {"start": start, "count": count} for start, count in buckets
                ],
                "gaps": gaps,
            },
            method="full_table_period_counts_v1",
            description=f"Rows per {grain} for {state.display}.{column.name}.",
            metadata={"basis": "full_table"},
        )
        state.evidence.append(item)
        for gap in gaps[:3]:
            state.findings.append(
                Finding.create(
                    title=f"Gap in {state.display}.{column.name}",
                    summary=(
                        f"No rows between {gap['start'][:10]} and {gap['end'][:10]} "
                        f"({gap['periods']} consecutive {grain}s), although the "
                        f"{grain}s on either side hold about "
                        f"{gap['typical_count']:,} rows each."
                    ),
                    severity="medium",
                    confidence=0.9,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm whether the data for this period was never "
                        "delivered or genuinely has no activity; features and "
                        "baselines computed across the gap will be distorted."
                    ),
                )
            )


def find_gaps(buckets: Sequence[tuple[str, int]], grain: str) -> list[dict[str, Any]]:
    """Runs of empty periods inside an otherwise busy timeline.

    A quiet timeline has empty periods everywhere, so a gap only counts when
    the series is dense: at least 20 periods, a typical period of 20 or more
    rows, and at least three consecutive empty periods with a busy period on
    both sides.
    """
    if len(buckets) < 20:
        return []
    frame = pd.DataFrame(buckets, columns=["start", "count"])
    frame["start"] = pd.to_datetime(frame["start"], errors="coerce")
    frame = frame.dropna().sort_values("start")
    if frame.empty:
        return []
    rule = {"hour": "h", "day": "D", "week": "W-MON", "month": "MS", "year": "YS"}[
        grain
    ]
    # Trim sparse leading/trailing periods (sentinel dates) before judging.
    typical = float(frame["count"].median())
    if typical < 20:
        return []
    busy = frame[frame["count"] >= typical * 0.2]
    if len(busy) < 20:
        return []
    start, end = busy["start"].min(), busy["start"].max()
    series = frame.set_index("start")["count"]
    expected = pd.date_range(start, end, freq=rule)
    if len(expected) > 20_000 or len(expected) < 20:
        return []
    filled = series.reindex(expected, fill_value=0)
    gaps: list[dict[str, Any]] = []
    run_start = None
    values = filled.to_numpy()
    index = filled.index
    for position, count in enumerate(values):
        if count == 0 and run_start is None:
            run_start = position
        elif count != 0 and run_start is not None:
            length = position - run_start
            if length >= 3 and run_start > 0:
                around = [values[run_start - 1], count]
                if min(around) >= typical * 0.2:
                    gaps.append(
                        {
                            "start": index[run_start].isoformat(),
                            "end": index[position].isoformat(),
                            "periods": int(length),
                            "typical_count": int(typical),
                        }
                    )
            run_start = None
    gaps.sort(key=lambda gap: -gap["periods"])
    return gaps


def _sample_analysis(
    state: TableState, sample: pd.DataFrame, config: ExploreConfig
) -> TableCatalog | None:
    """Distributions, associations and format checks on the row sample."""
    if sample.empty:
        return None
    rows = len(sample)
    total = state.rows
    sample_catalog = profile_table(
        state.display, sample, SourceInfo(kind="sample", location=state.info.key)
    )
    tag = lambda item: _tag_sample(item, rows, total)  # noqa: E731

    def quality() -> None:
        evidence, findings, _ = build_quality_findings(sample, sample_catalog)
        note = (
            f" Seen in a sample of {rows:,} of {total:,} rows." if rows < total else ""
        )
        state.evidence.extend(tag(item) for item in evidence)
        for finding in findings:
            state.findings.append(
                dataclasses.replace(finding, summary=finding.summary + note)
                if note
                else finding
            )

    _guard(state, "sample_quality_checks", quality)

    def charts() -> None:
        charted = 0
        for column in sample_catalog.columns:
            if charted >= MAX_CHARTED_COLUMNS:
                break
            series = sample[column.name]
            item: Evidence | None = None
            if column.semantic_type == "numeric":
                item = numeric_distribution_evidence(
                    series,
                    table=state.display,
                    column=column.name,
                    fit_rows=_SAMPLE_LIMITS.fit_rows,
                )
            elif column.semantic_type in {"categorical", "boolean"}:
                item = category_frequency_evidence(
                    series, table=state.display, column=column.name
                )
            if item is not None:
                if item.kind == "profile_distribution":
                    item = _clip_histogram(item, series)
                state.evidence.append(tag(item))
                charted += 1

    _guard(state, "sample_distributions", charts)

    def associations() -> None:
        matrix, _ = association_evidence(sample, sample_catalog, limits=_SAMPLE_LIMITS)
        if matrix is not None:
            state.evidence.append(tag(matrix))
        for item in missingness_evidence(sample, sample_catalog):
            if item.kind == "profile_co_missingness":
                state.evidence.append(tag(item))
        for item in sample_evidence(sample, sample_catalog, limits=_SAMPLE_LIMITS):
            if item.kind == "profile_sample_rows":
                state.evidence.append(tag(item))

    _guard(state, "sample_associations", associations)

    if rows < total:
        state.sampling.append(
            SamplingRecord(
                operation=f"row_sample:{state.display}",
                source_rows=total,
                sampled_rows=rows,
                strategy="deterministic_hash_or_tablesample",
                seed=config.seed,
                reason=(
                    "Distribution shapes, correlations and format checks do not "
                    "need every row."
                ),
                limitations=(
                    "Rare values and rare defects may be absent from the sample.",
                ),
            )
        )
    return sample_catalog


def _looks_like_dates(values: Sequence[str]) -> bool:
    if not values:
        return False
    hits = sum(1 for value in values[:200] if _DATE_TEXT.match(value))
    return hits >= 0.8 * min(len(values), 200)


def _key_columns(
    source: Source,
    state: TableState,
    stats: dict[str, ColumnStats],
    sample: pd.DataFrame,
    config: ExploreConfig,
) -> None:
    """Pick identifier-like columns and take their value samples."""
    chosen: list[tuple[ColumnInfo, ColumnStats, int]] = []
    for column in state.columns:
        found = stats.get(column.name)
        if found is None or column.kind not in {"integer", "string"}:
            continue
        distinct = found.distinct
        if distinct is None and column.name in sample.columns:
            # No full-table estimate: scale the sample's distinct ratio.
            series = sample[column.name].dropna()
            if len(series):
                distinct = int(series.nunique() / len(series) * found.non_null)
        if not distinct or distinct < config.min_key_distinct:
            continue
        if (
            column.kind == "string"
            and found.mean_length is not None
            and found.mean_length > config.max_key_length
        ):
            continue
        chosen.append((column, found, distinct))
    chosen.sort(key=lambda item: -item[2])
    chosen = chosen[: config.max_key_columns]
    if not chosen:
        return

    requests = {
        column.name: (column, sampling_rate(distinct, config.value_sample))
        for column, _, distinct in chosen
    }
    samples = _guard(
        state,
        "value_samples",
        lambda: source.distinct_sample(state.info, requests, config.value_sample * 2),
        default={},
    )
    for column, found, distinct in chosen:
        raw_values = samples.get(column.name) or []
        if _looks_like_dates(raw_values):
            continue
        canonical = sorted(
            {norm for raw in raw_values if (norm := normalise_value(raw))},
            key=value_hash,
        )[: config.value_sample]
        if len(canonical) < min(config.min_key_distinct, 10):
            continue
        lengths = sorted(len(value) for value in canonical)
        digits = sum(1 for value in canonical if value.isdigit())
        row_values: list[str] = []
        if column.name in sample.columns:
            series = sample[column.name].dropna()
            if len(series) > config.row_coverage_rows:
                series = series.sample(
                    n=config.row_coverage_rows, random_state=config.seed
                )
            row_values = [
                norm for value in series.tolist() if (norm := normalise_value(value))
            ]
        state.key_columns.append(
            KeyColumn(
                name=column.name,
                kind=column.kind,
                distinct=distinct,
                non_null=found.non_null,
                unique=_is_unique(distinct, found.non_null, found.distinct_exact),
                rate=requests[column.name][1],
                values=canonical,
                row_values=row_values,
                # Spread across the sorted sample, so the examples are not
                # all from one end of the value range.
                examples=[
                    str(raw)
                    for raw in raw_values[:: max(1, len(raw_values) // 40)][:40]
                ],
                digit_share=digits / len(canonical),
                median_length=float(lengths[len(lengths) // 2]),
                max_length=found.max_length,
            )
        )


def _drop_identifier_noise(state: TableState) -> None:
    """Account numbers, phone numbers and codes are text on purpose.

    "Numbers stored as text" is a real defect for a measure and a non-event for
    an identifier, where text is what preserves leading zeros.
    """
    identifiers = {column.name for column in state.key_columns}
    dropped = {
        item.id
        for item in state.evidence
        if item.kind == "quality_numeric_as_text"
        and item.scope.columns
        and item.scope.columns[0] in identifiers
    }
    if not dropped:
        return
    state.evidence = [item for item in state.evidence if item.id not in dropped]
    state.findings = [
        finding
        for finding in state.findings
        if not dropped.intersection(finding.evidence_ids)
    ]


def _catalog(
    state: TableState,
    stats: dict[str, ColumnStats],
    top: dict[str, list[tuple[Any, int]]],
    sample: pd.DataFrame,
    sample_catalog: TableCatalog | None,
) -> TableCatalog:
    sample_columns = (
        {column.name: column for column in sample_catalog.columns}
        if sample_catalog
        else {}
    )
    key_names = {column.name for column in state.key_columns}
    columns: list[ColumnCatalog] = []
    for column in state.columns:
        found = stats.get(column.name)
        sampled = sample_columns.get(column.name)
        if found is None:
            columns.append(
                ColumnCatalog(
                    name=column.name,
                    physical_type=column.native_type,
                    semantic_type=_semantic_type(column, None),
                    roles=(),
                    row_count=state.rows,
                    non_null_count=0,
                    missing_count=0,
                    missing_rate=0.0,
                    unique_count=None,
                    unique_rate=None,
                    statistics={"profiled": False},
                    warnings=("Statistics are not available for this column.",),
                )
            )
            continue
        rows = found.rows or state.rows
        missing = rows - found.non_null
        statistics: dict[str, Any] = {
            "kind": column.kind,
            "distinct_exact": found.distinct_exact,
            "basis": "full_table",
        }
        for label, value in (
            ("min", found.minimum),
            ("max", found.maximum),
            ("mean", found.mean),
            ("std", found.std),
            ("zero_count", found.zeros),
            ("negative_count", found.negatives),
            ("min_length", found.min_length),
            ("max_length", found.max_length),
            ("mean_length", found.mean_length),
        ):
            if value is not None:
                statistics[label] = _stat_value(value)
        if column.kind == "string":
            statistics["blank_count"] = found.blank
        quantiles = dict(found.quantiles)
        if not quantiles and sampled and column.kind in {"integer", "float"}:
            # The source could not afford exact quantiles on a table this size.
            for label in ("p5", "q1", "median", "q3", "p95"):
                if sampled.statistics.get(label) is not None:
                    quantiles[label] = sampled.statistics[label]
            if quantiles:
                statistics["quantiles_basis"] = "sample"
        statistics.update(quantiles)
        if sampled and column.kind in {"integer", "float"}:
            for label in ("skewness", "kurtosis"):
                if sampled.statistics.get(label) is not None:
                    statistics[label] = sampled.statistics[label]

        if column.name in top:
            top_values = tuple(
                {"value": _stat_value(value), "count": count}
                for value, count in top[column.name]
            )
            statistics["top_values_basis"] = "full_table"
        elif sampled is not None and column.kind != "other":
            # Scale sample counts to the table so the two bases read alike.
            scale = rows / len(sample) if len(sample) else 1.0
            top_values = tuple(
                {"value": entry["value"], "count": int(round(entry["count"] * scale))}
                for entry in sampled.top_values
                if entry["value"] is not None
            )
            statistics["top_values_basis"] = "sample"
        else:
            top_values = ()

        semantic = _semantic_type(column, found)
        distinct = found.distinct
        unique_rate = distinct / found.non_null if distinct and found.non_null else None
        roles: list[str] = []
        if column.kind in {"datetime", "date"}:
            roles.append("timestamp_candidate")
        if column.name in key_names:
            roles.append("identifier_candidate")
        if (
            distinct is not None
            and found.non_null == rows
            and _is_unique(distinct, rows, found.distinct_exact)
            and column.kind in {"integer", "string"}
        ):
            roles.append("unique_key_candidate")
        if semantic == "numeric" and column.name not in key_names:
            roles.append("measure_candidate")
        elif semantic in {"categorical", "boolean"}:
            roles.append("dimension_candidate")
        elif semantic == "text" and column.name not in key_names:
            roles.append("free_text_candidate")

        warnings: list[str] = []
        if rows and found.non_null == 0:
            warnings.append("Column is entirely empty.")
        elif rows and missing / rows >= 0.5:
            warnings.append("At least half of the values are missing.")
        if distinct == 1 and found.non_null:
            warnings.append("Column is constant among non-null values.")
        if column.kind == "string" and found.non_null and found.blank:
            share = found.blank / found.non_null
            if share >= 0.01:
                warnings.append(f"{share:.1%} of non-null values are blank text.")

        columns.append(
            ColumnCatalog(
                name=column.name,
                physical_type=column.native_type,
                semantic_type=semantic,
                roles=tuple(roles),
                row_count=rows,
                non_null_count=found.non_null,
                missing_count=missing,
                missing_rate=missing / rows if rows else 0.0,
                unique_count=distinct,
                unique_rate=unique_rate,
                memory_bytes=None,
                statistics=statistics,
                top_values=top_values,
                warnings=tuple(warnings),
            )
        )
    return TableCatalog(
        name=state.display,
        row_count=state.rows,
        column_count=len(state.columns),
        memory_bytes=int(state.info.bytes or 0),
        duplicate_row_count=None,
        fingerprint=_shape_fingerprint(state.info, state.rows, state.columns),
        fingerprint_method=FINGERPRINT_METHOD,
        source=SourceInfo(kind=state.info.source, location=state.info.key),
        columns=tuple(columns),
    )


def _column_findings(state: TableState) -> None:
    """Roll up empty, constant and blank-heavy columns: one finding per kind.

    A wide table can have dozens of dead columns; a finding each would bury
    everything else in the report.
    """
    if state.catalog is None or not state.rows:
        return
    summary = next(
        (item for item in state.evidence if item.kind == "explore_table_summary"), None
    )
    if summary is None:
        return
    empty = [
        c.name
        for c in state.catalog.columns
        if c.statistics.get("kind") and c.non_null_count == 0
    ]
    constant = [
        c.name
        for c in state.catalog.columns
        if c.unique_count == 1 and c.non_null_count and c.missing_count == 0
    ]
    sparse = [
        (c.name, c.missing_rate)
        for c in state.catalog.columns
        if c.non_null_count and c.missing_rate >= 0.5
    ]
    blank = [
        (c.name, c.statistics["blank_count"] / c.non_null_count)
        for c in state.catalog.columns
        if c.non_null_count
        and c.statistics.get("blank_count")
        and c.statistics["blank_count"] / c.non_null_count >= 0.05
    ]

    def listed(names: Sequence[str]) -> str:
        shown = ", ".join(names[:8])
        return shown + (f" and {len(names) - 8} more" if len(names) > 8 else "")

    if empty:
        state.findings.append(
            Finding.create(
                title=f"Empty columns in {state.display}",
                summary=(
                    f"{len(empty)} column(s) hold no values at all: {listed(empty)}."
                ),
                severity="medium" if len(empty) >= 5 else "low",
                confidence=1.0,
                evidence_ids=(summary.id,),
                recommendation=(
                    "Confirm whether these are unused fields or fields the "
                    "extract failed to populate."
                ),
            )
        )
    if sparse:
        worst = sorted(sparse, key=lambda item: -item[1])
        shown = ", ".join(f"{name} ({rate:.0%})" for name, rate in worst[:8])
        state.findings.append(
            Finding.create(
                title=f"Mostly missing columns in {state.display}",
                summary=(
                    f"{len(sparse)} column(s) are missing in at least half of the "
                    f"rows: {shown}{'…' if len(worst) > 8 else ''}."
                ),
                severity="medium",
                confidence=1.0,
                evidence_ids=(summary.id,),
                recommendation=(
                    "Check whether missingness is structural (the field applies "
                    "to some rows only) before using these columns."
                ),
            )
        )
    if blank:
        worst = sorted(blank, key=lambda item: -item[1])
        shown = ", ".join(f"{name} ({rate:.0%})" for name, rate in worst[:8])
        state.findings.append(
            Finding.create(
                title=f"Blank text standing in for missing values in {state.display}",
                summary=(
                    f"{len(blank)} text column(s) hold empty or whitespace-only "
                    f"strings rather than nulls: {shown}. Null counts understate "
                    "how much is really missing."
                ),
                severity="medium",
                confidence=1.0,
                evidence_ids=(summary.id,),
                recommendation="Treat blank text as missing when counting coverage.",
            )
        )
    if constant:
        state.findings.append(
            Finding.create(
                title=f"Constant columns in {state.display}",
                summary=(
                    f"{len(constant)} column(s) hold a single value in every row: "
                    f"{listed(constant)}."
                ),
                severity="low",
                confidence=1.0,
                evidence_ids=(summary.id,),
                recommendation="These carry no information for analysis or modelling.",
                category=OBSERVATION,
            )
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def profile_one_table(
    source: Source,
    info: TableInfo,
    display: str,
    config: ExploreConfig,
    tick: Callable[[str], None],
    extras: Callable[[TableState, pd.DataFrame], None] | None = None,
) -> TableState:
    """Profile a table end to end. Never raises for a problem with the table.

    ``extras`` is called with the row sample while it is still in memory, for
    sections that need rows but should not cost a second read of the table.
    """
    state = TableState(key=info.key, display=display, info=info)

    columns = _guard(state, "columns", lambda: source.columns(info))
    if not columns:
        state.status = "failed"
        state.message = (
            state.failures[-1].message
            if state.failures
            else "The table has no columns."
        )
        return state
    if len(columns) > config.max_columns:
        state.warnings.append(
            AnalysisWarning(
                code="columns_capped",
                message=(
                    f"{display} has {len(columns)} columns; the first "
                    f"{config.max_columns} are profiled."
                ),
                table=display,
            )
        )
        columns = columns[: config.max_columns]
    state.columns = list(columns)

    tick("counting rows")
    rows = _guard(state, "row_count", lambda: source.row_count(info))
    if rows is None:
        rows = info.approx_rows
        state.rows_exact = False
        if rows is None:
            state.status = "failed"
            state.message = "The table could not be counted or read."
            return state
        state.warnings.append(
            AnalysisWarning(
                code="row_count_estimated",
                message=(
                    f"{display} could not be counted exactly; the row count is "
                    "the database's own estimate."
                ),
                table=display,
            )
        )
    state.rows = int(rows)

    summary = Evidence.create(
        kind="explore_table_summary",
        scope=EvidenceScope(table=display),
        value={
            "source": info.source,
            "schema": info.schema,
            "name": info.name,
            "kind": info.kind,
            "engine": info.engine,
            "rows": state.rows,
            "rows_exact": state.rows_exact,
            "columns": len(state.columns),
            "bytes": info.bytes,
            "comment": info.comment,
        },
        method="source_catalog_and_count_v1",
        description=f"Size and shape of {display}.",
        metadata={"basis": "full_table"},
    )
    state.evidence.append(summary)

    if state.rows == 0:
        state.status = "empty"
        state.message = "The table has no rows."
        state.catalog = _catalog(state, {}, {}, pd.DataFrame(), None)
        return state

    tick("column statistics")
    stats = _column_stats(source, state, state.columns)

    tick("top values")
    top_columns = [
        column
        for column in state.columns
        if column.name in stats
        and column.kind in {"integer", "string", "boolean", "date"}
        and stats[column.name].distinct is not None
        and 0 < (stats[column.name].distinct or 0) <= config.exact_top_values_distinct
    ]
    top = (
        _guard(
            state,
            "top_values",
            lambda: source.top_values(state.info, top_columns, config.top_values),
            default={},
        )
        if top_columns
        else {}
    )

    tick("timelines")
    _timelines(source, state, stats, config)

    tick("row sample")
    readable = [column for column in state.columns if column.kind != "other"]
    by_distinct = sorted(
        (
            column
            for column in readable
            if column.name in stats and column.kind in {"integer", "string", "datetime"}
        ),
        key=lambda column: -(stats[column.name].distinct or 0),
    )
    sample = _guard(
        state,
        "row_sample",
        lambda: source.sample(
            state.info,
            readable,
            config.sample_rows,
            total_rows=state.rows,
            seed=config.seed,
            hash_columns=by_distinct[:3],
        ),
        default=pd.DataFrame(),
    )
    state.sample_rows = len(sample)

    tick("distributions and checks")
    sample_catalog = _sample_analysis(state, sample, config)
    if extras is not None and not sample.empty:
        _guard(state, "sample_extras", lambda: extras(state, sample))

    if config.relationships:
        tick("identifier samples")
        _key_columns(source, state, stats, sample, config)

    _drop_identifier_noise(state)
    state.catalog = _catalog(state, stats, top, sample, sample_catalog)
    _column_findings(state)
    if state.failures:
        state.status = "partial"
    return state
