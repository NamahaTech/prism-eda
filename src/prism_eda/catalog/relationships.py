"""Candidate key and relationship models plus deterministic discovery helpers."""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd
from pandas.api import types as ptypes

from prism_eda.config import AnalysisMode
from prism_eda.results import AnalysisWarning, SamplingRecord


@dataclass(frozen=True, slots=True)
class KeyCandidate:
    table: str
    columns: tuple[str, ...]
    uniqueness_rate: float
    completeness_rate: float
    confidence: float
    row_count: int
    evaluated_row_count: int
    distinct_count: int
    sampled: bool = False


@dataclass(frozen=True, slots=True)
class RelationshipCandidate:
    parent_table: str
    parent_columns: tuple[str, ...]
    child_table: str
    child_columns: tuple[str, ...]
    cardinality: str
    inclusion_rate: float
    row_coverage: float
    orphan_row_count: int
    parent_unmatched_count: int
    name_similarity: float
    type_compatibility: float
    confidence: float
    sampled: bool = False


@dataclass(frozen=True, slots=True)
class SharedGrain:
    """A set of columns that identifies rows the same way across many tables.

    This is the shape of a warehouse's conformed dimensions and of most public
    statistical releases: dozens of peer tables measuring different things at
    the same `(entity, period)` coordinate. None of them is the parent of any
    other, so describing the set as one grain is both shorter and truer than
    the N x N directional links the pairwise view would produce.
    """

    columns: tuple[str, ...]
    unique_tables: tuple[str, ...]
    repeating_tables: tuple[str, ...]

    @property
    def table_count(self) -> int:
        return len(self.unique_tables) + len(self.repeating_tables)


@dataclass(frozen=True, slots=True)
class SchemaDiscovery:
    keys: tuple[KeyCandidate, ...]
    relationships: tuple[RelationshipCandidate, ...]
    sampling: tuple[SamplingRecord, ...]
    warnings: tuple[AnalysisWarning, ...]
    grains: tuple[SharedGrain, ...] = ()


# Two independently unique ID columns whose value ranges happen to overlap look
# like a one-to-one relationship from inclusion alone. A genuine 1:1 link (a
# table split or extension) almost always shares the key name, so one-to-one
# candidates require real name agreement before they are reported.
_ONE_TO_ONE_MIN_NAME_SIMILARITY = 0.6


def default_max_key_columns(mode: AnalysisMode | str) -> int:
    normalized = AnalysisMode(mode)
    return {
        AnalysisMode.QUICK: 1,
        AnalysisMode.STANDARD: 2,
        AnalysisMode.DEEP: 3,
    }[normalized]


def _normalized_name(value: str) -> str:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    tokens = re.findall(r"[a-z0-9]+", separated.lower())
    singular = [
        token[:-1] if token.endswith("s") and len(token) > 3 else token
        for token in tokens
    ]
    return "_".join(singular)


def _name_similarity(
    parent_table: str,
    parent_columns: Sequence[str],
    child_columns: Sequence[str],
) -> float:
    scores: list[float] = []
    table_name = _normalized_name(parent_table)
    for parent, child in zip(parent_columns, child_columns, strict=True):
        parent_name = _normalized_name(parent)
        child_name = _normalized_name(child)
        if parent_name == child_name:
            scores.append(1.0)
            continue
        parent_tokens = set(parent_name.split("_"))
        child_tokens = set(child_name.split("_"))
        token_score = (
            len(parent_tokens & child_tokens) / len(parent_tokens | child_tokens)
            if parent_tokens | child_tokens
            else 0.0
        )
        table_key = f"{table_name}_id"
        table_score = 0.85 if child_name == table_key else 0.0
        sequence_score = SequenceMatcher(None, parent_name, child_name).ratio()
        scores.append(max(token_score, table_score, sequence_score * 0.75))
    return sum(scores) / len(scores) if scores else 0.0


def _type_family(series: pd.Series) -> str:
    if ptypes.is_bool_dtype(series.dtype):
        return "boolean"
    if ptypes.is_numeric_dtype(series.dtype):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if ptypes.is_string_dtype(series.dtype) or isinstance(
        series.dtype, pd.CategoricalDtype
    ):
        return "string"
    return "other"


def _type_compatibility(
    parent: pd.DataFrame,
    parent_columns: Sequence[str],
    child: pd.DataFrame,
    child_columns: Sequence[str],
) -> float:
    compatible = 0.0
    for parent_column, child_column in zip(parent_columns, child_columns, strict=True):
        parent_family = _type_family(parent[parent_column])
        child_family = _type_family(child[child_column])
        if parent_family == child_family:
            compatible += 1.0
        elif {parent_family, child_family} <= {"numeric", "string"}:
            compatible += 0.25
    return compatible / len(parent_columns)


def _key_name_score(table: str, columns: Sequence[str]) -> float:
    table_name = _normalized_name(table)
    scores = []
    for column in columns:
        name = _normalized_name(column)
        tokens = set(name.split("_"))
        if name == "id" or name == f"{table_name}_id":
            scores.append(1.0)
        elif name.endswith("_id") or "identifier" in tokens:
            scores.append(0.8)
        elif tokens & {"key", "uuid", "guid"}:
            scores.append(0.8)
        elif tokens & {
            "account",
            "code",
            "email",
            "isbn",
            "no",
            "number",
            "serial",
            "sku",
            "username",
        }:
            scores.append(0.65)
        else:
            scores.append(0.2)
    return sum(scores) / len(scores)


# A natural key is built from columns that *partition* the table — a country, a
# year, a sex breakdown — not from a per-row measurement. Averaging at most one
# row per two values is a loose bar deliberately: it admits a year column with
# many observations per year while still rejecting a float measure whose values
# happen to be distinct.
_MAX_DIMENSION_DISTINCT_RATIO = 0.5


# Integers are admitted as dimensions because years, quarters and small coded
# categories are the backbone of real natural keys. That opens one loophole: a
# percentage or a count stored as an int is a measure wearing a dimension's
# shape. Names close it — a measure is almost always named like one.
_MEASURE_NAME_TOKENS = frozenset(
    {
        "amount",
        "avg",
        "average",
        "cost",
        "count",
        "max",
        "mean",
        "median",
        "min",
        "pct",
        "percent",
        "percentage",
        "price",
        "rate",
        "ratio",
        "score",
        "sum",
        "total",
        "value",
    }
)


def _looks_like_measure(name: str) -> bool:
    if "%" in name:
        return True
    return bool(set(_normalized_name(name).split("_")) & _MEASURE_NAME_TOKENS)


def _is_dimension_like(series: pd.Series) -> bool:
    """True when a column reads as a grouping level rather than a measurement.

    Continuous measures are excluded outright: a `float` column that happens to
    be unique is the classic false primary key (a rate, an amount, a score), and
    no amount of repetition should make it look like an identifier.
    """
    family = _type_family(series)
    if family == "numeric" and not ptypes.is_integer_dtype(series.dtype):
        return False
    if family == "numeric" and _looks_like_measure(str(series.name)):
        return False
    if family not in {"string", "boolean", "datetime", "numeric"}:
        return False
    non_null = int(series.notna().sum())
    if not non_null:
        return False
    try:
        distinct = int(series.nunique(dropna=True))
    except (TypeError, ValueError):
        return False
    if distinct <= 1:
        return False
    return distinct / non_null <= _MAX_DIMENSION_DISTINCT_RATIO


def _key_is_plausible(table: str, frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    """Accept identifier-named keys and genuine composite natural keys.

    Two independent routes qualify a candidate:

    1. *Identifier naming* — the historical rule, which recognises `id`, `code`,
       `uuid` and friends. It is what makes `customer_id` a key.
    2. *Composite natural key* — every component is a dimension-like partition
       (`Location + Period`, `country + year + sex`). Real datasets key on
       business columns far more often than on surrogate ids, and requiring
       identifier vocabulary silently misses all of them.

    Route 2 is deliberately restricted to composites. A lone dimension-like
    column cannot be unique by definition, so any single column reaching this
    point is unique *and* weakly named — which is exactly the accidental key
    (a small lookup table's only text column, a distinct float measure) that
    would otherwise be crowned a hub the whole schema hangs off.
    """
    component_scores = [_key_name_score(table, (column,)) for column in columns]
    if sum(component_scores) / len(component_scores) >= 0.5 and all(
        score >= 0.5 or _type_family(frame[column]) == "string"
        for column, score in zip(columns, component_scores, strict=True)
    ):
        return True
    return len(columns) >= 2 and all(
        score >= 0.5 or _is_dimension_like(frame[column])
        for column, score in zip(columns, component_scores, strict=True)
    )


def _candidate_columns(frame: pd.DataFrame, limit: int = 12) -> list[str]:
    scored: list[tuple[float, str]] = []
    row_count = len(frame)
    for raw_name in frame.columns:
        if not isinstance(raw_name, str):
            continue
        name = raw_name
        series = frame[raw_name]
        non_null = int(series.notna().sum())
        if not row_count or not non_null:
            continue
        try:
            unique = int(series.nunique(dropna=True))
        except (TypeError, ValueError):
            continue
        if unique <= 1:
            continue
        unique_rate = unique / non_null
        identifier_bonus = 1.0 if name.lower().endswith("_id") else 0.0
        scored.append((unique_rate + identifier_bonus, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:limit]]


def _key_metrics(
    frame: pd.DataFrame, columns: Sequence[str]
) -> tuple[float, float, int]:
    if frame.empty:
        return 0.0, 0.0, 0
    values = frame[list(columns)]
    complete = values.notna().all(axis=1)
    complete_count = int(complete.sum())
    completeness = complete_count / len(frame)
    if not complete_count:
        return 0.0, completeness, 0
    try:
        distinct = int(values.loc[complete].drop_duplicates().shape[0])
    except (TypeError, ValueError):
        return 0.0, completeness, 0
    uniqueness = distinct / complete_count
    return uniqueness, completeness, distinct


def discover_key_candidates(
    tables: Mapping[str, pd.DataFrame],
    *,
    max_key_columns: int,
    min_uniqueness: float,
    min_completeness: float,
    max_rows: int,
    sampling: str,
    random_seed: int,
) -> tuple[
    tuple[KeyCandidate, ...],
    tuple[SamplingRecord, ...],
    tuple[AnalysisWarning, ...],
]:
    candidates: list[KeyCandidate] = []
    sampling_records: list[SamplingRecord] = []
    warnings: list[AnalysisWarning] = []
    for table_name, full_frame in tables.items():
        frame = full_frame
        sampled = False
        if sampling == "auto":
            frame, sampled = _sample_frame(
                full_frame, max_rows=max_rows, random_seed=random_seed
            )
        if sampled:
            sampling_records.append(
                SamplingRecord(
                    operation=f"key_search:{table_name}",
                    source_rows=len(full_frame),
                    sampled_rows=len(frame),
                    strategy="deterministic_random_rows",
                    seed=random_seed,
                    reason=(
                        "Composite-key search exceeded the configured exact row budget."
                    ),
                    limitations=(
                        "Sample uniqueness can overestimate full-table uniqueness.",
                        "Confirm sampled candidates before enforcing constraints.",
                    ),
                )
            )
        available = _candidate_columns(frame)
        minimal_keys: list[tuple[str, ...]] = []
        for width in range(1, max_key_columns + 1):
            for columns in itertools.combinations(available, width):
                if any(set(existing).issubset(columns) for existing in minimal_keys):
                    continue
                uniqueness, completeness, distinct = _key_metrics(frame, columns)
                if uniqueness < min_uniqueness or completeness < min_completeness:
                    continue
                name_score = _key_name_score(table_name, columns)
                if not _key_is_plausible(table_name, frame, columns):
                    continue
                confidence = min(
                    1.0,
                    0.55 * uniqueness + 0.3 * completeness + 0.15 * name_score,
                )
                if sampled:
                    confidence *= 0.85
                candidates.append(
                    KeyCandidate(
                        table=table_name,
                        columns=columns,
                        uniqueness_rate=uniqueness,
                        completeness_rate=completeness,
                        confidence=confidence,
                        row_count=len(full_frame),
                        evaluated_row_count=len(frame),
                        distinct_count=distinct,
                        sampled=sampled,
                    )
                )
                minimal_keys.append(columns)
    if sampling_records:
        warnings.append(
            AnalysisWarning(
                code="sampled_key_discovery",
                message=(
                    "Some key candidates were inferred from deterministic samples; "
                    "confirm full-table uniqueness before enforcing constraints."
                ),
            )
        )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (item.table, len(item.columns), item.columns),
        )
    )
    return ordered, tuple(sampling_records), tuple(warnings)


def _sample_frame(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, bool]:
    if len(frame) <= max_rows:
        return frame, False
    return frame.sample(n=max_rows, random_state=random_seed), True


def _best_child_order(
    parent_table: str,
    parent_columns: Sequence[str],
    parent: pd.DataFrame,
    child_columns: Sequence[str],
    child: pd.DataFrame,
) -> tuple[tuple[str, ...], float, float]:
    best: tuple[tuple[str, ...], float, float] | None = None
    for ordered in itertools.permutations(child_columns):
        name_score = _name_similarity(parent_table, parent_columns, ordered)
        type_score = _type_compatibility(parent, parent_columns, child, ordered)
        current = (ordered, name_score, type_score)
        if best is None or (name_score + type_score) > (best[1] + best[2]):
            best = current
    if best is None:
        return (), 0.0, 0.0
    return best


def _relationship_metrics(
    parent: pd.DataFrame,
    parent_columns: Sequence[str],
    child: pd.DataFrame,
    child_columns: Sequence[str],
) -> tuple[float, float, int, int, bool]:
    join_columns = [f"__key_{index}" for index in range(len(parent_columns))]
    parent_values = parent[list(parent_columns)].copy()
    child_values = child[list(child_columns)].copy()
    parent_values.columns = join_columns
    child_values.columns = join_columns
    parent_values = parent_values.dropna().drop_duplicates()
    child_values = child_values.dropna()
    if child_values.empty or parent_values.empty:
        return 0.0, 0.0, len(child_values), len(parent_values), False

    child_with_match = child_values.merge(
        parent_values.assign(__matched=True), how="left", on=join_columns
    )
    matched_rows = int(child_with_match["__matched"].notna().sum())
    orphan_rows = len(child_values) - matched_rows
    row_coverage = matched_rows / len(child_values)

    child_distinct = child_values.drop_duplicates()
    distinct_with_match = child_distinct.merge(
        parent_values.assign(__matched=True), how="left", on=join_columns
    )
    matched_distinct = int(distinct_with_match["__matched"].notna().sum())
    inclusion_rate = matched_distinct / len(child_distinct)

    parent_with_child = parent_values.merge(
        child_distinct.assign(__referenced=True), how="left", on=join_columns
    )
    parent_unmatched = int(parent_with_child["__referenced"].isna().sum())
    child_is_unique = len(child_values) == len(child_distinct)
    return inclusion_rate, row_coverage, orphan_rows, parent_unmatched, child_is_unique


# A shape shared by only two tables is as likely to be a coincidence as a
# design. Three independent tables keyed the same way is a convention.
_MIN_GRAIN_TABLES = 3


def discover_shared_grains(
    tables: Mapping[str, pd.DataFrame],
    keys: Sequence[KeyCandidate],
    *,
    min_uniqueness: float = 0.98,
    min_completeness: float = 0.98,
) -> tuple[SharedGrain, ...]:
    """Group candidate keys that repeat, column-for-column, across tables.

    The discriminator against a star schema is that the shape must be *unique
    in several tables at once*. A dimension's key is unique in the dimension
    and duplicated in every fact that references it, so it never qualifies —
    which is what keeps genuine parent/child schemas on the relationship path.

    Once a shape clears that bar, membership is settled by measurement rather
    than by the naming heuristics that proposed it. Those heuristics exist to
    avoid inventing a key out of nothing; a shape independently unique in three
    or more tables is not nothing, so a fourth table carrying the same columns
    is judged on whether they are actually unique there.
    """
    tables_by_shape: dict[tuple[str, ...], set[str]] = {}
    for key in keys:
        tables_by_shape.setdefault(key.columns, set()).add(key.table)

    grains: list[SharedGrain] = []
    for columns, seed_tables in tables_by_shape.items():
        if len(seed_tables) < _MIN_GRAIN_TABLES:
            continue
        unique_tables = set(seed_tables)
        repeating: set[str] = set()
        for name, frame in tables.items():
            if name in unique_tables or not len(frame):
                continue
            if not all(column in frame.columns for column in columns):
                continue
            uniqueness, completeness, _ = _key_metrics(frame, columns)
            if uniqueness >= min_uniqueness and completeness >= min_completeness:
                unique_tables.add(name)
            else:
                repeating.add(name)
        grains.append(
            SharedGrain(
                columns=columns,
                unique_tables=tuple(sorted(unique_tables)),
                repeating_tables=tuple(sorted(repeating)),
            )
        )
    # Widest, most widely shared grain first: that is the join an analyst
    # reaches for before any narrower one.
    grains.sort(
        key=lambda grain: (-grain.table_count, -len(grain.columns), grain.columns)
    )
    return tuple(grains)


def _is_co_grain_pair(
    grains: Sequence[SharedGrain],
    parent_table: str,
    parent_columns: tuple[str, ...],
    child_table: str,
    tables: Mapping[str, pd.DataFrame],
) -> bool:
    """True when a matching pair is two slices of one conformed grain.

    The exact-shape case is the obvious one. The subset case is subtler and
    just as wrong to report as a foreign key: a table holding a single period
    is unique on `(Location, Dim1)` even though the shared grain is
    `(Location, Period, Dim1)`, and it would otherwise be crowned the parent of
    every table that spans multiple years.

    Requiring *both* tables to carry the grain's full column set is what keeps
    real star schemas intact. A dimension table is unique on its key but has no
    period column at all, so it never matches here and stays on the
    relationship path where it belongs.
    """
    parent_frame = tables.get(parent_table)
    child_frame = tables.get(child_table)
    if parent_frame is None or child_frame is None:
        return False
    parent_set = set(parent_columns)
    for grain in grains:
        if not all(column in parent_frame.columns for column in grain.columns):
            continue
        if not all(column in child_frame.columns for column in grain.columns):
            continue
        if parent_set <= set(grain.columns):
            return True
        # The parent may also be keyed on columns outside the grain and still be
        # a slice of it, when the grain columns it does not key on are pinned to
        # a single value. A table holding one year is trivially unique on
        # `(Location, Dim1)`; that says the table is one year wide, not that it
        # identifies the years in every other table.
        for column in grain.columns:
            if column in parent_set:
                continue
            try:
                parent_distinct = int(parent_frame[column].nunique(dropna=True))
                child_distinct = int(child_frame[column].nunique(dropna=True))
            except (TypeError, ValueError):
                continue
            if parent_distinct <= 1 < child_distinct:
                return True
    return False


def discover_relationship_candidates(
    tables: Mapping[str, pd.DataFrame],
    keys: Sequence[KeyCandidate],
    *,
    min_inclusion: float,
    min_confidence: float,
    max_rows: int,
    sampling: str,
    random_seed: int,
    grains: Sequence[SharedGrain] = (),
) -> tuple[
    tuple[RelationshipCandidate, ...],
    tuple[SamplingRecord, ...],
    tuple[AnalysisWarning, ...],
]:
    relationships: list[RelationshipCandidate] = []
    sampling_records: list[SamplingRecord] = []
    warnings: list[AnalysisWarning] = []
    seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()

    for key in keys:
        parent_full = tables[key.table]
        parent = parent_full
        for child_table, child_full in tables.items():
            if child_table == key.table or len(child_full) == 0:
                continue
            child = child_full
            child_sampled = False
            if sampling == "auto":
                child, child_sampled = _sample_frame(
                    child_full, max_rows=max_rows, random_seed=random_seed
                )
            available = _candidate_columns(child)
            if len(available) < len(key.columns):
                continue
            for unordered_child_columns in itertools.combinations(
                available, len(key.columns)
            ):
                ordered, name_score, type_score = _best_child_order(
                    key.table,
                    key.columns,
                    parent,
                    unordered_child_columns,
                    child,
                )
                if type_score < 0.75 or name_score < 0.2:
                    continue
                if tuple(ordered) == key.columns and _is_co_grain_pair(
                    grains, key.table, key.columns, child_table, tables
                ):
                    # Both sides sit on the same conformed grain. The inclusion
                    # test would pass and report a foreign key, but neither
                    # table owns the other: they are peers measured at the same
                    # coordinate. The grain says this once instead of N x N
                    # times, and says it without inventing a false hierarchy.
                    continue
                identity = (key.table, key.columns, child_table, ordered)
                if identity in seen:
                    continue
                seen.add(identity)
                try:
                    (
                        inclusion_rate,
                        row_coverage,
                        orphan_rows,
                        parent_unmatched,
                        child_is_unique,
                    ) = _relationship_metrics(parent, key.columns, child, ordered)
                except (TypeError, ValueError):
                    continue
                effective_inclusion = inclusion_rate
                if not child_is_unique and name_score < _ONE_TO_ONE_MIN_NAME_SIMILARITY:
                    parent_len = len(parent)
                    parent_coverage = (
                        (parent_len - parent_unmatched) / parent_len
                        if parent_len
                        else 0.0
                    )
                    effective_inclusion *= parent_coverage

                confidence = min(
                    1.0,
                    0.5 * effective_inclusion
                    + 0.18 * type_score
                    + 0.17 * name_score
                    + 0.15 * key.confidence,
                )
                if inclusion_rate < min_inclusion or confidence < min_confidence:
                    continue
                if child_is_unique and name_score < _ONE_TO_ONE_MIN_NAME_SIMILARITY:
                    # Likely coincidental range overlap between two unrelated
                    # unique ID columns, not a real one-to-one link.
                    continue
                sampled = key.sampled or child_sampled
                relationships.append(
                    RelationshipCandidate(
                        parent_table=key.table,
                        parent_columns=key.columns,
                        child_table=child_table,
                        child_columns=ordered,
                        cardinality="one_to_one" if child_is_unique else "one_to_many",
                        inclusion_rate=inclusion_rate,
                        row_coverage=row_coverage,
                        orphan_row_count=orphan_rows,
                        parent_unmatched_count=parent_unmatched,
                        name_similarity=name_score,
                        type_compatibility=type_score,
                        confidence=confidence,
                        sampled=sampled,
                    )
                )
                if child_sampled:
                    operation = (
                        f"relationship:{key.table}{key.columns}->{child_table}{ordered}"
                    )
                    sampling_records.append(
                        SamplingRecord(
                            operation=operation,
                            source_rows=len(parent_full) + len(child_full),
                            sampled_rows=len(parent) + len(child),
                            strategy="deterministic_random_rows",
                            seed=random_seed,
                            reason=(
                                "Relationship inclusion exceeded the configured exact "
                                "row budget."
                            ),
                            limitations=(
                                "Inclusion and orphan rates are estimates from sampled "
                                "child rows.",
                                "Parent coverage metrics are conservative when the "
                                "child table is sampled.",
                            ),
                        )
                    )

    if sampling_records:
        warnings.append(
            AnalysisWarning(
                code="sampled_relationship_discovery",
                message=(
                    "Some relationship confidence scores use deterministic samples; "
                    "review sampling metadata before accepting candidate foreign keys."
                ),
            )
        )
    relationships.sort(
        key=lambda item: (
            -item.confidence,
            item.parent_table,
            item.child_table,
            item.parent_columns,
            item.child_columns,
        )
    )
    return tuple(relationships), tuple(sampling_records), tuple(warnings)


def discover_schema_candidates(
    tables: Mapping[str, pd.DataFrame],
    *,
    mode: AnalysisMode | str,
    max_key_columns: int | None,
    min_key_uniqueness: float,
    min_key_completeness: float,
    min_relationship_inclusion: float,
    min_relationship_confidence: float,
    sampling: str,
    random_seed: int,
) -> SchemaDiscovery:
    """Discover minimal candidate keys and directional inter-table relationships."""
    normalized_mode = AnalysisMode(mode)
    structural_warnings: tuple[AnalysisWarning, ...] = ()
    tables_with_non_string_columns = [
        name
        for name, frame in tables.items()
        if any(not isinstance(column, str) for column in frame.columns)
    ]
    if tables_with_non_string_columns:
        structural_warnings = (
            AnalysisWarning(
                code="non_string_column_names_skipped",
                message=(
                    "Schema discovery skipped non-string column names in: "
                    + ", ".join(sorted(tables_with_non_string_columns))
                    + "."
                ),
            ),
        )
    resolved_max_columns = (
        default_max_key_columns(normalized_mode)
        if max_key_columns is None
        else max_key_columns
    )
    if resolved_max_columns < 1 or resolved_max_columns > 3:
        raise ValueError("max_key_columns must be between 1 and 3")
    row_budget = {
        AnalysisMode.QUICK: 25_000,
        AnalysisMode.STANDARD: 100_000,
        AnalysisMode.DEEP: 250_000,
    }[normalized_mode]
    keys, key_sampling, key_warnings = discover_key_candidates(
        tables,
        max_key_columns=resolved_max_columns,
        min_uniqueness=min_key_uniqueness,
        min_completeness=min_key_completeness,
        max_rows=row_budget,
        sampling=sampling,
        random_seed=random_seed,
    )
    grains = discover_shared_grains(
        tables,
        keys,
        min_uniqueness=min_key_uniqueness,
        min_completeness=min_key_completeness,
    )
    relationships, sampling_records, warnings = discover_relationship_candidates(
        tables,
        keys,
        min_inclusion=min_relationship_inclusion,
        min_confidence=min_relationship_confidence,
        max_rows=row_budget,
        sampling=sampling,
        random_seed=random_seed,
        grains=grains,
    )
    return SchemaDiscovery(
        keys=keys,
        relationships=relationships,
        sampling=key_sampling + sampling_records,
        warnings=structural_warnings + key_warnings + warnings,
        grains=grains,
    )
