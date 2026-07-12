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
class SchemaDiscovery:
    keys: tuple[KeyCandidate, ...]
    relationships: tuple[RelationshipCandidate, ...]
    sampling: tuple[SamplingRecord, ...]
    warnings: tuple[AnalysisWarning, ...]


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


def _key_is_plausible(table: str, frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    component_scores = [_key_name_score(table, (column,)) for column in columns]
    if sum(component_scores) / len(component_scores) < 0.5:
        return False
    return all(
        score >= 0.5 or _type_family(frame[column]) == "string"
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


def discover_relationship_candidates(
    tables: Mapping[str, pd.DataFrame],
    keys: Sequence[KeyCandidate],
    *,
    min_inclusion: float,
    min_confidence: float,
    max_rows: int,
    sampling: str,
    random_seed: int,
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
                    parent_coverage = (len(parent) - parent_unmatched) / len(parent) if len(parent) else 0.0
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
    relationships, sampling_records, warnings = discover_relationship_candidates(
        tables,
        keys,
        min_inclusion=min_relationship_inclusion,
        min_confidence=min_relationship_confidence,
        max_rows=row_budget,
        sampling=sampling,
        random_seed=random_seed,
    )
    return SchemaDiscovery(
        keys=keys,
        relationships=relationships,
        sampling=key_sampling + sampling_records,
        warnings=structural_warnings + key_warnings + warnings,
    )
