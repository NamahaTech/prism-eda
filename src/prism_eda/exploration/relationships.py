"""Find columns that hold the same identifiers — without looking at names.

The question asked of every pair of identifier-like columns, in the same table
or across tables and databases, is: *how many of this column's values also
exist in that one?* It is answered by probing, not by joining:

1. **Sample.** Each column contributes a hash sample of its distinct values
   (taken by the profile stage) in canonical form.
2. **Screen.** A few hundred values per column are looked up in every other
   column — one scan per table, however many columns are asking.
3. **Verify.** Pairs with any hits get the column's full value sample looked
   up, giving a coverage figure with a confidence interval, plus a
   row-weighted figure from the row sample.

A lookup is exact for the values asked about, so the only estimate is the
sampling of *which* values to ask about, and the interval says how tight it is.

Being name-blind has one known failure: small integers overlap by coincidence
(a serial ``id`` 1..N is "contained" in every other serial). Those pairs are
held to stricter evidence and never promoted above "possible" on values alone.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from prism_eda.artifacts import Artifact
from prism_eda.evidence.models import OBSERVATION, Evidence, EvidenceScope, Finding
from prism_eda.exploration.config import ExploreConfig
from prism_eda.exploration.tables import KeyColumn, TableState
from prism_eda.results import AnalysisFailure
from prism_eda.sources.base import Source, SourceError

ColumnId = tuple[str, str]  # (table key, column name)

_GENERIC_TOKENS = {
    "id", "key", "code", "no", "num", "number", "ref", "name", "type", "val",
    "value", "data", "col", "field",
}  # fmt: skip
MAX_RELATIONSHIP_FINDINGS = 30


@dataclass(slots=True)
class Directed:
    """Coverage of one column's values by another."""

    asked: int = 0
    found: int = 0
    rows_asked: int = 0
    rows_found: int = 0

    @property
    def coverage(self) -> float:
        return self.found / self.asked if self.asked else 0.0

    @property
    def row_coverage(self) -> float | None:
        return self.rows_found / self.rows_asked if self.rows_asked else None


@dataclass(slots=True)
class Relationship:
    child: ColumnId
    parent: ColumnId
    forward: Directed
    backward: Directed
    kind: str
    tier: str
    same_table: bool
    cross_source: bool
    weak_values: bool
    format_note: str | None = None
    evidence_id: str | None = None


@dataclass(slots=True)
class Domain:
    id: str
    label: str
    members: list[ColumnId] = field(default_factory=list)
    reference: ColumnId | None = None


@dataclass(slots=True)
class RelationshipResult:
    relationships: list[Relationship] = field(default_factory=list)
    domains: list[Domain] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    failures: list[AnalysisFailure] = field(default_factory=list)
    columns_compared: int = 0
    pairs_screened: int = 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def wilson_interval(found: int, asked: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion that behaves at 0 and 1."""
    if asked <= 0:
        return 0.0, 1.0
    share = found / asked
    denominator = 1 + z * z / asked
    centre = (share + z * z / (2 * asked)) / denominator
    margin = (
        z * math.sqrt(share * (1 - share) / asked + z * z / (4 * asked * asked))
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _tokens(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [token for token in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if token]


def name_affinity(child_column: str, parent_table: str, parent_column: str) -> bool:
    """Whether names *hint* that the child refers to the parent.

    Used only to break a tie that values cannot: a small-integer pair is
    indistinguishable from coincidence, and a shared word root is the one
    extra signal available. It never creates a relationship on its own.
    """
    child = [t for t in _tokens(child_column) if t not in _GENERIC_TOKENS]
    targets = [
        t
        for t in _tokens(parent_table) + _tokens(parent_column)
        if t not in _GENERIC_TOKENS
    ]
    for left in child:
        for right in targets:
            if len(left) >= 3 and left == right:
                return True
            shared = len(_common_prefix(left, right))
            if shared >= 4:
                return True
    return False


def _common_prefix(left: str, right: str) -> str:
    size = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        size += 1
    return left[:size]


def _is_weak(column: KeyColumn, config: ExploreConfig) -> bool:
    """Short all-digit values: the shape that overlaps by coincidence."""
    return (
        column.digit_share >= 0.95 and column.median_length < config.weak_value_length
    )


def _signature(column: KeyColumn) -> dict[str, float]:
    examples = column.examples
    if not examples:
        return {}
    count = len(examples)
    return {
        "zero_led": sum(1 for v in examples if v.strip(" ").startswith("0")) / count,
        "upper": sum(1 for v in examples if any(c.isupper() for c in v)) / count,
        "spaced": sum(1 for v in examples if v != v.strip(" ")) / count,
        "length": sorted(len(v) for v in examples)[count // 2],
    }


def format_note(
    child: KeyColumn, parent: KeyColumn, child_name: str, parent_name: str
) -> str | None:
    """Describe a systematic difference in how two columns write one value."""
    left, right = _signature(child), _signature(parent)
    if not left or not right:
        return None
    reasons = []
    if abs(left["zero_led"] - right["zero_led"]) >= 0.5:
        reasons.append("leading zeros")
    if abs(left["upper"] - right["upper"]) >= 0.5:
        reasons.append("letter case")
    if abs(left["spaced"] - right["spaced"]) >= 0.5:
        reasons.append("surrounding spaces")
    if not reasons and child.kind != parent.kind:
        return (
            f"{child_name} is stored as {child.kind} and {parent_name} as "
            f"{parent.kind}; a join needs a cast."
        )
    if not reasons:
        return None
    return (
        f"The two columns write the same identifiers differently "
        f"({', '.join(reasons)}): {child_name} looks like "
        f"'{child.examples[0]}', {parent_name} like '{parent.examples[0]}'. "
        "A plain equality join between them matches nothing."
    )


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------


def _eligible(value: str, target: KeyColumn) -> bool:
    if target.kind == "integer" and not value.isdigit():
        return False
    if target.max_length is not None and len(value) > target.max_length:
        return False
    return True


def _probe_tables(
    states: Sequence[TableState],
    sources: dict[str, Source],
    wanted: dict[ColumnId, set[str]],
    result: RelationshipResult,
    tick: Callable[[str], None],
    label: str,
) -> dict[ColumnId, set[str]]:
    """Look values up, one request per table covering all its columns."""
    present: dict[ColumnId, set[str]] = {}
    for state in states:
        requests = {}
        for column in state.key_columns:
            values = wanted.get((state.key, column.name))
            if values:
                requests[column.name] = (state.column(column.name), sorted(values))
        if not requests:
            continue
        tick(f"{label}: {state.display}")
        try:
            answer = sources[state.info.source].probe(state.info, requests)
        except SourceError as error:
            result.failures.append(
                AnalysisFailure(
                    stage="relationship_probe",
                    message=str(error),
                    recoverable=True,
                    table=state.display,
                )
            )
            continue
        for name, found in answer.items():
            present[(state.key, name)] = set(found)
    return present


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover_relationships(
    states: Sequence[TableState],
    sources: dict[str, Source],
    config: ExploreConfig,
    tick: Callable[[str], None],
) -> RelationshipResult:
    result = RelationshipResult()
    by_key = {state.key: state for state in states}
    columns: dict[ColumnId, KeyColumn] = {
        (state.key, column.name): column
        for state in states
        for column in state.key_columns
    }
    result.columns_compared = len(columns)
    if len(columns) < 2:
        return result

    def display(column_id: ColumnId) -> str:
        return f"{by_key[column_id[0]].display}.{column_id[1]}"

    # -- screen ------------------------------------------------------------
    screens = {
        column_id: column.values[: config.screen_values]
        for column_id, column in columns.items()
    }
    owners: dict[str, list[ColumnId]] = defaultdict(list)
    for column_id, values in screens.items():
        for value in values:
            owners[value].append(column_id)
    pool = set(owners)
    wanted: dict[ColumnId, set[str]] = {}
    for column_id, column in columns.items():
        wanted[column_id] = {value for value in pool if _eligible(value, column)}
    present = _probe_tables(states, sources, wanted, result, tick, "screening")

    hits: dict[tuple[ColumnId, ColumnId], int] = defaultdict(int)
    for target, found in present.items():
        for value in found:
            for asker in owners.get(value, ()):
                if asker != target:
                    hits[(asker, target)] += 1
    candidates = {
        pair
        for pair, count in hits.items()
        if count >= 2 or (count >= 1 and len(screens[pair[0]]) <= 64)
    }
    result.pairs_screened = len(columns) * (len(columns) - 1)
    if not candidates:
        return result

    # -- verify ------------------------------------------------------------
    wanted = defaultdict(set)
    for asker, target in candidates:
        column = columns[asker]
        target_column = columns[target]
        wanted[target].update(
            value for value in column.values if _eligible(value, target_column)
        )
        wanted[target].update(
            value for value in set(column.row_values) if _eligible(value, target_column)
        )
    present = _probe_tables(states, sources, wanted, result, tick, "verifying")

    directed: dict[tuple[ColumnId, ColumnId], Directed] = {}
    for asker, target in candidates:
        if target not in present:
            continue
        found = present[target]
        column = columns[asker]
        directed[(asker, target)] = Directed(
            asked=len(column.values),
            found=sum(1 for value in column.values if value in found),
            rows_asked=len(column.row_values),
            rows_found=sum(1 for value in column.row_values if value in found),
        )

    # -- pair up and judge -------------------------------------------------
    seen: set[frozenset[ColumnId]] = set()
    for left, right in sorted(directed):
        pair = frozenset((left, right))
        if pair in seen:
            continue
        seen.add(pair)
        relationship = _judge(
            left,
            right,
            directed.get((left, right), Directed()),
            directed.get((right, left), Directed()),
            columns,
            by_key,
            config,
        )
        if relationship is not None:
            result.relationships.append(relationship)

    result.relationships.sort(
        key=lambda item: (
            {"strong": 0, "likely": 1, "possible": 2}[item.tier],
            -item.forward.coverage,
            display(item.child),
        )
    )
    _domains(result, columns, by_key)
    _evidence(result, columns, by_key, display)
    return result


def _judge(
    left: ColumnId,
    right: ColumnId,
    left_in_right: Directed,
    right_in_left: Directed,
    columns: dict[ColumnId, KeyColumn],
    by_key: dict[str, TableState],
    config: ExploreConfig,
) -> Relationship | None:
    a, b = columns[left], columns[right]
    # Orientation: the unique side is the reference; failing that, the side
    # that contains more of the other.
    if a.unique != b.unique:
        child, parent = (right, left) if a.unique else (left, right)
    elif left_in_right.coverage >= right_in_left.coverage:
        child, parent = left, right
    else:
        child, parent = right, left
    forward = left_in_right if child == left else right_in_left
    backward = right_in_left if child == left else left_in_right
    child_column, parent_column = columns[child], columns[parent]

    best_found = max(forward.found, backward.found)
    best_coverage = max(forward.coverage, backward.coverage)
    weak = _is_weak(child_column, config) or _is_weak(parent_column, config)

    if weak:
        # Two serials "contain" each other by construction.
        if child_column.unique and parent_column.unique:
            return None
        if not parent_column.unique or forward.coverage < 0.9 or forward.found < 10:
            return None
        affinity = name_affinity(child[1], by_key[parent[0]].info.name, parent[1])
        # Without a supporting name, also ask that the child uses a good part
        # of the parent's keys: a real reference to a small list usually does,
        # an unrelated counter that happens to fit inside it usually does not.
        if not affinity and backward.coverage < 0.5:
            return None
        tier = "likely" if affinity else "possible"
    else:
        if best_found < config.min_matches or best_coverage < config.min_coverage:
            return None
        tier = "strong"

    if parent_column.unique and child_column.unique:
        kind = "one_to_one"
    elif parent_column.unique:
        kind = "many_to_one"
    else:
        kind = "shared_values"

    child_state, parent_state = by_key[child[0]], by_key[parent[0]]
    return Relationship(
        child=child,
        parent=parent,
        forward=forward,
        backward=backward,
        kind=kind,
        tier=tier,
        same_table=child[0] == parent[0],
        cross_source=child_state.info.source != parent_state.info.source,
        weak_values=weak,
        format_note=format_note(
            child_column,
            parent_column,
            f"{child_state.display}.{child[1]}",
            f"{parent_state.display}.{parent[1]}",
        ),
    )


def _domains(
    result: RelationshipResult,
    columns: dict[ColumnId, KeyColumn],
    by_key: dict[str, TableState],
) -> None:
    """Group columns that share identifiers into domains (connected sets)."""
    parent: dict[ColumnId, ColumnId] = {}

    def find(item: ColumnId) -> ColumnId:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for relationship in result.relationships:
        if relationship.tier == "possible":
            continue
        parent[find(relationship.child)] = find(relationship.parent)

    groups: dict[ColumnId, list[ColumnId]] = defaultdict(list)
    for item in list(parent):
        groups[find(item)].append(item)
    ordered = sorted(
        groups.values(),
        key=lambda members: (-len(members), sorted(members)[0]),
    )
    for index, members in enumerate(ordered, start=1):
        members.sort(key=lambda item: (-columns[item].distinct, item))
        unique = [item for item in members if columns[item].unique]
        reference = (
            max(unique, key=lambda item: columns[item].distinct) if unique else None
        )
        anchor = reference or members[0]
        result.domains.append(
            Domain(
                id=f"D{index}",
                label=f"{by_key[anchor[0]].display}.{anchor[1]}",
                members=members,
                reference=reference,
            )
        )


def domain_of(result: RelationshipResult, column_id: ColumnId) -> Domain | None:
    for domain in result.domains:
        if column_id in domain.members:
            return domain
    return None


def _evidence(
    result: RelationshipResult,
    columns: dict[ColumnId, KeyColumn],
    by_key: dict[str, TableState],
    display: Callable[[ColumnId], str],
) -> None:
    def describe(column_id: ColumnId) -> dict[str, Any]:
        column = columns[column_id]
        state = by_key[column_id[0]]
        return {
            "table": state.display,
            "column": column_id[1],
            "source": state.info.source,
            "distinct": column.distinct,
            "unique": column.unique,
            "kind": column.kind,
            "examples": column.examples[:3],
        }

    for relationship in result.relationships:
        forward, backward = relationship.forward, relationship.backward
        low, high = wilson_interval(forward.found, forward.asked)
        value = {
            "child": describe(relationship.child),
            "parent": describe(relationship.parent),
            "kind": relationship.kind,
            "tier": relationship.tier,
            "same_table": relationship.same_table,
            "cross_source": relationship.cross_source,
            "coverage": forward.coverage,
            "coverage_interval": [low, high],
            "values_checked": forward.asked,
            "values_found": forward.found,
            "row_coverage": forward.row_coverage,
            "rows_checked": forward.rows_asked,
            "reverse_coverage": backward.coverage if backward.asked else None,
            "reverse_values_checked": backward.asked,
            "weak_values": relationship.weak_values,
            "format_note": relationship.format_note,
            "primary": _is_primary(relationship, result),
        }
        item = Evidence.create(
            kind="explore_relationship",
            scope=EvidenceScope(
                table=by_key[relationship.child[0]].display,
                columns=(relationship.child[1],),
            ),
            value=value,
            method="canonical_value_probe_v1",
            description=(
                f"Share of {display(relationship.child)} values that exist in "
                f"{display(relationship.parent)}."
            ),
            confidence={"strong": 0.95, "likely": 0.75, "possible": 0.45}[
                relationship.tier
            ],
            assumptions=(
                "Values are compared in canonical form: trimmed, lower-cased, "
                "leading zeros removed.",
                "Coverage is measured on a hash sample of distinct values; the "
                "interval is a 95% Wilson interval.",
                "A relationship found from values is a candidate until confirmed.",
            ),
        )
        relationship.evidence_id = item.id
        result.evidence.append(item)

    for domain in result.domains:
        result.evidence.append(
            Evidence.create(
                kind="explore_domain",
                scope=EvidenceScope(),
                value={
                    "id": domain.id,
                    "label": domain.label,
                    "reference": describe(domain.reference)
                    if domain.reference
                    else None,
                    "members": [describe(member) for member in domain.members],
                    "tables": sorted(
                        {by_key[member[0]].display for member in domain.members}
                    ),
                },
                method="connected_columns_v1",
                description=(
                    f"Columns that hold the same identifiers as {domain.label}."
                ),
                confidence=0.9,
            )
        )

    _findings(result, columns, by_key, display)
    _graph_artifact(result, by_key)


def _is_primary(relationship: Relationship, result: RelationshipResult) -> bool:
    """Whether a pair belongs in the short list a reader looks at first.

    Five columns holding account numbers relate in ten pairs, but a reader
    needs four: each column against the reference list. A pair is primary when
    it involves its domain's reference, when its domain has none, or when it
    links two columns of one table.
    """
    if relationship.same_table or relationship.tier == "possible":
        return True
    domain = domain_of(result, relationship.child)
    if domain is None or domain.reference is None:
        return True
    return domain.reference in (relationship.child, relationship.parent)


def _findings(
    result: RelationshipResult,
    columns: dict[ColumnId, KeyColumn],
    by_key: dict[str, TableState],
    display: Callable[[ColumnId], str],
) -> None:
    ranked = sorted(
        (item for item in result.relationships if item.tier != "possible"),
        key=lambda item: (not _is_primary(item, result), -by_key[item.child[0]].rows),
    )
    format_reported: set[ColumnId] = set()
    for relationship in ranked[:MAX_RELATIONSHIP_FINDINGS]:
        assert relationship.evidence_id is not None
        evidence_ids = (relationship.evidence_id,)
        child, parent = display(relationship.child), display(relationship.parent)
        forward = relationship.forward
        if (
            relationship.format_note
            and "cast" not in relationship.format_note
            and not format_reported.intersection(
                (relationship.child, relationship.parent)
            )
        ):
            # One finding per odd column, not one per column it disagrees with.
            format_reported.update((relationship.child, relationship.parent))
            result.findings.append(
                Finding.create(
                    title=f"Same identifiers, different format: {child} and {parent}",
                    summary=relationship.format_note,
                    severity="high",
                    confidence=0.9,
                    evidence_ids=evidence_ids,
                    recommendation=(
                        "Normalise both sides (trim, case, zero-padding) before "
                        "joining, or the join silently drops rows."
                    ),
                )
            )
        if not _is_primary(relationship, result):
            continue
        if relationship.kind == "shared_values" and not relationship.same_table:
            continue
        if relationship.same_table:
            continue  # links inside one table are reported by the network stage
        missing = 1 - forward.coverage
        if missing < 0.001 or forward.asked < 30:
            continue
        low, high = wilson_interval(forward.found, forward.asked)
        rows = forward.row_coverage
        row_text = (
            f" By rows, {rows:.1%} of {child} rows have a match."
            if rows is not None
            else ""
        )
        if forward.coverage >= 0.9:
            result.findings.append(
                Finding.create(
                    title=f"Unmatched references from {child} to {parent}",
                    summary=(
                        f"{missing:.1%} of distinct {child} values do not exist in "
                        f"{parent} (95% interval {1 - high:.1%}–{1 - low:.1%}, "
                        f"{forward.asked:,} values checked).{row_text}"
                    ),
                    severity="medium" if missing >= 0.02 else "low",
                    confidence=0.85,
                    evidence_ids=evidence_ids,
                    recommendation=(
                        "Rows that reference a missing record drop out of inner "
                        "joins; decide whether they are late-arriving, deleted, "
                        "or invalid."
                    ),
                )
            )
        else:
            result.findings.append(
                Finding.create(
                    title=f"{child} is only partly covered by {parent}",
                    summary=(
                        f"{forward.coverage:.1%} of distinct {child} values exist "
                        f"in {parent} (95% interval {low:.1%}–{high:.1%}, "
                        f"{forward.asked:,} values checked); the other "
                        f"{missing:.1%} refer to records this data does not hold."
                        f"{row_text}"
                    ),
                    severity="info",
                    confidence=0.85,
                    evidence_ids=evidence_ids,
                    recommendation=(
                        "Anything derived by joining these two only describes "
                        "the covered share; the rest is outside the data."
                    ),
                    category=OBSERVATION,
                )
            )


def _graph_artifact(result: RelationshipResult, by_key: dict[str, TableState]) -> None:
    if not result.relationships:
        return
    tables = sorted(
        {by_key[item.child[0]].display for item in result.relationships}
        | {by_key[item.parent[0]].display for item in result.relationships}
    )
    rows = {state.display: state.rows for state in by_key.values()}
    result.artifacts.append(
        Artifact.create(
            kind="relationship_graph",
            title="How the tables connect",
            data={
                "nodes": [{"id": name, "rows": rows.get(name, 0)} for name in tables],
                "edges": [
                    {
                        "source": by_key[item.child[0]].display,
                        "target": by_key[item.parent[0]].display,
                        "source_column": item.child[1],
                        "target_column": item.parent[1],
                        "coverage": item.forward.coverage,
                        "kind": item.kind,
                        "tier": item.tier,
                        "same_table": item.same_table,
                    }
                    for item in result.relationships
                ],
            },
            evidence_ids=tuple(
                item.evidence_id for item in result.relationships if item.evidence_id
            ),
        )
    )
