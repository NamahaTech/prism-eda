"""Task recipe for candidate key and relationship discovery."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any

import pandas as pd

from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog
from prism_eda.catalog.relationships import (
    KeyCandidate,
    RelationshipCandidate,
    discover_schema_candidates,
)
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import (
    Evidence,
    EvidenceScope,
    Finding,
    sort_findings,
)
from prism_eda.results import AnalysisResult, AnalysisStatus, AnalysisWarning


def _truncate(value: str, length: int) -> str:
    if len(value) <= length:
        return value
    return f"{value[: length - 1]}…"


def _table_layers(
    table_names: list[str], relationships: tuple[RelationshipCandidate, ...]
) -> list[list[str]]:
    """Create deterministic left-to-right layers without a graph dependency."""
    connected = {
        name
        for relationship in relationships
        for name in (relationship.parent_table, relationship.child_table)
    }
    isolated = [name for name in table_names if name not in connected]
    if not connected:
        return [table_names]

    adjacency: dict[str, set[str]] = {name: set() for name in connected}
    incoming: dict[str, set[str]] = {name: set() for name in connected}
    for relationship in relationships:
        adjacency[relationship.parent_table].add(relationship.child_table)
        incoming[relationship.child_table].add(relationship.parent_table)

    indegree = {name: len(parents) for name, parents in incoming.items()}
    queue = deque(sorted(name for name, degree in indegree.items() if degree == 0))
    depths = {name: 0 for name in connected}
    processed: list[str] = []
    while queue:
        name = queue.popleft()
        processed.append(name)
        for child in sorted(adjacency[name]):
            depths[child] = max(depths[child], depths[name] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(processed) == len(connected):
        layer_by_name = depths
    else:
        # Cycles are plausible in inferred schemas. Role balance still separates
        # heavily referenced entities from mostly referencing event tables.
        scores = {
            name: len(adjacency[name]) - len(incoming[name]) for name in connected
        }
        distinct_scores = sorted(set(scores.values()), reverse=True)
        layer_by_score = {score: index for index, score in enumerate(distinct_scores)}
        layer_by_name = {name: layer_by_score[scores[name]] for name in connected}

    layers = [
        sorted(name for name in connected if layer_by_name[name] == index)
        for index in range(max(layer_by_name.values()) + 1)
    ]
    layers = [layer for layer in layers if layer]

    # A few barycentric sweeps reduce crossings while preserving determinism.
    for _ in range(3):
        positions = {
            name: (layer_index, item_index)
            for layer_index, layer in enumerate(layers)
            for item_index, name in enumerate(layer)
        }
        for index in range(1, len(layers)):
            layers[index].sort(
                key=lambda name: (
                    sum(positions[parent][1] for parent in incoming[name])
                    / max(1, len(incoming[name])),
                    name,
                )
            )
        positions = {
            name: (layer_index, item_index)
            for layer_index, layer in enumerate(layers)
            for item_index, name in enumerate(layer)
        }
        for index in range(len(layers) - 2, -1, -1):
            layers[index].sort(
                key=lambda name: (
                    sum(positions[child][1] for child in adjacency[name])
                    / max(1, len(adjacency[name])),
                    name,
                )
            )

    if isolated:
        layers.append(sorted(isolated))
    return layers


def _cardinality_marks(
    x: float, y: float, side: str, cardinality: str
) -> tuple[tuple[float, float], list[dict[str, float]]]:
    directions = {
        "left": (-1.0, 0.0, 0.0, 1.0),
        "right": (1.0, 0.0, 0.0, 1.0),
        "top": (0.0, -1.0, 1.0, 0.0),
        "bottom": (0.0, 1.0, 1.0, 0.0),
    }
    dx, dy, px, py = directions[side]

    def line(distance: float, half_length: float) -> dict[str, float]:
        center_x = x + dx * distance
        center_y = y + dy * distance
        return {
            "x1": center_x - px * half_length,
            "y1": center_y - py * half_length,
            "x2": center_x + px * half_length,
            "y2": center_y + py * half_length,
        }

    marks = [line(4, 6)]
    if cardinality == "one":
        marks.append(line(10, 6))
    else:
        apex_x = x + dx * 14
        apex_y = y + dy * 14
        for offset in (-7.0, 0.0, 7.0):
            marks.append(
                {
                    "x1": apex_x,
                    "y1": apex_y,
                    "x2": x + dx * 6 + px * offset,
                    "y2": y + dy * 6 + py * offset,
                }
            )
    return (x + dx * 18, y + dy * 18), marks


def _key_evidence(candidate: KeyCandidate) -> Evidence:
    return Evidence.create(
        kind="candidate_key",
        scope=EvidenceScope(table=candidate.table, columns=candidate.columns),
        value={
            "uniqueness_rate": candidate.uniqueness_rate,
            "completeness_rate": candidate.completeness_rate,
            "distinct_count": candidate.distinct_count,
            "row_count": candidate.row_count,
            "evaluated_row_count": candidate.evaluated_row_count,
            "sampled": candidate.sampled,
        },
        method="minimal_key_search_v1",
        description=(
            f"Candidate key {candidate.table}.{', '.join(candidate.columns)}."
        ),
        confidence=candidate.confidence,
        assumptions=(
            "Uniqueness and completeness support a key candidate but do not prove "
            "business-level identity or stability over time.",
            *(
                (
                    "This candidate was measured on a deterministic row sample and "
                    "requires full-table confirmation.",
                )
                if candidate.sampled
                else ()
            ),
        ),
        metadata={"candidate_type": "primary_key"},
    )


def _relationship_evidence(candidate: RelationshipCandidate) -> Evidence:
    return Evidence.create(
        kind="candidate_relationship",
        scope=EvidenceScope(
            table=candidate.child_table, columns=candidate.child_columns
        ),
        value={
            "parent_table": candidate.parent_table,
            "parent_columns": candidate.parent_columns,
            "child_table": candidate.child_table,
            "child_columns": candidate.child_columns,
            "cardinality": candidate.cardinality,
            "inclusion_rate": candidate.inclusion_rate,
            "row_coverage": candidate.row_coverage,
            "orphan_row_count": candidate.orphan_row_count,
            "parent_unmatched_count": candidate.parent_unmatched_count,
            "name_similarity": candidate.name_similarity,
            "type_compatibility": candidate.type_compatibility,
            "sampled": candidate.sampled,
        },
        method="typed_inclusion_dependency_v1",
        description=(
            f"Candidate relationship {candidate.parent_table}."
            f"{', '.join(candidate.parent_columns)} -> {candidate.child_table}."
            f"{', '.join(candidate.child_columns)}."
        ),
        confidence=candidate.confidence,
        assumptions=(
            "Value inclusion, compatible types, names, and parent uniqueness are "
            "evidence for a relationship but do not establish database constraints.",
        ),
        metadata={"candidate_type": "foreign_key"},
    )


def _findings(
    relationships: tuple[RelationshipCandidate, ...],
    evidence: tuple[Evidence, ...],
) -> tuple[Finding, ...]:
    relationship_evidence = {
        (
            item.value["parent_table"],
            tuple(item.value["parent_columns"]),
            item.value["child_table"],
            tuple(item.value["child_columns"]),
        ): item
        for item in evidence
        if item.kind == "candidate_relationship"
    }
    findings: list[Finding] = []
    for relationship in relationships:
        item = relationship_evidence[
            (
                relationship.parent_table,
                relationship.parent_columns,
                relationship.child_table,
                relationship.child_columns,
            )
        ]
        parent = f"{relationship.parent_table}.{', '.join(relationship.parent_columns)}"
        child = f"{relationship.child_table}.{', '.join(relationship.child_columns)}"
        findings.append(
            Finding.create(
                title=(
                    f"Candidate {relationship.cardinality.replace('_', '-')} "
                    f"relationship: {child} → {parent}"
                ),
                summary=(
                    f"{child} is {relationship.inclusion_rate:.1%} contained in "
                    f"{parent}; confidence is {relationship.confidence:.1%}."
                ),
                severity="low" if relationship.orphan_row_count == 0 else "medium",
                confidence=relationship.confidence,
                evidence_ids=(item.id,),
                recommendation=(
                    "Confirm the business meaning and expected join cardinality before "
                    "treating this candidate as a foreign key."
                ),
            )
        )
        if relationship.orphan_row_count:
            findings.append(
                Finding.create(
                    title=f"Possible orphan rows in {relationship.child_table}",
                    summary=(
                        f"{relationship.orphan_row_count:,} evaluated child rows did "
                        f"not match {parent}."
                    ),
                    severity="high" if relationship.row_coverage < 0.95 else "medium",
                    confidence=relationship.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Review unmatched values before enforcing or relying on the "
                        "candidate relationship."
                    ),
                )
            )
    return tuple(sort_findings(findings))


def _schema_verdict(
    catalog: DatasetCatalog,
    relationships: tuple[RelationshipCandidate, ...],
    analysis: dict[str, dict[str, Any]],
) -> str | None:
    """One plain-language headline that leads with the schema's structure.

    Signal over noise: name the hub tables everything hangs off, rather than
    make the analyst infer them from a flat list of dozens of relationships.
    """
    if not relationships:
        return None
    hubs = sorted(
        (name for name, info in analysis.items() if info["referenced_by"] >= 2),
        key=lambda name: (-analysis[name]["referenced_by"], name),
    )
    rel_count = len(relationships)
    if hubs:
        top = hubs[:2]
        names = " and ".join(_truncate(name, 32) for name in top)
        ref = analysis[top[0]]["referenced_by"]
        lead = (
            f"{names} {'are' if len(top) > 1 else 'is'} the hub"
            f"{'s' if len(top) > 1 else ''} — referenced by "
            f"{ref} of {catalog.table_count} tables."
        )
        return (
            f"{lead} {rel_count} candidate relationship(s) across "
            f"{catalog.table_count} tables; filter or focus the diagram to read them."
        )
    return (
        f"{rel_count} candidate relationship(s) across {catalog.table_count} tables — "
        "no single hub table; review them by pair."
    )


def _pagerank(
    nodes: list[str],
    out_edges: dict[str, set[str]],
    *,
    damping: float = 0.85,
    iterations: int = 60,
) -> dict[str, float]:
    """Plain power-iteration PageRank (no third-party dependency).

    Edges run child -> parent, so importance accumulates on the tables that many
    others reference — the schema's structural anchors.
    """
    count = len(nodes)
    if count == 0:
        return {}
    rank = {node: 1.0 / count for node in nodes}
    base = (1.0 - damping) / count
    for _ in range(iterations):
        updated = dict.fromkeys(nodes, base)
        dangling = damping * sum(
            rank[node] for node in nodes if not out_edges.get(node)
        ) / count
        for node in nodes:
            updated[node] += dangling
        for node in nodes:
            targets = out_edges.get(node)
            if targets:
                share = damping * rank[node] / len(targets)
                for target in targets:
                    updated[target] += share
        rank = updated
    return rank


def _graph_analysis(
    catalog: DatasetCatalog,
    relationships: tuple[RelationshipCandidate, ...],
) -> dict[str, dict[str, Any]]:
    """Classify each table's role in the candidate FK graph and rank importance.

    Referenced-by count (in-degree) marks lookup/dimension/hub tables; the
    references count (out-degree) marks fact/transaction tables. This is what
    lets the report lead with the few tables that actually structure the schema
    instead of dumping every inclusion dependency as an equal peer.
    """
    table_names = [table.name for table in catalog.tables]
    column_counts = {table.name: table.column_count for table in catalog.tables}
    referenced_by: dict[str, set[str]] = {name: set() for name in table_names}
    references: dict[str, set[str]] = {name: set() for name in table_names}
    for relationship in relationships:
        parent, child = relationship.parent_table, relationship.child_table
        if parent in referenced_by and child in references:
            referenced_by[parent].add(child)
            references[child].add(parent)

    rank = _pagerank(table_names, references)
    max_rank = max(rank.values(), default=0.0) or 1.0
    hub_threshold = max(3, math.ceil(len(table_names) * 0.4))

    analysis: dict[str, dict[str, Any]] = {}
    for name in table_names:
        in_degree = len(referenced_by[name])
        out_degree = len(references[name])
        if in_degree == 0 and out_degree == 0:
            role = "standalone"
        elif in_degree >= hub_threshold:
            # Referenced by many tables is the defining hub signature, even if it
            # also references a peer (e.g. two mutually-keyed dimension tables).
            role = "hub"
        elif out_degree == 0:
            role = "dimension"
        elif in_degree == 0:
            # A thin table that only points at others is a junction/bridge.
            role = (
                "junction"
                if out_degree >= 2 and column_counts.get(name, 99) <= out_degree + 3
                else "fact"
            )
        else:
            role = "bridge"
        analysis[name] = {
            "role": role,
            "referenced_by": in_degree,
            "references": out_degree,
            "is_hub": role == "hub",
            "importance": rank.get(name, 0.0) / max_rank,
        }
    return analysis


def _graph_artifact(
    catalog: DatasetCatalog,
    keys: tuple[KeyCandidate, ...],
    relationships: tuple[RelationshipCandidate, ...],
    evidence: tuple[Evidence, ...],
    analysis: dict[str, dict[str, Any]],
) -> Artifact:
    key_by_table: dict[str, list[KeyCandidate]] = {}
    for key in keys:
        key_by_table.setdefault(key.table, []).append(key)

    references_by_table: dict[str, list[RelationshipCandidate]] = defaultdict(list)
    for relationship in relationships:
        references_by_table[relationship.child_table].append(relationship)

    node_width = 360
    node_data: dict[str, dict] = {}
    for table in catalog.tables:
        key_role_rows = [
            {
                "kind": "PK",
                "label": _truncate(" + ".join(key.columns), 34),
                "full_label": " + ".join(key.columns),
                "detail": f"{key.confidence:.0%}{' · sampled' if key.sampled else ''}",
            }
            for key in key_by_table.get(table.name, [])
        ]
        key_role_rows.extend(
            {
                "kind": "FK",
                "label": _truncate(" + ".join(reference.child_columns), 30),
                "full_label": " + ".join(reference.child_columns),
                "detail": f"→ {_truncate(reference.parent_table, 20)}",
            }
            for reference in sorted(
                references_by_table.get(table.name, []),
                key=lambda item: (
                    item.child_columns,
                    item.parent_table,
                    item.parent_columns,
                ),
            )
        )
        visible_rows = key_role_rows[:5]
        overflow_count = len(key_role_rows) - len(visible_rows)
        table_analysis = analysis.get(table.name, {})
        node_data[table.name] = {
            "table": table.name,
            "display_table": _truncate(table.name, 33),
            "width": node_width,
            "height": 130
            + max(1, len(visible_rows)) * 34
            + (18 if overflow_count else 0),
            "row_count": table.row_count,
            "column_count": table.column_count,
            "rows": visible_rows,
            "all_rows": key_role_rows,
            "overflow_count": overflow_count,
            "role": table_analysis.get("role", "standalone"),
            "referenced_by": table_analysis.get("referenced_by", 0),
            "references": table_analysis.get("references", 0),
            "is_hub": table_analysis.get("is_hub", False),
            "importance": table_analysis.get("importance", 0.0),
        }

    table_names = [table.name for table in catalog.tables]
    layers = _table_layers(table_names, relationships)
    margin = 64
    horizontal_gap = 360
    vertical_gap = 78
    positions: dict[str, tuple[float, float]] = {}
    layer_by_table: dict[str, int] = {}

    if relationships:
        layer_heights = [
            sum(node_data[name]["height"] for name in layer)
            + max(0, len(layer) - 1) * vertical_gap
            for layer in layers
        ]
        content_height = max(layer_heights)
        for layer_index, (layer, layer_height) in enumerate(
            zip(layers, layer_heights, strict=True)
        ):
            layer_y = margin + (content_height - layer_height) / 2
            layer_x = margin + layer_index * (node_width + horizontal_gap)
            for name in layer:
                positions[name] = (layer_x, layer_y)
                layer_by_table[name] = layer_index
                layer_y += node_data[name]["height"] + vertical_gap
        width = (
            2 * margin
            + len(layers) * node_width
            + max(0, len(layers) - 1) * horizontal_gap
        )
        height = 2 * margin + content_height
    else:
        columns = min(3, max(1, math.ceil(math.sqrt(catalog.table_count))))
        horizontal_gap = 72
        row_gap = 72
        grid_row_count = math.ceil(catalog.table_count / columns)
        row_heights = [0.0] * grid_row_count
        for index, name in enumerate(table_names):
            row_heights[index // columns] = max(
                row_heights[index // columns], node_data[name]["height"]
            )
        row_offsets: list[float] = []
        current_y = float(margin)
        for row_height in row_heights:
            row_offsets.append(current_y)
            current_y += row_height + row_gap
        for index, name in enumerate(table_names):
            row, column = divmod(index, columns)
            positions[name] = (
                margin + column * (node_width + horizontal_gap),
                row_offsets[row],
            )
            layer_by_table[name] = column
        width = 2 * margin + columns * node_width + max(0, columns - 1) * horizontal_gap
        height = current_y - row_gap + margin

    for name, (node_x, node_y) in positions.items():
        node_data[name]["x"] = node_x
        node_data[name]["y"] = node_y

    edge_work: list[dict] = []
    side_ports: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for index, relationship in enumerate(relationships):
        parent = node_data[relationship.parent_table]
        child = node_data[relationship.child_table]
        parent_layer = layer_by_table[relationship.parent_table]
        child_layer = layer_by_table[relationship.child_table]
        if parent_layer < child_layer:
            source_side, target_side = "right", "left"
        elif parent_layer > child_layer:
            source_side, target_side = "left", "right"
        elif parent["y"] < child["y"]:
            source_side, target_side = "bottom", "top"
        else:
            source_side, target_side = "top", "bottom"
        parent_center = parent["y"] + parent["height"] / 2
        child_center = child["y"] + child["height"] / 2
        side_ports[(relationship.parent_table, source_side)].append(
            (index, child_center)
        )
        side_ports[(relationship.child_table, target_side)].append(
            (index, parent_center)
        )
        edge_work.append(
            {
                "relationship": relationship,
                "source_side": source_side,
                "target_side": target_side,
            }
        )

    boundaries: dict[tuple[int, str], tuple[float, float]] = {}
    for (table_name, side), port_items in side_ports.items():
        node = node_data[table_name]
        ordered = sorted(port_items, key=lambda item: (item[1], item[0]))
        count = len(ordered)
        for position, (edge_index, _) in enumerate(ordered, start=1):
            fraction = position / (count + 1)
            if side in {"left", "right"}:
                x = node["x"] if side == "left" else node["x"] + node["width"]
                y = node["y"] + 74 + fraction * (node["height"] - 106)
            else:
                x = node["x"] + 42 + fraction * (node["width"] - 84)
                y = node["y"] if side == "top" else node["y"] + node["height"]
            role = (
                "source"
                if edge_work[edge_index]["source_side"] == side
                and edge_work[edge_index]["relationship"].parent_table == table_name
                else "target"
            )
            boundaries[(edge_index, role)] = (x, y)

    relationship_ids = [
        item.id for item in evidence if item.kind == "candidate_relationship"
    ]
    route_buckets: dict[tuple, list[int]] = defaultdict(list)
    for index, work in enumerate(edge_work):
        relationship = work["relationship"]
        route_buckets[
            (
                layer_by_table[relationship.parent_table],
                layer_by_table[relationship.child_table],
                work["source_side"],
                work["target_side"],
            )
        ].append(index)

    route_offsets: dict[int, float] = {}
    for indexes in route_buckets.values():
        for position, edge_index in enumerate(indexes):
            route_offsets[edge_index] = (position - (len(indexes) - 1) / 2) * 12

    edges = []
    for index, work in enumerate(edge_work):
        relationship = work["relationship"]
        source_boundary = boundaries[(index, "source")]
        target_boundary = boundaries[(index, "target")]
        source_point, source_marks = _cardinality_marks(
            *source_boundary, work["source_side"], "one"
        )
        target_cardinality = (
            "many" if relationship.cardinality == "one_to_many" else "one"
        )
        target_point, target_marks = _cardinality_marks(
            *target_boundary, work["target_side"], target_cardinality
        )
        source_x, source_y = source_point
        target_x, target_y = target_point
        offset = route_offsets[index]
        source_side = work["source_side"]
        target_side = work["target_side"]
        if source_side in {"left", "right"} and target_side in {"left", "right"}:
            if source_side != target_side:
                lane = (source_x + target_x) / 2 + offset
            else:
                lane = (
                    min(source_x, target_x) - 48 - abs(offset)
                    if source_side == "left"
                    else max(source_x, target_x) + 48 + abs(offset)
                )
            path = (
                f"M {source_x:g} {source_y:g} H {lane:g} V {target_y:g} H {target_x:g}"
            )
        elif source_side in {"top", "bottom"} and target_side in {"top", "bottom"}:
            if source_side != target_side:
                lane = (source_y + target_y) / 2 + offset
            else:
                lane = (
                    min(source_y, target_y) - 48 - abs(offset)
                    if source_side == "top"
                    else max(source_y, target_y) + 48 + abs(offset)
                )
            path = (
                f"M {source_x:g} {source_y:g} V {lane:g} H {target_x:g} V {target_y:g}"
            )
        elif source_side in {"left", "right"}:
            path = f"M {source_x:g} {source_y:g} H {target_x:g} V {target_y:g}"
        else:
            path = f"M {source_x:g} {source_y:g} V {target_y:g} H {target_x:g}"

        label_offsets = {
            "left": (-32, 0),
            "right": (32, 0),
            "top": (0, -25),
            "bottom": (0, 25),
        }
        label_dx, label_dy = label_offsets[target_side]
        confidence = relationship.confidence
        confidence_bin = (
            "high" if confidence >= 0.85 else "medium" if confidence >= 0.7 else "low"
        )
        edges.append(
            {
                "parent_table": relationship.parent_table,
                "child_table": relationship.child_table,
                "parent_columns": relationship.parent_columns,
                "child_columns": relationship.child_columns,
                "cardinality": relationship.cardinality,
                "confidence": confidence,
                "confidence_bin": confidence_bin,
                "inclusion_rate": relationship.inclusion_rate,
                "name_similarity": relationship.name_similarity,
                "type_compatibility": relationship.type_compatibility,
                "path": path,
                "source_marks": source_marks,
                "target_marks": target_marks,
                "label_x": target_x + label_dx,
                "label_y": target_y + label_dy,
            }
        )
    return Artifact.create(
        kind="schema_graph",
        title="Candidate schema graph",
        data={
            "width": width,
            "height": height,
            "layout": "layered_er_v2",
            "nodes": [node_data[name] for name in table_names],
            "edges": edges,
        },
        evidence_ids=tuple(relationship_ids),
        metadata={"candidate_graph": True, "layout_version": 2},
    )


def discover_schema_dataset(
    tables: Mapping[str, pd.DataFrame],
    catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
    max_key_columns: int | None,
    min_key_uniqueness: float,
    min_key_completeness: float,
    min_relationship_inclusion: float,
    min_relationship_confidence: float,
    callbacks: tuple[EventCallback, ...] = (),
) -> AnalysisResult:
    """Discover candidate keys and cross-table relationships."""
    emit(
        callbacks,
        Event(
            EventKind.RUN_STARTED,
            "Schema discovery started.",
            stage="schema_discovery",
        ),
    )
    emit(
        callbacks,
        Event(
            EventKind.STAGE_STARTED,
            "Searching for minimal candidate keys.",
            stage="key_discovery",
        ),
    )
    discovery = discover_schema_candidates(
        tables,
        mode=config.mode,
        max_key_columns=max_key_columns,
        min_key_uniqueness=min_key_uniqueness,
        min_key_completeness=min_key_completeness,
        min_relationship_inclusion=min_relationship_inclusion,
        min_relationship_confidence=min_relationship_confidence,
        sampling=config.sampling,
        random_seed=config.random_seed,
    )
    key_evidence = tuple(_key_evidence(candidate) for candidate in discovery.keys)
    relationship_evidence = tuple(
        _relationship_evidence(candidate) for candidate in discovery.relationships
    )
    evidence = key_evidence + relationship_evidence
    for item in evidence:
        emit(
            callbacks,
            Event(
                EventKind.EVIDENCE_CREATED,
                item.description,
                stage="schema_discovery",
                data={"evidence_id": item.id, "kind": item.kind},
            ),
        )

    warnings = list(discovery.warnings)
    insufficient = catalog.table_count < 2 or catalog.row_count == 0
    if catalog.table_count < 2:
        warnings.append(
            AnalysisWarning(
                code="single_table_schema_discovery",
                message=(
                    "At least two tables are required to infer foreign-key "
                    "relationships; candidate keys were still evaluated."
                ),
            )
        )
    if insufficient and not config.allow_insufficient_evidence:
        status = AnalysisStatus.INSUFFICIENT_EVIDENCE
    elif not discovery.relationships:
        status = AnalysisStatus.NO_MEANINGFUL_STRUCTURE
    elif warnings:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
    else:
        status = AnalysisStatus.COMPLETED

    if status == AnalysisStatus.INSUFFICIENT_EVIDENCE:
        summary = (
            f"Found {len(discovery.keys)} candidate key(s), but there is insufficient "
            "multi-table evidence for relationship discovery."
        )
    elif not discovery.relationships:
        summary = (
            f"Found {len(discovery.keys)} candidate key(s), but no relationships met "
            "the configured evidence thresholds."
        )
    else:
        summary = (
            f"Found {len(discovery.keys)} candidate key(s) and "
            f"{len(discovery.relationships)} candidate relationship(s) across "
            f"{catalog.table_count} tables."
        )

    analysis = _graph_analysis(catalog, discovery.relationships)
    artifact = _graph_artifact(
        catalog, discovery.keys, discovery.relationships, evidence, analysis
    )
    result = AnalysisResult(
        goal="schema_discovery",
        status=status,
        summary=summary,
        catalog=catalog,
        findings=_findings(discovery.relationships, evidence),
        evidence=evidence,
        artifacts=(artifact,),
        assumptions=context.assumptions,
        warnings=tuple(warnings),
        sampling=discovery.sampling,
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
            "max_key_columns": (
                {"quick": 1, "standard": 2, "deep": 3}[AnalysisMode(config.mode).value]
                if max_key_columns is None
                else max_key_columns
            ),
            "candidate_keys": len(discovery.keys),
            "candidate_relationships": len(discovery.relationships),
            "verdict": _schema_verdict(catalog, discovery.relationships, analysis),
            "table_roles": {
                name: info["role"] for name, info in analysis.items()
            },
        },
    )
    emit(
        callbacks,
        Event(
            EventKind.STAGE_COMPLETED,
            "Candidate keys and relationships evaluated.",
            stage="schema_discovery",
            progress=1.0,
        ),
    )
    emit(
        callbacks,
        Event(
            EventKind.RUN_COMPLETED,
            result.summary,
            stage="schema_discovery",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
