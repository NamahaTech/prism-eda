"""Profile links between records of the same kind.

When two columns of one table hold the same identifiers — sender and receiver,
employee and manager, referrer and referred — each row is a link, and the table
is a network. Whether that network is usable depends on facts no column profile
shows: how many of the counterparties we actually hold data for, how many
links an entity has, and whether following links leads anywhere.

Everything here is measured on *sampled entities with their complete links*
(never on sampled rows, which would thin every entity's links and understate
every figure), and every walk has a hard size cap that is reported when hit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from prism_eda.evidence.models import OBSERVATION, Evidence, EvidenceScope, Finding
from prism_eda.exploration.config import ExploreConfig
from prism_eda.exploration.relationships import (
    ColumnId,
    RelationshipResult,
    domain_of,
)
from prism_eda.exploration.tables import TableState
from prism_eda.results import AnalysisFailure
from prism_eda.sources.base import (
    Source,
    SourceError,
    normalise_value,
    sampling_rate,
    value_hash,
)

#: Above this many rows the whole link list is not pulled; components are then
#: estimated by walking outward from sampled entities.
EXACT_COMPONENT_ROWS = 1_000_000
COMPONENT_SEEDS = 60
COMPONENT_DEPTH = 6
#: Fan-out is measured for the widest identifier columns of a table only.
FANOUT_COLUMNS = 6
FANOUT_TABLES = 15


@dataclass(slots=True)
class NetworkResult:
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    failures: list[AnalysisFailure] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)


def _distribution(values: Sequence[int] | np.ndarray) -> dict[str, Any]:
    """Summary of a count distribution, with log-spaced histogram bins."""
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        return {"count": 0}
    edges = [1, 2, 3, 4, 6, 11, 21, 51, 101, 501, 1_001, 10_001]
    labels = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "51-100",
              "101-500", "501-1k", "1k-10k", "10k+"]  # fmt: skip
    counts = [0] * len(labels)
    positions = np.searchsorted(edges, array, side="right") - 1
    for position in positions:
        if position >= 0:
            counts[int(min(position, len(labels) - 1))] += 1
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "share_one": float((array == 1).mean()),
        "histogram": [
            {"label": label, "count": count}
            for label, count in zip(labels, counts, strict=True)
        ],
    }


def _concentration(rows: np.ndarray) -> dict[str, Any]:
    """How unevenly rows are spread over entities."""
    if rows.size == 0 or rows.sum() == 0:
        return {}
    ordered = np.sort(rows)[::-1]
    total = float(ordered.sum())
    top1 = max(1, int(round(0.01 * len(ordered))))
    top10 = max(1, int(round(0.10 * len(ordered))))
    return {
        "top_1pct_share": float(ordered[:top1].sum() / total),
        "top_10pct_share": float(ordered[:top10].sum() / total),
    }


def find_links(
    states: Sequence[TableState], relationships: RelationshipResult
) -> list[tuple[TableState, str, str]]:
    """Column pairs inside one table that hold the same kind of identifier."""
    by_key = {state.key: state for state in states}
    links: list[tuple[TableState, str, str]] = []
    seen: set[tuple[str, frozenset[str]]] = set()
    for item in relationships.relationships:
        if not item.same_table or item.tier == "possible":
            continue
        marker = (item.child[0], frozenset((item.child[1], item.parent[1])))
        if marker in seen:
            continue
        seen.add(marker)
        state = by_key[item.child[0]]
        # Keep the table's own column order: by convention the origin of a
        # link is written before its destination (from/to, sender/receiver).
        names = [column.name for column in state.columns]
        first, second = sorted(
            (item.child[1], item.parent[1]), key=lambda name: names.index(name)
        )
        links.append((state, first, second))
    links.sort(key=lambda link: -link[0].rows)
    return links


# --------------------------------------------------------------------------
# One link
# --------------------------------------------------------------------------


def _coverage_by_reference(
    relationships: RelationshipResult, column_id: ColumnId
) -> dict[str, Any] | None:
    domain = domain_of(relationships, column_id)
    if domain is None or domain.reference is None or domain.reference == column_id:
        return None
    for item in relationships.relationships:
        if item.child == column_id and item.parent == domain.reference:
            return {
                "coverage": item.forward.coverage,
                "row_coverage": item.forward.row_coverage,
                "values_checked": item.forward.asked,
            }
    return None


def _walk(
    source: Source,
    state: TableState,
    origin: str,
    destination: str,
    seeds: list[str],
    config: ExploreConfig,
) -> dict[str, Any]:
    """Follow links outward from seed entities, hop by hop."""
    left, right = state.column(origin), state.column(destination)
    reach: dict[str, set[str]] = {seed: set() for seed in seeds}
    adjacency: dict[str, set[str]] = defaultdict(set)
    expanded: set[str] = set()
    frontier = list(seeds)
    truncated: list[int] = []
    per_hop: list[dict[str, Any]] = []
    current: dict[str, set[str]] = {seed: {seed} for seed in seeds}
    reciprocal = 0
    first_hop_pairs = 0
    continuing = 0
    first_hop_nodes: set[str] = set()
    first_reach: dict[str, set[str]] = {}

    for hop in range(1, config.max_hops + 1):
        frontier = [node for node in frontier if node not in expanded]
        if len(frontier) > config.hop_frontier:
            frontier = sorted(frontier, key=value_hash)[: config.hop_frontier]
            truncated.append(hop)
        if not frontier:
            break
        pairs = source.neighbours(state.info, left, right, frontier, config.hop_pairs)
        if len(pairs) >= config.hop_pairs and hop not in truncated:
            truncated.append(hop)
        expanded.update(frontier)
        for a, b in pairs:
            if a != b:
                adjacency[a].add(b)

        following: dict[str, set[str]] = {}
        next_frontier: set[str] = set()
        for seed in seeds:
            reached: set[str] = set()
            for node in current[seed]:
                reached.update(adjacency.get(node, ()))
            reached.discard(seed)
            new = reached - reach[seed]
            reach[seed].update(new)
            following[seed] = new
            next_frontier.update(new)
        current = following
        sizes = np.array([len(reach[seed]) for seed in seeds])
        per_hop.append(
            {
                "hop": hop,
                "seeds_reaching": float(np.mean([bool(following[s]) for s in seeds])),
                "median_reach": float(np.median(sizes)),
                "p90_reach": float(np.quantile(sizes, 0.9)),
                "max_reach": int(sizes.max()),
                "entities_followed": len(frontier),
            }
        )
        if hop == 1:
            first_hop_nodes = set(next_frontier)
            first_reach = {seed: set(following[seed]) for seed in seeds}
            first_hop_pairs = sum(len(following[seed]) for seed in seeds)
        if hop == 2:
            # Hop 2 expanded every hop-1 entity that was followed, so it shows
            # which of them have onward links, and which link straight back.
            followed = first_hop_nodes & expanded
            continuing = sum(1 for node in followed if adjacency.get(node))
            for seed in seeds:
                for node in first_reach[seed] & followed:
                    if seed in adjacency.get(node, ()):
                        reciprocal += 1
            per_hop[-1]["first_hop_followed"] = len(followed)
        frontier = list(next_frontier)

    followed_count = per_hop[1].get("first_hop_followed", 0) if len(per_hop) > 1 else 0
    return {
        "seeds": len(seeds),
        "hops": per_hop,
        "truncated_hops": sorted(set(truncated)),
        "continuation_rate": continuing / followed_count if followed_count else None,
        "reciprocity": reciprocal / first_hop_pairs if first_hop_pairs else None,
        "first_hop_pairs": first_hop_pairs,
    }


def _exact_components(
    source: Source, state: TableState, origin: str, destination: str, seed: int
) -> dict[str, Any] | None:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    left, right = state.column(origin), state.column(destination)
    frame = source.sample(
        state.info, [left, right], state.rows + 1, total_rows=state.rows, seed=seed
    )
    pairs = pd.DataFrame(
        {
            "a": frame[origin].map(normalise_value),
            "b": frame[destination].map(normalise_value),
        }
    ).dropna()
    if pairs.empty:
        return None
    codes, uniques = pd.factorize(pd.concat([pairs["a"], pairs["b"]]), sort=False)
    half = len(pairs)
    graph = coo_matrix(
        (np.ones(half, dtype="int8"), (codes[:half], codes[half:])),
        shape=(len(uniques), len(uniques)),
    )
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return {
        "method": "exact",
        "entities": int(len(uniques)),
        "components": int(count),
        "largest": int(sizes.max()),
        "largest_share": float(sizes.max() / len(uniques)),
        "isolated_pairs": int((sizes == 2).sum()),
        "sizes": _component_buckets(sizes, weights=sizes),
    }


def _component_buckets(sizes: np.ndarray, weights: np.ndarray) -> list[dict[str, Any]]:
    """Share of entities by the size of the component they sit in."""
    edges = [2, 3, 6, 11, 51, 1_001]
    labels = ["2", "3-5", "6-10", "11-50", "51-1k", "1k+"]
    total = float(weights.sum())
    buckets = []
    positions = np.clip(
        np.searchsorted(edges, sizes, side="right") - 1, 0, len(labels) - 1
    )
    for index, label in enumerate(labels):
        share = float(weights[positions == index].sum() / total) if total else 0.0
        buckets.append({"label": label, "entity_share": share})
    return buckets


def _walked_components(
    source: Source,
    state: TableState,
    origin: str,
    destination: str,
    seeds: list[str],
    config: ExploreConfig,
) -> dict[str, Any] | None:
    """Estimate component sizes by walking in both directions from seeds.

    Each seed's group grows until it runs out of links (its true component) or
    the walk hits its size cap (a lower bound: "at least this large").
    """
    if not seeds:
        return None
    left, right = state.column(origin), state.column(destination)
    parent: dict[str, str] = {seed: seed for seed in seeds}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    frontier = list(seeds)
    expanded: set[str] = set()
    cap = config.hop_frontier * 2
    capped = False
    for _ in range(COMPONENT_DEPTH):
        frontier = [node for node in frontier if node not in expanded]
        if not frontier:
            break
        if len(expanded) + len(frontier) > cap:
            capped = True
            break
        pairs = source.neighbours(state.info, left, right, frontier, config.hop_pairs)
        pairs += [
            (b, a)
            for a, b in source.neighbours(
                state.info, right, left, frontier, config.hop_pairs
            )
        ]
        if len(pairs) >= config.hop_pairs:
            capped = True
        expanded.update(frontier)
        discovered: set[str] = set()
        for a, b in pairs:
            parent[find(a)] = find(b)
            discovered.update((a, b))
        frontier = list(discovered - expanded)
    open_groups = {find(node) for node in frontier if node not in expanded}
    group_sizes: dict[str, int] = defaultdict(int)
    for node in list(parent):
        group_sizes[find(node)] += 1
    closed = [
        group_sizes[find(seed)] for seed in seeds if find(seed) not in open_groups
    ]
    open_seeds = [
        group_sizes[find(seed)] for seed in seeds if find(seed) in open_groups
    ]
    sizes = np.array(closed, dtype="int64")
    return {
        "method": "walk_from_sampled_entities",
        "seeds": len(seeds),
        "closed_share": len(closed) / len(seeds),
        "open_share": len(open_seeds) / len(seeds),
        "open_at_least": int(max(open_seeds)) if open_seeds else None,
        "walk_capped": capped,
        "sizes": _component_buckets(sizes, weights=np.ones_like(sizes))
        if len(sizes)
        else [],
        "closed_median": float(np.median(sizes)) if len(sizes) else None,
    }


def profile_link(
    source: Source,
    state: TableState,
    origin: str,
    destination: str,
    relationships: RelationshipResult,
    config: ExploreConfig,
    result: NetworkResult,
) -> None:
    left, right = state.column(origin), state.column(destination)
    left_key, right_key = state.key_column(origin), state.key_column(destination)
    if left_key is None or right_key is None:
        return
    label = f"{state.display}: {origin} → {destination}"

    def attempt(stage: str, work: Callable[[], Any]) -> Any:
        try:
            return work()
        except SourceError as error:
            result.failures.append(
                AnalysisFailure(
                    stage=stage,
                    message=str(error),
                    recoverable=True,
                    table=state.display,
                )
            )
        except Exception as error:  # noqa: BLE001 - one measure never costs the link
            result.failures.append(
                AnalysisFailure(
                    stage=stage,
                    message=f"{type(error).__name__}: {error}",
                    recoverable=True,
                    table=state.display,
                )
            )
        return None

    value: dict[str, Any] = {
        "table": state.display,
        "origin": origin,
        "destination": destination,
        "rows": state.rows,
        "distinct_origins": left_key.distinct,
        "distinct_destinations": right_key.distinct,
    }

    pair = attempt("link_pairs", lambda: source.pair_profile(state.info, left, right))
    if pair:
        value["linked_rows"] = pair["rows"]
        value["self_links"] = pair["equal_rows"]
        value["distinct_pairs"] = pair["distinct_pairs"]
        value["distinct_pairs_exact"] = pair["distinct_pairs_exact"]
        if pair["distinct_pairs"] and pair["rows"]:
            value["rows_per_pair"] = pair["rows"] / pair["distinct_pairs"]

    # Role overlap: how many destinations also act as origins, and the reverse.
    for pair_item in relationships.relationships:
        if pair_item.same_table and {pair_item.child, pair_item.parent} == {
            (state.key, origin),
            (state.key, destination),
        }:
            forward_is_origin = pair_item.child == (state.key, origin)
            origin_in_destination = (
                pair_item.forward if forward_is_origin else pair_item.backward
            )
            destination_in_origin = (
                pair_item.backward if forward_is_origin else pair_item.forward
            )
            if destination_in_origin.asked:
                value["destinations_also_origins"] = destination_in_origin.coverage
                value["destination_rows_to_origins"] = (
                    destination_in_origin.row_coverage
                )
            if origin_in_destination.asked:
                value["origins_also_destinations"] = origin_in_destination.coverage
    value["origin_known"] = _coverage_by_reference(relationships, (state.key, origin))
    value["destination_known"] = _coverage_by_reference(
        relationships, (state.key, destination)
    )
    domain = domain_of(relationships, (state.key, origin))
    if domain is not None:
        value["domain"] = domain.id
        value["reference"] = domain.label if domain.reference else None

    out_frame = attempt(
        "link_degrees",
        lambda: source.fanout_sample(
            state.info,
            left,
            [right],
            sampling_rate(left_key.distinct, config.degree_entities),
            config.degree_entities * 2,
        ),
    )
    in_frame = attempt(
        "link_degrees",
        lambda: source.fanout_sample(
            state.info,
            right,
            [left],
            sampling_rate(right_key.distinct, config.degree_entities),
            config.degree_entities * 2,
        ),
    )
    if out_frame is not None and len(out_frame):
        value["out_degree"] = _distribution(out_frame[destination].to_numpy())
        value["origin_activity"] = _concentration(out_frame["rows"].to_numpy())
    if in_frame is not None and len(in_frame):
        value["in_degree"] = _distribution(in_frame[origin].to_numpy())

    seeds: list[str] = []
    if out_frame is not None and len(out_frame):
        seeds = sorted(out_frame["key"].astype(str), key=value_hash)[: config.hop_seeds]
    if seeds:
        walk = attempt(
            "link_reach",
            lambda: _walk(source, state, origin, destination, seeds, config),
        )
        if walk:
            value["reach"] = walk
        if state.rows <= EXACT_COMPONENT_ROWS:
            components = attempt(
                "link_components",
                lambda: _exact_components(
                    source, state, origin, destination, config.seed
                ),
            )
        else:
            components = attempt(
                "link_components",
                lambda: _walked_components(
                    source, state, origin, destination, seeds[:COMPONENT_SEEDS], config
                ),
            )
        if components:
            value["components"] = components

    item = Evidence.create(
        kind="explore_link",
        scope=EvidenceScope(table=state.display, columns=(origin, destination)),
        value=value,
        method="entity_sampled_link_profile_v1",
        description=f"Network formed by {label}.",
        confidence=0.85,
        assumptions=(
            "Degrees and reach are measured on hash-sampled entities with all "
            "of their links, not on sampled rows.",
            "Reach follows links in the origin → destination direction only.",
            "A walk that hit its size cap reports a lower bound.",
        ),
    )
    result.evidence.append(item)
    result.links.append(value)
    _link_finding(result, item, value, label)


def _link_finding(
    result: NetworkResult, item: Evidence, value: dict[str, Any], label: str
) -> None:
    parts = [
        f"{value['table']} links {value['origin']} to {value['destination']}: "
        "both hold the same kind of identifier, so each row connects two records."
    ]
    known = value.get("destination_known")
    if known:
        parts.append(
            f"{known['coverage']:.1%} of distinct {value['destination']} values "
            f"exist in {value.get('reference')}"
            + (
                f" ({known['row_coverage']:.1%} of rows)"
                if known.get("row_coverage") is not None
                else ""
            )
            + "; the rest are counterparties this data holds no record for."
        )
    also = value.get("destinations_also_origins")
    if also is not None:
        parts.append(
            f"{also:.1%} of {value['destination']} values also appear as "
            f"{value['origin']}, which is what makes multi-step paths possible."
        )
    reach = value.get("reach") or {}
    hops = {hop["hop"]: hop for hop in reach.get("hops", [])}
    if 2 in hops:
        text = (
            f"Starting from a sampled {value['origin']}, "
            f"{hops[2]['seeds_reaching']:.0%} reach a 2-step connection"
        )
        if 3 in hops:
            text += f" and {hops[3]['seeds_reaching']:.0%} a 3-step one"
        text += (
            f" (median reach {hops[1]['median_reach']:,.0f} → "
            f"{hops[2]['median_reach']:,.0f}"
            + (f" → {hops[3]['median_reach']:,.0f}" if 3 in hops else "")
            + " entities)."
        )
        if reach.get("truncated_hops"):
            text += " The walk hit its size cap, so later hops are lower bounds."
        parts.append(text)
    elif 1 in hops:
        parts.append(
            "No sampled entity reaches a second step: destinations do not "
            "appear as origins, so there are no multi-step paths to follow."
        )
    result.findings.append(
        Finding.create(
            title=f"Record-to-record links in {label}",
            summary=" ".join(parts),
            severity="info",
            confidence=0.85,
            evidence_ids=(item.id,),
            recommendation=(
                "Path- and neighbourhood-based analysis is only as deep as these "
                "figures: check the covered share before relying on 2- or 3-step "
                "patterns."
            ),
            category=OBSERVATION,
        )
    )


# --------------------------------------------------------------------------
# Fan-out between different identifiers
# --------------------------------------------------------------------------


def profile_fanout(
    source: Source, state: TableState, config: ExploreConfig, result: NetworkResult
) -> None:
    """For each pair of identifier columns: how many of one go with the other.

    This is what shows that a device is shared by forty accounts, or that a
    column everyone assumed was one-per-customer is not.
    """
    keys = [column for column in state.key_columns if not column.unique][
        :FANOUT_COLUMNS
    ]
    others_all = [column for column in state.key_columns if not column.unique][
        : FANOUT_COLUMNS + 2
    ]
    if not keys or len(others_all) < 2:
        return
    rows: list[dict[str, Any]] = []
    for key in keys:
        others = [column for column in others_all if column.name != key.name]
        if not others:
            continue
        try:
            frame = source.fanout_sample(
                state.info,
                state.column(key.name),
                [state.column(other.name) for other in others],
                sampling_rate(key.distinct, config.degree_entities),
                config.degree_entities * 2,
            )
        except SourceError as error:
            result.failures.append(
                AnalysisFailure(
                    stage="fanout",
                    message=str(error),
                    recoverable=True,
                    table=state.display,
                )
            )
            continue
        if frame.empty:
            continue
        for other in others:
            counts = frame[other.name].to_numpy()
            counts = counts[counts > 0]
            if counts.size < 30:
                continue
            rows.append(
                {
                    "key": key.name,
                    "other": other.name,
                    "entities": int(counts.size),
                    "share_multiple": float((counts > 1).mean()),
                    "median": float(np.median(counts)),
                    "p99": float(np.quantile(counts, 0.99)),
                    "max": int(counts.max()),
                }
            )
    if not rows:
        return
    item = Evidence.create(
        kind="explore_fanout",
        scope=EvidenceScope(
            table=state.display, columns=tuple(key.name for key in keys)
        ),
        value={"table": state.display, "pairs": rows},
        method="entity_sampled_distinct_counts_v1",
        description=(
            f"How many distinct values of one identifier accompany another in "
            f"{state.display}."
        ),
        confidence=0.85,
        assumptions=(
            "Measured on hash-sampled values of the first column with all of "
            "their rows.",
        ),
    )
    result.evidence.append(item)
    for row in rows:
        # A relation that is one-to-one for nearly everyone, with a few values
        # shared very widely, is the pattern worth a reader's attention.
        if row["p99"] <= 3 and row["max"] >= 10 and row["share_multiple"] <= 0.3:
            result.findings.append(
                Finding.create(
                    title=(
                        f"A few {row['key']} values are shared widely in "
                        f"{state.display}"
                    ),
                    summary=(
                        f"Most {row['key']} values appear with a single "
                        f"{row['other']} (median {row['median']:.0f}, 99th "
                        f"percentile {row['p99']:.0f}), but at least one appears "
                        f"with {row['max']:,} different {row['other']} values "
                        f"(among {row['entities']:,} sampled)."
                    ),
                    severity="info",
                    confidence=0.8,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Look at the widely shared values: they are either "
                        "placeholders or genuine hubs, and both matter."
                    ),
                    category=OBSERVATION,
                )
            )


def profile_network(
    states: Sequence[TableState],
    sources: dict[str, Source],
    relationships: RelationshipResult,
    config: ExploreConfig,
    tick: Callable[[str], None],
) -> NetworkResult:
    result = NetworkResult()
    links = find_links(states, relationships)
    for state, origin, destination in links[: config.max_links]:
        tick(f"links: {state.display} {origin} → {destination}")
        profile_link(
            sources[state.info.source],
            state,
            origin,
            destination,
            relationships,
            config,
            result,
        )
    fan_tables = sorted(
        (state for state in states if len(state.key_columns) >= 2 and state.rows),
        key=lambda state: -state.rows,
    )[:FANOUT_TABLES]
    for state in fan_tables:
        tick(f"fan-out: {state.display}")
        profile_fanout(sources[state.info.source], state, config, result)
    return result
