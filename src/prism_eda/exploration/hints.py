"""Sections that need a little domain knowledge to be worth computing.

Exploration is name-blind by default. A caller who knows which column is the
label, the entity, the event time or the amount can say so, and gets analysis
that would be noise if guessed: what separates labelled rows from the rest, how
labelled entities sit in the link network, when activity happens, and how
amounts cluster on round figures.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from prism_eda.evidence.models import OBSERVATION, Evidence, EvidenceScope, Finding
from prism_eda.exploration.config import ColumnHint, ExploreConfig, Hints
from prism_eda.exploration.network import NetworkResult, _concentration, _distribution
from prism_eda.exploration.relationships import RelationshipResult, domain_of
from prism_eda.exploration.tables import TableState
from prism_eda.results import AnalysisFailure, AnalysisWarning
from prism_eda.sources.base import (
    Source,
    SourceError,
    matches,
    normalise_value,
    sampling_rate,
)

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MIN_LABELLED_ROWS = 30
MAX_SEPARATORS = 15


@dataclass(slots=True)
class HintResult:
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    warnings: list[AnalysisWarning] = field(default_factory=list)
    failures: list[AnalysisFailure] = field(default_factory=list)


def hint_matches(hint: ColumnHint | None, state: TableState) -> str | None:
    """The hinted column's name in this table, if the hint applies to it."""
    if hint is None:
        return None
    if hint.table and not matches(state.info, [hint.table]):
        if hint.table.lower() != state.display.lower():
            return None
    for column in state.columns:
        if column.name.lower() == hint.column.lower():
            return column.name
    return None


# --------------------------------------------------------------------------
# Sample-based: computed while the table's row sample is in memory
# --------------------------------------------------------------------------


def time_pattern_evidence(
    state: TableState, sample: pd.DataFrame, column: str
) -> Evidence | None:
    """Rows by weekday and hour, from the row sample."""
    if column not in sample.columns:
        return None
    stamps = pd.to_datetime(sample[column], errors="coerce").dropna()
    if len(stamps) < 200:
        return None
    hours = stamps.dt.hour
    # A column holding dates only has every row at midnight; an hourly
    # pattern for it would be a picture of nothing.
    if (hours == 0).mean() > 0.95:
        return None
    grid = np.zeros((7, 24), dtype="int64")
    np.add.at(grid, (stamps.dt.dayofweek.to_numpy(), hours.to_numpy()), 1)
    return Evidence.create(
        kind="explore_time_pattern",
        scope=EvidenceScope(table=state.display, columns=(column,)),
        value={
            "column": column,
            "weekdays": _WEEKDAYS,
            "grid": grid.tolist(),
            "rows": int(len(stamps)),
            "weekend_share": float((stamps.dt.dayofweek >= 5).mean()),
            "night_share": float(((hours < 6) | (hours >= 22)).mean()),
        },
        method="sample_weekday_hour_counts_v1",
        description=f"When {state.display}.{column} events happen.",
        metadata={"basis": "sample", "sample_rows": int(len(sample))},
    )


def amount_evidence(
    state: TableState, sample: pd.DataFrame, column: str
) -> tuple[Evidence, list[Finding]] | None:
    """Round-figure heaping and repeated exact amounts, from the row sample."""
    if column not in sample.columns:
        return None
    values = pd.to_numeric(sample[column], errors="coerce").dropna()
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 200:
        return None
    array = values.to_numpy(dtype="float64")
    whole = np.isclose(array, np.round(array))
    shares = {
        f"multiple_of_{base}": float(np.isclose(array % base, 0).mean())
        for base in (100, 1_000, 10_000)
    }
    counts = values.round(2).value_counts()
    top = [
        {
            "value": float(value),
            "count": int(count),
            "share": float(count / len(values)),
        }
        for value, count in counts.head(10).items()
    ]
    logs = np.log10(array)
    histogram, edges = np.histogram(logs, bins=30)
    value = {
        "column": column,
        "rows": int(len(values)),
        "whole_number_share": float(whole.mean()),
        **shares,
        "top_amounts": top,
        "log10_histogram": {
            "counts": [int(count) for count in histogram],
            "edges": [float(edge) for edge in edges],
        },
        "quantiles": {
            label: float(np.quantile(array, q))
            for label, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p999", 0.999))
        },
    }
    item = Evidence.create(
        kind="explore_amount",
        scope=EvidenceScope(table=state.display, columns=(column,)),
        value=value,
        method="sample_heaping_and_repeats_v1",
        description=f"How {state.display}.{column} values cluster.",
        metadata={"basis": "sample", "sample_rows": int(len(sample))},
    )
    findings: list[Finding] = []
    # One exact amount carrying a visible share of all rows, and sitting just
    # under a round figure, is the classic threshold-avoidance signature.
    for entry in top[:5]:
        amount = entry["value"]
        if entry["share"] < 0.01 or amount < 1_000:
            continue
        magnitude = 10 ** int(np.floor(np.log10(amount)))
        ceiling = np.ceil(amount / magnitude) * magnitude
        gap = (ceiling - amount) / ceiling
        if 0 < gap <= 0.005:
            findings.append(
                Finding.create(
                    title=(
                        "Repeated amount just below a round figure in "
                        f"{state.display}.{column}"
                    ),
                    summary=(
                        f"{entry['share']:.1%} of sampled rows carry exactly "
                        f"{amount:,.2f}, just under {ceiling:,.0f}."
                    ),
                    severity="info",
                    confidence=0.7,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Check whether a limit or reporting threshold sits at "
                        "the round figure; amounts held just beneath one are "
                        "worth separating out."
                    ),
                    category=OBSERVATION,
                )
            )
    return item, findings


# --------------------------------------------------------------------------
# Source-based: computed after relationships and links are known
# --------------------------------------------------------------------------


def _positive_values(
    hint: ColumnHint, state: TableState, column: str
) -> tuple[list[Any], int | None, str | None]:
    """Which label values count as positive, and how many rows carry them."""
    assert state.catalog is not None
    catalog = next(c for c in state.catalog.columns if c.name == column)
    top = list(catalog.top_values)
    if hint.positive:
        wanted = {normalise_value(value) for value in hint.positive}
        count = sum(
            entry["count"] for entry in top if normalise_value(entry["value"]) in wanted
        )
        return list(hint.positive), count or None, None
    if catalog.unique_count != 2 or len(top) != 2:
        return (
            [],
            None,
            (
                f"The label {state.display}.{column} has "
                f"{catalog.unique_count} distinct values; say which are positive "
                "with the hint's 'positive' list."
            ),
        )
    rare = min(top, key=lambda entry: entry["count"])
    return [rare["value"]], int(rare["count"]), None


def _auc(positive: np.ndarray, rest: np.ndarray) -> float:
    """Probability that a positive value exceeds a non-positive one."""
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([positive, rest]))
    total = ranks[: len(positive)].sum()
    return float(
        (total - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(rest))
    )


def _separators(
    positive: pd.DataFrame, rest: pd.DataFrame, skip: set[str]
) -> list[dict[str, Any]]:
    """Columns whose values differ most between labelled rows and the rest."""
    found: list[dict[str, Any]] = []
    for column in positive.columns:
        if column in skip:
            continue
        left, right = positive[column].dropna(), rest[column].dropna()
        if len(left) < MIN_LABELLED_ROWS or len(right) < 200:
            continue
        if pd.api.types.is_bool_dtype(left) or right.nunique() <= 30:
            shares_left = left.astype(str).value_counts(normalize=True)
            shares_right = right.astype(str).value_counts(normalize=True)
            categories = shares_left.index.union(shares_right.index)
            difference = shares_left.reindex(
                categories, fill_value=0.0
            ) - shares_right.reindex(categories, fill_value=0.0)
            strength = float(difference.abs().sum() / 2)
            # Sampling alone moves these shares; demand a visible difference.
            if strength < max(0.08, 3 * np.sqrt(len(categories) / len(left)) / 2):
                continue
            worst = difference.abs().idxmax()
            found.append(
                {
                    "column": column,
                    "type": "categorical",
                    "strength": strength,
                    "value": str(worst),
                    "labelled_share": float(shares_left.get(worst, 0.0)),
                    "other_share": float(shares_right.get(worst, 0.0)),
                }
            )
        elif pd.api.types.is_numeric_dtype(left):
            a = pd.to_numeric(left, errors="coerce").dropna().to_numpy(dtype="float64")
            b = pd.to_numeric(right, errors="coerce").dropna().to_numpy(dtype="float64")
            if len(a) < MIN_LABELLED_ROWS or len(b) < 200:
                continue
            auc = _auc(a, b)
            noise = np.sqrt(1 / (12 * len(a)) + 1 / (12 * len(b)))
            if abs(auc - 0.5) < max(0.05, 3 * noise):
                continue
            found.append(
                {
                    "column": column,
                    "type": "numeric",
                    "strength": abs(auc - 0.5) * 2,
                    "auc": auc,
                    "labelled_median": float(np.median(a)),
                    "other_median": float(np.median(b)),
                }
            )
    found.sort(key=lambda item: -item["strength"])
    return found[:MAX_SEPARATORS]


def label_analysis(
    states: Sequence[TableState],
    sources: dict[str, Source],
    hints: Hints,
    relationships: RelationshipResult,
    network: NetworkResult,
    config: ExploreConfig,
    tick: Callable[[str], None],
) -> HintResult:
    result = HintResult()
    if hints.label is None:
        return result
    target = next(
        (
            (state, name)
            for state in states
            if (name := hint_matches(hints.label, state))
        ),
        None,
    )
    if target is None or target[0].catalog is None or not target[0].rows:
        result.warnings.append(
            AnalysisWarning(
                code="label_not_found",
                message=(
                    f"The label column '{hints.label.column}' was not found in "
                    "any explored table."
                ),
            )
        )
        return result
    state, column = target
    source = sources[state.info.source]
    positives, positive_rows, problem = _positive_values(hints.label, state, column)
    if problem:
        result.warnings.append(
            AnalysisWarning("label_ambiguous", problem, state.display)
        )
        return result

    tick(f"label: {state.display}.{column}")
    readable = [item for item in state.columns if item.kind != "other"]
    hash_columns = [state.column(key.name) for key in state.key_columns[:2]]
    try:
        labelled = source.sample(
            state.info,
            readable,
            config.label_rows,
            total_rows=positive_rows or config.label_rows,
            seed=config.seed,
            hash_columns=hash_columns,
            where=(state.column(column), positives),
        )
        everyone = source.sample(
            state.info,
            readable,
            config.sample_rows,
            total_rows=state.rows,
            seed=config.seed + 1,
            hash_columns=hash_columns,
        )
    except SourceError as error:
        result.failures.append(
            AnalysisFailure("label_analysis", str(error), True, state.display)
        )
        return result
    wanted = {normalise_value(value) for value in positives}
    rest = everyone[~everyone[column].map(normalise_value).isin(wanted).to_numpy()]
    value: dict[str, Any] = {
        "table": state.display,
        "column": column,
        "positive_values": [str(item) for item in positives],
        "labelled_rows": positive_rows,
        "table_rows": state.rows,
        "labelled_rate": (positive_rows / state.rows) if positive_rows else None,
        "labelled_sampled": int(len(labelled)),
        "others_sampled": int(len(rest)),
    }
    if len(labelled) < MIN_LABELLED_ROWS:
        result.warnings.append(
            AnalysisWarning(
                code="label_too_few",
                message=(
                    f"Only {len(labelled)} labelled rows were found in "
                    f"{state.display}; that is too few to compare against the rest."
                ),
                table=state.display,
            )
        )
        return result
    value["separators"] = _separators(labelled, rest, {column})

    # -- where labelled entities sit in the link network -------------------
    entity = hint_matches(hints.entity, state)
    if entity is None:
        linked = [
            key.name
            for key in state.key_columns
            if domain_of(relationships, (state.key, key.name)) is not None
        ]
        unique = [name for name in linked if (k := state.key_column(name)) and k.unique]
        candidates = unique or linked
        entity = candidates[0] if candidates else None
    if entity and entity in labelled.columns:
        domain = domain_of(relationships, (state.key, entity))
        labelled_ids = sorted(
            {
                norm
                for raw in labelled[entity].tolist()
                if (norm := normalise_value(raw))
            }
        )
        complete = positive_rows is not None and len(labelled) >= positive_rows
        value["entity_column"] = entity
        value["labelled_entities"] = len(labelled_ids)
        by_key = {item.key: item for item in states}
        in_links: list[dict[str, Any]] = []
        for link in network.links:
            link_state = next(
                (item for item in by_key.values() if item.display == link["table"]),
                None,
            )
            if link_state is None or domain is None:
                continue
            if (link_state.key, link["origin"]) not in domain.members:
                continue
            tick(f"label links: {link_state.display}")
            link_source = sources[link_state.info.source]
            origin, destination = link["origin"], link["destination"]
            try:
                present = link_source.probe(
                    link_state.info,
                    {
                        origin: (link_state.column(origin), labelled_ids),
                        destination: (link_state.column(destination), labelled_ids),
                    },
                )
                probes = labelled_ids[:2_000]
                pairs = link_source.neighbours(
                    link_state.info,
                    link_state.column(origin),
                    link_state.column(destination),
                    probes,
                    config.hop_pairs,
                )
            except SourceError as error:
                result.failures.append(
                    AnalysisFailure("label_links", str(error), True, link_state.display)
                )
                continue
            partners: dict[str, set[str]] = defaultdict(set)
            for a, b in pairs:
                if a != b:
                    partners[a].add(b)
            degrees = [len(found) for found in partners.values()]
            labelled_set = set(labelled_ids)
            to_labelled = sum(
                1
                for found in partners.values()
                for node in found
                if node in labelled_set
            )
            total_partners = sum(degrees)
            baseline = (link.get("out_degree") or {}).get("median")
            in_links.append(
                {
                    "table": link_state.display,
                    "origin": origin,
                    "destination": destination,
                    "labelled_as_origin": len(present[origin]) / len(labelled_ids),
                    "labelled_as_destination": len(present[destination])
                    / len(labelled_ids),
                    "labelled_out_degree": _distribution(degrees) if degrees else None,
                    "baseline_out_degree_median": baseline,
                    "partners_also_labelled": (
                        to_labelled / total_partners if total_partners else None
                    ),
                    "labelled_set_complete": complete,
                }
            )
        value["links"] = in_links

    item = Evidence.create(
        kind="explore_label",
        scope=EvidenceScope(table=state.display, columns=(column,)),
        value=value,
        method="labelled_versus_rest_v1",
        description=f"What distinguishes rows where {state.display}.{column} is set.",
        confidence=0.8,
        assumptions=(
            "Labelled rows are compared with a sample of all other rows; a "
            "difference is an association, not a cause.",
            "Columns filled in *because* a row was labelled will look like "
            "perfect separators; they are leakage, not signal.",
        ),
    )
    result.evidence.append(item)

    rate = value["labelled_rate"]
    parts = [
        f"{positive_rows:,} of {state.rows:,} rows ({rate:.2%}) are labelled."
        if positive_rows and rate is not None
        else f"{len(labelled):,} labelled rows were sampled."
    ]
    top = value["separators"][:3]
    if top:
        described = []
        for entry in top:
            if entry["type"] == "numeric":
                described.append(
                    f"{entry['column']} (median {entry['labelled_median']:,.4g} "
                    f"vs {entry['other_median']:,.4g})"
                )
            else:
                described.append(
                    f"{entry['column']} ('{entry['value']}' in "
                    f"{entry['labelled_share']:.0%} vs {entry['other_share']:.0%})"
                )
        parts.append("The columns that differ most are " + "; ".join(described) + ".")
    else:
        parts.append("No column differs clearly between labelled rows and the rest.")
    for link in value.get("links", [])[:2]:
        degree = link.get("labelled_out_degree")
        text = (
            f"In {link['table']}, {link['labelled_as_origin']:.0%} of labelled "
            f"entities appear as {link['origin']} and "
            f"{link['labelled_as_destination']:.0%} as {link['destination']}"
        )
        if degree and link.get("baseline_out_degree_median") is not None:
            text += (
                f"; their median number of distinct {link['destination']} values "
                f"is {degree['median']:.0f} against "
                f"{link['baseline_out_degree_median']:.0f} overall"
            )
        if (
            link.get("partners_also_labelled") is not None
            and link["labelled_set_complete"]
        ):
            text += (
                f"; {link['partners_also_labelled']:.1%} of their counterparties "
                "are themselves labelled"
            )
        parts.append(text + ".")
    result.findings.append(
        Finding.create(
            title=f"What sets labelled rows apart in {state.display}",
            summary=" ".join(parts),
            severity="info",
            confidence=0.75,
            evidence_ids=(item.id,),
            recommendation=(
                "Treat a near-perfect separator as suspected leakage until its "
                "timing is understood."
            ),
            category=OBSERVATION,
        )
    )
    return result


def entity_activity(
    states: Sequence[TableState],
    sources: dict[str, Source],
    hints: Hints,
    config: ExploreConfig,
    tick: Callable[[str], None],
) -> HintResult:
    """Rows per entity in every table that carries the hinted entity column."""
    result = HintResult()
    if hints.entity is None:
        return result
    for state in states:
        column = hint_matches(hints.entity, state)
        if column is None or not state.rows:
            continue
        key = state.key_column(column)
        if key is None or key.unique:
            continue
        tick(f"activity per {column}: {state.display}")
        try:
            frame = sources[state.info.source].fanout_sample(
                state.info,
                state.column(column),
                [],
                sampling_rate(key.distinct, config.degree_entities),
                config.degree_entities * 2,
            )
        except SourceError as error:
            result.failures.append(
                AnalysisFailure("entity_activity", str(error), True, state.display)
            )
            continue
        if len(frame) < 30:
            continue
        rows = frame["rows"].to_numpy()
        result.evidence.append(
            Evidence.create(
                kind="explore_entity_activity",
                scope=EvidenceScope(table=state.display, columns=(column,)),
                value={
                    "table": state.display,
                    "column": column,
                    "entities_sampled": int(len(frame)),
                    "rows_per_entity": _distribution(rows),
                    **_concentration(rows),
                },
                method="entity_sampled_row_counts_v1",
                description=f"Rows per {column} in {state.display}.",
                confidence=0.85,
            )
        )
    return result
