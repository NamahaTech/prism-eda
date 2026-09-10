"""What the plan has to say, beyond running fast.

A planner that only made things faster would be a performance tool. The reason
this lives in prism is that building the plan means holding, at once, the exact
definition of every feature *and* a measurement of what each one cost and
produced -- which is enough to answer questions a feature author cannot easily
ask of their own code: which of these are the same feature twice, which one is
eating the pipeline, which quietly degrades, and which is reading something it
should not be able to see.

Every detector here is built against the rule that a detector which never stays
quiet is not a detector. Each was calibrated on a clean, deliberately varied
feature set and produces nothing on it; the thresholds are recorded next to the
measurement that set them, not chosen for looking reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from prism_eda.evidence.models import (
    OBSERVATION,
    QUALITY_ISSUE,
    Evidence,
    EvidenceScope,
    Finding,
    sort_findings,
)
from prism_eda.features.executors.pandas_exec import ExecutionStats
from prism_eda.features.ir import iter_nodes
from prism_eda.results import AnalysisWarning

if TYPE_CHECKING:  # pragma: no cover
    from prism_eda.features.plan import FeaturePlan

__all__ = ["Thresholds", "cost_table", "diagnose_plan"]


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where each detector fires, and the clean-data reading behind it."""

    # Clean baseline: the highest |r| between any two of fifteen deliberately
    # varied features was 0.92. At 0.9999 a pair is the same column twice.
    redundant_correlation: float = 0.9999
    # Clean baseline: zero features fell back. Any sustained rate is real.
    fallback_share: float = 0.20
    # Clean baseline: the costliest of fifteen features held 12.2% of runtime.
    # Fires only several times above an even split, so it cannot fire at all
    # on a small feature set where a large share is simply arithmetic.
    # Set well clear of the clean reading so a run-to-run wobble in the
    # measurement cannot flip the finding on and off between builds.
    cost_share_floor: float = 0.40
    cost_share_multiple: float = 5.0
    # The multiple rule is unreachable below five features (5/n > 1), which
    # would let one feature hold nearly all the work of a small plan and go
    # unmentioned. Three quarters in a single feature is worth saying at any
    # size, so it qualifies on its own.
    cost_share_dominant: float = 0.75


# A static cost model, in units of "roughly how much work is this operation".
# The findings below are built on this rather than on the clock: a measured
# duration depends on what else the machine was doing, and a detector built on
# it fires at random. That is not hypothetical -- the timing-based version of
# the cost finding fired on clean data the first time the suite ran under load.
# The weights are ordered by the measurements in the module docstring: a
# two-level groupby (mode_count, entropy) really is several times a plain
# reduction, and an opaque feature really is two orders worse than anything
# the planner can see inside.
_OP_WEIGHTS: dict[str, int] = {
    "opaque": 100,
    "run_length_max": 5,
    "agg": 3,
    "window": 3,
    "lag": 3,
    "lead": 3,
    "diff": 3,
    "upper": 2,
    "lower": 2,
    "strip": 2,
    "replace": 2,
    "isin": 2,
}
_HEAVY_AGGREGATES = frozenset({"mode_count", "entropy"})
_HEAVY_AGGREGATE_WEIGHT = 8


def _node_weight(target: Any) -> int:
    """Static cost of one operation, independent of the machine."""
    if target.op == "agg" and target.param("func") in _HEAVY_AGGREGATES:
        return _HEAVY_AGGREGATE_WEIGHT
    return _OP_WEIGHTS.get(target.op, 1)


def _weight_by_feature(plan: FeaturePlan) -> tuple[dict[str, int], int]:
    """Exclusive static cost per feature, and the total across the plan."""
    reach = _reach(plan)
    by_id = {item.id: item for item in plan.nodes}
    exclusive = {item.name: 0 for item in plan.features}
    total = 0
    for item in plan.features:
        if item.kind == "opaque":
            exclusive[item.name] += _OP_WEIGHTS["opaque"]
            total += _OP_WEIGHTS["opaque"]
    for node_id, owners in reach.items():
        target = by_id.get(node_id)
        if target is None:
            continue
        weight = _node_weight(target)
        total += weight
        if len(owners) == 1:
            exclusive[next(iter(owners))] += weight
    return exclusive, total


def _pairs(names: list[str]) -> list[tuple[str, str]]:
    """Every unordered pair within a group of names."""
    return [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]


def _scope(columns: tuple[str, ...] = ()) -> EvidenceScope:
    return EvidenceScope(table=None, columns=columns)


def _reach(plan: FeaturePlan) -> dict[str, set[str]]:
    """Which features reach each node."""
    reach: dict[str, set[str]] = {}
    for item in plan.features:
        for reached in iter_nodes(item.node):
            reach.setdefault(reached.id, set()).add(item.name)
    return reach


def _cost_by_feature(
    plan: FeaturePlan, stats: ExecutionStats
) -> tuple[dict[str, float], float]:
    """Exclusive cost per feature, and the cost of work more than one shares."""
    reach = _reach(plan)
    exclusive = {item.name: 0.0 for item in plan.features}
    shared = 0.0
    for node_id, seconds in stats.node_seconds.items():
        owners = reach.get(node_id, set())
        if len(owners) == 1:
            exclusive[next(iter(owners))] += seconds
        else:
            shared += seconds
    for name, seconds in stats.opaque_seconds.items():
        if name in exclusive:
            exclusive[name] += seconds
    return exclusive, shared


def _sharing_evidence(plan: FeaturePlan, stats: ExecutionStats) -> Evidence:
    """What the planner shares, stated structurally.

    Deliberately free of wall-clock time. An evidence id is a hash of the
    value, so banking a measured duration would give the same analysis a
    different lineage on every run and on every machine. Time is reported --
    it is just reported as a measurement of one execution rather than as
    evidence about the data.
    """
    shared = plan.shared_nodes()
    return Evidence.create(
        kind="feature_plan_sharing",
        scope=_scope(tuple(item.name for item in plan.features)),
        value={
            "features": len(plan.features),
            "distinct_nodes": len(plan.nodes),
            "shared_nodes": len(shared),
            "shared_ops": sorted({item.op for item in shared}),
            "entities": stats.entity_count,
            "rows": stats.row_count,
        },
        method="node_id_reachability",
        description=(
            f"{len(shared)} of {len(plan.nodes)} operations are reached by more "
            f"than one feature, so the plan computes each of them once."
        ),
    )


def cost_table(plan: FeaturePlan, stats: ExecutionStats) -> list[dict[str, Any]]:
    """Per-feature measured cost, for the report rather than for evidence.

    Cost is exclusive: work a feature shares with others is not charged to it,
    so the column sums to the whole rather than counting a shared subtree once
    per feature that reached it.
    """
    exclusive, _ = _cost_by_feature(plan, stats)
    measured = sum(exclusive.values()) or 1.0
    entities = stats.entity_count or 1
    weights, total_weight = _weight_by_feature(plan)
    return [
        {
            "feature": name,
            "milliseconds": round(seconds * 1000, 3),
            "share": round(seconds / measured, 4),
            # Reported beside the measurement because the cost *finding* is
            # computed from this, not from the clock. On a small frame the two
            # disagree sharply -- an opaque feature is cheap over a hundred
            # entities and ruinous over three million -- and showing only one
            # of them makes the finding look like it contradicts the table.
            "estimated_share": round(weights.get(name, 0) / (total_weight or 1), 4),
            "fallback_share": round(stats.fallback_counts.get(name, 0) / entities, 4),
        }
        for name, seconds in sorted(exclusive.items(), key=lambda item: -item[1])
    ]


def _duplicate_findings(plan: FeaturePlan) -> tuple[list[Evidence], list[Finding]]:
    """Two names, one computation. Structural, so there are no false positives."""
    by_root: dict[str, list[str]] = {}
    for item in plan.features:
        by_root.setdefault(item.node.id, []).append(item.name)
    groups = [names for names in by_root.values() if len(names) > 1]
    if not groups:
        return [], []

    evidence = Evidence.create(
        kind="feature_duplicate_definitions",
        scope=_scope(tuple(name for group in groups for name in group)),
        value={"groups": [sorted(group) for group in groups]},
        method="structural_node_identity",
        description=(
            f"{len(groups)} group(s) of features compile to an identical expression."
        ),
    )
    findings = [
        Finding.create(
            title=f"{' and '.join(sorted(group))} are the same feature",
            summary=(
                f"{', '.join(sorted(group))} trace to an identical expression, so "
                "they will hold identical values in every row of every dataset. "
                "This is a structural fact about the definitions, not a "
                "correlation measured on this data."
            ),
            severity="high",
            confidence=1.0,
            evidence_ids=(evidence.id,),
            recommendation=(
                "Keep one and drop the rest. A model given the same column twice "
                "splits its importance between the copies, which makes both look "
                "weaker than the feature is."
            ),
            category=QUALITY_ISSUE,
        )
        for group in groups
    ]
    return [evidence], findings


def _redundancy_findings(
    produced: pd.DataFrame,
    thresholds: Thresholds,
    *,
    already_reported: frozenset[frozenset[str]] = frozenset(),
) -> tuple[list[Evidence], list[Finding]]:
    """Features that are not defined identically but behave identically.

    Pairs already reported as structurally identical are skipped: that finding
    is strictly stronger -- it holds on every dataset, not just this one -- and
    filing both would report one fact twice.
    """
    numeric = produced.select_dtypes("number")
    varying = [name for name in numeric.columns if numeric[name].nunique() > 1]
    if len(varying) < 2:
        return [], []
    matrix = numeric[varying].corr().abs().to_numpy(copy=True)
    np.fill_diagonal(matrix, 0.0)

    pairs = []
    for i in range(len(varying)):
        for j in range(i + 1, len(varying)):
            value = matrix[i, j]
            if not np.isfinite(value) or value < thresholds.redundant_correlation:
                continue
            if frozenset({varying[i], varying[j]}) in already_reported:
                continue
            pairs.append((varying[i], varying[j], float(value)))
    if not pairs:
        return [], []

    evidence = Evidence.create(
        kind="feature_redundant_pairs",
        scope=_scope(tuple(sorted({n for a, b, _ in pairs for n in (a, b)}))),
        value={
            "threshold": thresholds.redundant_correlation,
            "pairs": [
                {"left": a, "right": b, "correlation": round(r, 6)} for a, b, r in pairs
            ],
        },
        method="pearson_correlation_on_computed_features",
        description=f"{len(pairs)} feature pair(s) move together almost exactly.",
        assumptions=(
            "Measured on this dataset; a pair that is redundant here may "
            "separate on data with a wider range.",
        ),
    )
    findings = [
        Finding.create(
            title=f"{a} and {b} carry the same signal",
            summary=(
                f"{a} and {b} correlate at {r:.6f} across "
                f"{len(produced)} entities. They are defined differently, so "
                "this is a property of this data rather than of the definitions."
            ),
            severity="low",
            confidence=0.8,
            evidence_ids=(evidence.id,),
            recommendation=(
                "Check whether both are earning their place before shipping "
                "them; near-duplicates split importance between themselves."
            ),
            category=OBSERVATION,
        )
        for a, b, r in pairs
    ]
    return [evidence], findings


def _constant_findings(produced: pd.DataFrame) -> tuple[list[Evidence], list[Finding]]:
    """A feature with one value everywhere cannot inform anything."""
    constant = [
        name for name in produced.columns if produced[name].nunique(dropna=False) <= 1
    ]
    if not constant:
        return [], []
    evidence = Evidence.create(
        kind="feature_constant_values",
        scope=_scope(tuple(constant)),
        value={
            "features": [
                {"feature": name, "value": float(produced[name].iloc[0])}
                for name in constant
            ]
        },
        method="distinct_value_count_across_entities",
        description=f"{len(constant)} feature(s) hold one value for every entity.",
    )
    return [evidence], [
        Finding.create(
            title=f"{name} is the same for every entity",
            summary=(
                f"{name} holds {produced[name].iloc[0]:g} for all "
                f"{len(produced)} entities, so it cannot separate them. Either "
                "the definition is degenerate on this data or the column it "
                "reads does not vary here."
            ),
            severity="medium",
            confidence=1.0,
            evidence_ids=(evidence.id,),
            recommendation=(
                "Confirm on a wider extract before concluding the feature is "
                "useless; a constant here may vary in production."
            ),
            category=OBSERVATION,
        )
        for name in constant
    ]


def _fragility_findings(
    plan: FeaturePlan, stats: ExecutionStats, thresholds: Thresholds
) -> tuple[list[Evidence], list[Finding]]:
    """Features whose computation produced nothing usable and fell back.

    A fallback is not the same as a value that happens to equal the default:
    the first is a computation that failed, the second is an ordinary result.
    Only the first is counted, which is what keeps this quiet on clean data.
    """
    entities = stats.entity_count or 1
    degraded = {
        name: count
        for name, count in stats.fallback_counts.items()
        if count / entities >= thresholds.fallback_share
    }
    if not degraded:
        return [], []
    defaults = plan.defaults
    evidence = Evidence.create(
        kind="feature_default_fallbacks",
        scope=_scope(tuple(sorted(degraded))),
        value={
            "entities": entities,
            "threshold_share": thresholds.fallback_share,
            "features": [
                {
                    "feature": name,
                    "entities": count,
                    "share": round(count / entities, 4),
                    "default": defaults.get(name),
                }
                for name, count in sorted(degraded.items(), key=lambda i: -i[1])
            ],
        },
        method="non_finite_result_count_before_default_substitution",
        description=(
            f"{len(degraded)} feature(s) fell back to their declared default for "
            "a substantial share of entities."
        ),
    )
    findings = [
        Finding.create(
            title=(
                f"{name} falls back to its default for "
                f"{count / entities:.0%} of entities"
            ),
            summary=(
                f"{name} produced no usable value for {count:,} of {entities:,} "
                f"entities and was filled with {defaults.get(name)}. A default "
                "is indistinguishable from a genuine reading of that value, so "
                "a model cannot tell the two apart."
            ),
            severity="high" if count / entities > 0.5 else "medium",
            confidence=1.0,
            evidence_ids=(evidence.id,),
            recommendation=(
                "Either narrow the feature to entities where it is defined, or "
                "add a companion flag saying the value was imputed."
            ),
            category=QUALITY_ISSUE,
        )
        for name, count in sorted(degraded.items(), key=lambda item: -item[1])
    ]
    return [evidence], findings


def _cost_findings(
    plan: FeaturePlan, thresholds: Thresholds
) -> tuple[list[Evidence], list[Finding]]:
    """A feature doing a disproportionate share of the plan's expensive work.

    Computed from the operations in the plan, not from the clock, so the same
    plan produces the same finding on every machine and every run.
    """
    exclusive, total = _weight_by_feature(plan)
    if not total or len(exclusive) < 2:
        return [], []
    even = 1.0 / len(exclusive)
    limit = max(thresholds.cost_share_floor, even * thresholds.cost_share_multiple)
    heavy = [
        (name, weight)
        for name, weight in sorted(exclusive.items(), key=lambda item: -item[1])
        if weight / total >= limit or weight / total >= thresholds.cost_share_dominant
    ]
    if not heavy:
        return [], []

    evidence = Evidence.create(
        kind="feature_cost_concentration",
        scope=_scope(tuple(name for name, _ in heavy)),
        value={
            "threshold_share": round(limit, 4),
            "total_weight": total,
            "features": [
                {"feature": name, "weight": weight, "share": round(weight / total, 4)}
                for name, weight in heavy
            ],
        },
        method="static_operation_weights_attributed_to_the_only_reaching_feature",
        description=(
            f"{len(heavy)} feature(s) hold most of the plan's expensive work."
        ),
        assumptions=(
            "Cost is estimated from the operations in the plan, not measured, "
            "so it is reproducible but approximate.",
        ),
    )
    findings = [
        Finding.create(
            title=f"{name} dominates the cost of computing this feature set",
            summary=(
                f"{name} accounts for {weight / total:.0%} of the plan's "
                f"estimated work, against an even share of {even:.0%} across "
                f"{len(exclusive)} features. This counts only work no other "
                "feature shares, and is estimated from the operations "
                "themselves rather than timed, so it does not move between runs."
            ),
            severity="low",
            confidence=0.9,
            evidence_ids=(evidence.id,),
            recommendation=(
                "Worth confirming it earns that cost before it goes into a "
                "path with a latency budget."
            ),
            category=OBSERVATION,
        )
        for name, weight in heavy
    ]
    return [evidence], findings


def _opaque_findings(plan: FeaturePlan) -> list[Finding]:
    opaque = [item.name for item in plan.features if item.kind == "opaque"]
    if not opaque:
        return []
    return [
        Finding.create(
            title=f"{len(opaque)} feature(s) run unplanned, one entity at a time",
            summary=(
                f"{', '.join(opaque)} are declared opaque, so the planner cannot "
                "see inside them, share their work, or vectorise them across "
                "entities. They are correct; they are just paying the per-entity "
                "cost the rest of the plan avoids."
            ),
            severity="low",
            confidence=1.0,
            evidence_ids=(),
            recommendation=(
                "If one of these turns out to dominate the cost table, "
                "expressing it in the algebra is where the time is."
            ),
            category=OBSERVATION,
        )
    ]


def _point_in_time(
    plan: FeaturePlan, frame: pd.DataFrame
) -> tuple[list[Evidence], list[Finding], list[str]]:
    """Whether the frame carries rows the decision could not have seen."""
    if plan.time is None:
        return [], [], []
    if plan.reference_time is None:
        # Nothing to check against. This is stated as an assumption rather
        # than reported as a finding: it is true of every plan without a
        # declared decision moment, and a detector that fires on every run
        # tells the reader nothing.
        return (
            [],
            [],
            [
                "Time windows are anchored at each entity's most recent row. "
                "That is correct when the newest row is the event being scored, "
                "and wrong for a backfill assembled later; declare "
                "reference_time= to anchor on the decision moment instead."
            ],
        )

    stamps = pd.to_datetime(frame[plan.time], errors="coerce")
    anchors = pd.to_datetime(frame[plan.reference_time], errors="coerce")
    after = stamps > anchors
    affected = int(after.sum())
    if affected == 0:
        return [], [], []

    entities = int(frame.loc[after.fillna(False), plan.entity].nunique())
    total_entities = int(frame[plan.entity].nunique())
    evidence = Evidence.create(
        kind="feature_future_rows",
        scope=_scope((plan.time, plan.reference_time)),
        value={
            "rows_after_decision": affected,
            "rows": int(len(frame)),
            "entities_affected": entities,
            "entities": total_entities,
        },
        method="row_timestamp_compared_to_declared_reference_time",
        description=(
            f"{affected:,} of {len(frame):,} rows fall after the decision moment "
            f"they are attached to, across {entities:,} entities."
        ),
    )
    finding = Finding.create(
        title="The source frame contains rows from after the decision moment",
        summary=(
            f"{affected:,} rows across {entities:,} of {total_entities:,} entities "
            f"have {plan.time} later than {plan.reference_time}. The plan excludes "
            "them, so the features are point-in-time correct \u2014 but their "
            "presence "
            "means the extract was not filtered to the decision moment, and any "
            "other consumer of this frame that does not exclude them is leaking."
        ),
        severity="high",
        confidence=1.0,
        evidence_ids=(evidence.id,),
        recommendation=(
            "Filter the extract at the source. Relying on every downstream "
            "consumer to re-apply the cutoff is the arrangement that produces "
            "a model that scores well in training and badly in production."
        ),
        category=QUALITY_ISSUE,
    )
    return [evidence], [finding], []


def _target_leakage(plan: FeaturePlan) -> tuple[list[Evidence], list[Finding]]:
    """Features that read the outcome column they are meant to predict."""
    if plan.target is None:
        return [], []
    offenders: list[str] = []
    for item in plan.features:
        reads = {
            str(reached.param("name"))
            for reached in iter_nodes(item.node)
            if reached.op == "column"
        }
        if item.kind == "opaque":
            reads = set(item.requires)
        if plan.target in reads:
            offenders.append(item.name)
    if not offenders:
        return [], []

    evidence = Evidence.create(
        kind="feature_target_reads",
        scope=_scope(tuple(offenders)),
        value={"target": plan.target, "features": sorted(offenders)},
        method="column_reads_extracted_from_the_expression_tree",
        description=(
            f"{len(offenders)} feature(s) read the declared target {plan.target!r}."
        ),
    )
    return [evidence], [
        Finding.create(
            title=(
                f"{', '.join(sorted(offenders))} "
                + ("reads" if len(offenders) == 1 else "read")
                + " the target column"
            ),
            summary=(
                f"These features read {plan.target!r}, the column the model is "
                "meant to predict. This is read off the expression tree, so it "
                "is what the features actually touch rather than a guess from "
                "their names."
            ),
            severity="critical",
            confidence=1.0,
            evidence_ids=(evidence.id,),
            recommendation=(
                "Remove the target from these definitions. A feature that has "
                "seen the outcome will look excellent in validation and be "
                "unavailable at the moment a prediction is needed."
            ),
            category=QUALITY_ISSUE,
        )
    ]


def diagnose_plan(
    plan: FeaturePlan,
    frame: pd.DataFrame,
    *,
    thresholds: Thresholds | None = None,
) -> tuple[
    pd.DataFrame,
    tuple[Evidence, ...],
    tuple[Finding, ...],
    list[dict[str, Any]],
    tuple[AnalysisWarning, ...],
    tuple[str, ...],
    ExecutionStats,
]:
    """Execute the plan and report on what the definitions and results show."""
    from prism_eda.features.executors.pandas_exec import execute_instrumented

    limits = thresholds or Thresholds()
    produced, stats = execute_instrumented(
        plan.features,
        frame,
        entity=plan.entity,
        time=plan.time,
        reference_time=plan.reference_time,
    )

    evidence: list[Evidence] = [
        _sharing_evidence(plan, stats),
    ]
    findings: list[Finding] = []
    warnings: list[AnalysisWarning] = []
    assumptions: list[str] = []

    duplicate_evidence, duplicate_findings = _duplicate_findings(plan)
    identical = frozenset(
        frozenset(pair)
        for item in duplicate_evidence
        for group in item.value["groups"]
        for pair in _pairs(group)
    )

    for collected, produced_findings in (
        (duplicate_evidence, duplicate_findings),
        _redundancy_findings(produced, limits, already_reported=identical),
        _constant_findings(produced),
        _fragility_findings(plan, stats, limits),
        _target_leakage(plan),
    ):
        evidence.extend(collected)
        findings.extend(produced_findings)

    pit_evidence, pit_findings, pit_assumptions = _point_in_time(plan, frame)
    evidence.extend(pit_evidence)
    findings.extend(pit_findings)
    assumptions.extend(pit_assumptions)

    cost_evidence, cost_findings = _cost_findings(plan, limits)
    evidence.extend(cost_evidence)
    findings.extend(cost_findings)
    findings.extend(_opaque_findings(plan))

    if plan.verification is not None and plan.verification.sampling is not None:
        warnings.append(
            AnalysisWarning(
                code="sampled_feature_verification",
                message=plan.verification.sampling.limitations[0],
            )
        )

    return (
        produced,
        tuple(evidence),
        tuple(sort_findings(findings)),
        cost_table(plan, stats),
        tuple(warnings),
        tuple(assumptions),
        stats,
    )
