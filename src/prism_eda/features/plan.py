"""The planned form of a feature set, and the things you can do with it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from prism_eda.features.contract import FeatureContract, build_contract
from prism_eda.features.executors.pandas_exec import execute_planned
from prism_eda.features.executors.reference import (
    execute_reference as _execute_reference,
)
from prism_eda.features.ir import Node, iter_nodes
from prism_eda.features.results import (
    FeaturePlanResult,
    FeatureRun,
    build_summary,
)
from prism_eda.features.tracing import TracedFeature
from prism_eda.features.verify import (
    DEFAULT_RTOL,
    DEFAULT_SAMPLE_ENTITIES,
    VerificationReport,
    compare_frames,
    oracle_frame,
    sample_entities,
)
from prism_eda.results import AnalysisStatus, SamplingRecord

__all__ = ["FeaturePlan"]


@dataclass(frozen=True, slots=True)
class FeaturePlan:
    """A traced feature set bound to a schema, ready to execute."""

    features: tuple[TracedFeature, ...]
    entity: str
    time: str | None
    schema: tuple[str, ...]
    reference_time: str | None = None
    target: str | None = None
    verification: VerificationReport | None = None

    # -- execution ---------------------------------------------------------
    def execute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute every feature, one row per entity, in declared order.

        Uses the planned executor: one sort, one grouper, each normalisation
        and window mask built once. Never mutates the caller's frame
        (invariant 1).
        """
        return execute_planned(
            self.features,
            frame,
            entity=self.entity,
            time=self.time,
            reference_time=self.reference_time,
        )

    def execute_reference(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute via the unoptimised oracle. Used by verification."""
        return _execute_reference(
            self.features,
            frame,
            entity=self.entity,
            time=self.time,
            reference_time=self.reference_time,
        )

    def run(self, frame: pd.DataFrame) -> FeatureRun:
        """Compute the features and the report on them, in one pass.

        Returned together because producing the report means producing the
        features; handing back one would mean doing the work twice.
        """
        from prism_eda.features.diagnostics import diagnose_plan

        (
            produced,
            evidence,
            findings,
            costs,
            warnings,
            assumptions,
            stats,
        ) = diagnose_plan(self, frame)

        status = AnalysisStatus.COMPLETED
        if warnings:
            status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        if stats.entity_count == 0:
            status = AnalysisStatus.INSUFFICIENT_EVIDENCE

        sampling: tuple[SamplingRecord, ...] = ()
        if self.verification is not None and self.verification.sampling is not None:
            sampling = (self.verification.sampling,)

        report = FeaturePlanResult(
            goal="feature_plan",
            status=status,
            summary=build_summary(
                feature_count=len(self.features),
                entity_count=stats.entity_count,
                findings=findings,
                shared_nodes=len(self.shared_nodes()),
                total_nodes=len(self.nodes),
                has_warnings=bool(warnings),
            ),
            entity=self.entity,
            contract=self.contract(),
            time=self.time,
            reference_time=self.reference_time,
            plan_description=self.explain(),
            findings=findings,
            evidence=evidence,
            assumptions=assumptions,
            warnings=warnings,
            sampling=sampling,
            metadata={
                "entity_count": stats.entity_count,
                "row_count": stats.row_count,
                "seconds": round(stats.total_seconds, 6),
                "shared_nodes": len(self.shared_nodes()),
                "distinct_nodes": len(self.nodes),
                # Measured, so it is metadata about this run rather than
                # evidence about the data: it would differ on another machine.
                "cost": costs,
                "verified": (
                    self.verification.ok if self.verification is not None else None
                ),
            },
        )
        return FeatureRun(features=produced, report=report)

    def diagnose(self, frame: pd.DataFrame) -> FeaturePlanResult:
        """Compute the features and return only the report about them."""
        return self.run(frame).report

    def contract(self) -> FeatureContract:
        """The exportable description of what this plan produces."""
        return build_contract(
            self.features,
            entity=self.entity,
            time=self.time,
            reference_time=self.reference_time,
        )

    # -- verification ------------------------------------------------------
    def verify(
        self,
        frame: pd.DataFrame,
        *,
        sample: int = DEFAULT_SAMPLE_ENTITIES,
        rtol: float = DEFAULT_RTOL,
        seed: int = 42,
    ) -> VerificationReport:
        """Check the planned answer against the unoptimised one.

        Samples whole entities rather than rows: a feature reads an entity's
        whole history, so dropping rows would change the computation instead
        of sampling it.
        """
        subset, record = sample_entities(
            frame, entity=self.entity, limit=sample, seed=seed
        )
        return compare_frames(
            self.execute_reference(subset),
            self.execute(subset),
            rtol=rtol,
            kind="planner_soundness",
            sampling=record,
            features=self.feature_names,
        )

    def verify_against(
        self,
        function: Callable[[pd.DataFrame], Any],
        frame: pd.DataFrame,
        *,
        sample: int = DEFAULT_SAMPLE_ENTITIES,
        rtol: float = DEFAULT_RTOL,
        seed: int = 42,
    ) -> VerificationReport:
        """Check this plan against feature code you already have.

        ``function`` receives one entity's rows and returns a mapping of
        feature name to value -- the shape hand-written feature code already
        has. Only features present in both are compared, so an existing set
        can be ported a few at a time with proof at every step.

        Internal consistency cannot establish this: the plan agreeing with its
        own reference executor says nothing about whether it agrees with the
        code it is replacing.
        """
        subset, record = sample_entities(
            frame, entity=self.entity, limit=sample, seed=seed
        )
        expected = oracle_frame(function, subset, entity=self.entity, time=self.time)
        return compare_frames(
            expected,
            self.execute(subset),
            rtol=rtol,
            kind="migration_fidelity",
            sampling=record,
            features=self.feature_names,
        )

    # -- introspection -----------------------------------------------------
    @property
    def nodes(self) -> tuple[Node, ...]:
        """Every distinct IR node across all features, children first."""
        return tuple(iter_nodes([item.node for item in self.features]))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Output column names, in contract order."""
        return tuple(item.name for item in self.features)

    @property
    def defaults(self) -> dict[str, float]:
        """Each feature's declared fallback value."""
        return {item.name: item.default for item in self.features}

    def shared_nodes(self) -> tuple[Node, ...]:
        """Nodes reached by more than one feature.

        These are what the planner has to gain from: each one is work the
        unoptimised path would repeat once per feature that needs it.
        """
        counts: dict[str, int] = {}
        for item in self.features:
            for reached in iter_nodes(item.node):
                counts[reached.id] = counts.get(reached.id, 0) + 1
        by_id = {reached.id: reached for reached in self.nodes}
        return tuple(
            by_id[node_id]
            for node_id, count in counts.items()
            if count > 1 and node_id in by_id
        )

    def explain(self) -> str:
        """A readable account of the plan and what it shares."""
        lines = [
            f"FeaturePlan over entity={self.entity!r}"
            + (f", time={self.time!r}" if self.time else ""),
            f"  {len(self.features)} feature(s), {len(self.nodes)} distinct node(s)",
        ]
        windows = [n for n in self.nodes if n.op == "window"]
        if windows:
            described = ", ".join(
                f"{n.param('name')}({n.param('kind')})" for n in windows
            )
            lines.append(f"  windows: {described}")
        aggregates = [n for n in self.nodes if n.op == "agg"]
        lines.append(f"  aggregates: {len(aggregates)}")
        shared = self.shared_nodes()
        if shared:
            kinds: dict[str, int] = {}
            for reached in shared:
                kinds[reached.op] = kinds.get(reached.op, 0) + 1
            summary = ", ".join(f"{op} x{n}" for op, n in sorted(kinds.items()))
            lines.append(f"  shared subexpressions: {len(shared)} ({summary})")
        else:
            lines.append("  shared subexpressions: none")
        opaque = [item for item in self.features if item.kind == "opaque"]
        if opaque:
            names = ", ".join(item.name for item in opaque)
            lines.append(f"  opaque (unplanned, runs per entity): {names}")
        lines.append("  output order: " + ", ".join(self.feature_names))
        if self.verification is not None:
            lines.append("  verified: " + self.verification.describe().splitlines()[0])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A machine-readable description of the plan."""
        return {
            "entity": self.entity,
            "time": self.time,
            "features": [
                {
                    "name": item.name,
                    "default": item.default,
                    "kind": item.kind,
                    "root": item.node.id,
                }
                for item in self.features
            ],
            "nodes": [
                {
                    "id": reached.id,
                    "op": reached.op,
                    "level": reached.level.value,
                    "children": [child.id for child in reached.children],
                }
                for reached in self.nodes
            ],
            "shared_node_ids": sorted(n.id for n in self.shared_nodes()),
            "verified": self.verification.ok if self.verification is not None else None,
        }
