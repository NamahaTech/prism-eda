"""The public entry point for AI-assisted investigation.

``Investigator`` wires a dataset, a provider, and a privacy policy into the
orchestration graph; ``InvestigationSession.run()`` executes it and returns a
standard :class:`~prism_eda.results.AnalysisResult` — the *same* type the
deterministic recipes return, so reports and JSON export work unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism_eda.assisted_analysis.graph import GraphDeps, build_graph
from prism_eda.assisted_analysis.interpretation import interpret
from prism_eda.assisted_analysis.providers.base import LLMProvider
from prism_eda.assisted_analysis.state import GraphState
from prism_eda.assisted_analysis.tools import build_tool_registry
from prism_eda.catalog.loaders import DataSource
from prism_eda.config import AnalysisContext
from prism_eda.dataset import Dataset
from prism_eda.events import EventCallback
from prism_eda.privacy.models import PrivacyPolicy
from prism_eda.results import AnalysisResult, AnalysisStatus, AnalysisWarning

_STATUS_MAP = {
    "completed": AnalysisStatus.COMPLETED,
    "completed_with_warnings": AnalysisStatus.COMPLETED_WITH_WARNINGS,
    "insufficient_evidence": AnalysisStatus.INSUFFICIENT_EVIDENCE,
    "no_meaningful_structure": AnalysisStatus.NO_MEANINGFUL_STRUCTURE,
    "failed": AnalysisStatus.FAILED,
}

_DISCLOSURE = (
    "Findings were proposed by an AI investigator and kept only where they cite "
    "deterministic evidence produced by Prism's tools."
)


class Investigator:
    """Plan and explain an analysis with an LLM over deterministic tools.

    The model can only call registered tools; it never sees raw rows and never
    runs code. Every finding it reports is validated against real evidence before
    it enters the result.

    Parameters:
        source: A :class:`Dataset` or anything ``pe.load`` accepts.
        provider: The :class:`LLMProvider` that drives decisions.
        privacy: Controls what dataset context may reach the provider. Defaults to
            sending column names and aggregates but never raw cell values.
        callbacks: Event observers (progress, evidence, completion).
        max_steps: Maximum number of tool calls before the run is forced to
            conclude.
    """

    def __init__(
        self,
        source: DataSource | Dataset,
        *,
        provider: LLMProvider,
        privacy: PrivacyPolicy | None = None,
        callbacks: Sequence[EventCallback] = (),
        max_steps: int = 8,
    ) -> None:
        self.dataset = source if isinstance(source, Dataset) else Dataset.load(source)
        self.provider = provider
        self.privacy = privacy or PrivacyPolicy()
        self.callbacks = tuple(callbacks)
        self.max_steps = max_steps

    def start(
        self,
        goal: str = "profile",
        *,
        context: AnalysisContext | Mapping[str, Any] | None = None,
    ) -> InvestigationSession:
        """Create a runnable investigation session for ``goal``."""
        normalized_goal = goal.strip().lower().replace("-", "_")
        if context is None:
            analysis_context = AnalysisContext(goal=normalized_goal)
        elif isinstance(context, AnalysisContext):
            analysis_context = context
        else:
            analysis_context = AnalysisContext(goal=normalized_goal, **dict(context))
        return InvestigationSession(self, normalized_goal, analysis_context)


class InvestigationSession:
    """A single, runnable investigation. Held in memory; re-runnable."""

    def __init__(
        self,
        investigator: Investigator,
        goal: str,
        context: AnalysisContext,
    ) -> None:
        self._investigator = investigator
        self.goal = goal
        self.context = context

    def run(self) -> AnalysisResult:
        """Execute the investigation and return a standard ``AnalysisResult``."""
        inv = self._investigator
        deps = GraphDeps(
            dataset=inv.dataset,
            privacy=inv.privacy,
            provider=inv.provider,
            registry=build_tool_registry(),
            callbacks=inv.callbacks,
            max_steps=inv.max_steps,
        )
        graph = build_graph(deps)
        initial: GraphState = {
            "goal": self.goal,
            "domain_notes": self.context.domain_notes,
            "assumptions": list(self.context.assumptions),
            "target": self.context.target,
            "entity_id": self.context.entity_id,
            "timestamp": self.context.timestamp,
            "remaining_steps": inv.max_steps,
        }
        recursion_limit = inv.max_steps * 2 + 10
        final: GraphState = graph.invoke(
            initial, config={"recursion_limit": recursion_limit}
        )
        result = self._to_result(final)
        interpretation = self._interpret(final)
        if interpretation:
            result.metadata["ai_interpretation"] = interpretation
        return result

    def _interpret(self, state: GraphState) -> dict[str, Any]:
        """Grounded value-add layer: semantic reads, narrative, next steps.

        Kept out of the tool loop on purpose — it reasons over the evidence the
        loop already gathered, so a slow or unsupported text endpoint degrades to
        no interpretation rather than derailing the investigation.
        """
        inv = self._investigator
        try:
            return interpret(
                inv.provider,
                catalog=inv.dataset.catalog(),
                privacy=inv.privacy,
                findings=state.get("findings", []),
                evidence=state.get("evidence", []),
                goal=self.goal,
                summary=state.get("summary", ""),
            )
        except Exception:  # noqa: BLE001 - interpretation is best-effort
            return {}

    def _to_result(self, state: GraphState) -> AnalysisResult:
        status = _STATUS_MAP.get(
            state.get("status", "completed"), AnalysisStatus.COMPLETED
        )
        warnings: list[AnalysisWarning] = []
        metadata = dict(state.get("metadata", {}))
        if metadata.get("converged") is False:
            warnings.append(
                AnalysisWarning(
                    code="investigator_not_converged",
                    message=(
                        "The investigator reached its step budget before "
                        "concluding; deterministic findings are shown instead."
                    ),
                )
            )

        assumptions = (
            tuple(self.context.assumptions)
            + tuple(state.get("model_assumptions", []))
            + (_DISCLOSURE,)
        )

        return AnalysisResult(
            goal=self.goal,
            status=status,
            summary=state.get("summary", "Investigation complete."),
            catalog=self._investigator.dataset.catalog(),
            findings=tuple(state.get("findings", [])),
            evidence=tuple(state.get("evidence", [])),
            artifacts=tuple(state.get("artifacts", [])),
            assumptions=assumptions,
            warnings=tuple(warnings),
            metadata=metadata,
        )
