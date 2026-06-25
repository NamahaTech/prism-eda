"""A deterministic, offline provider for tests, docs, and demos.

``FakeProvider`` implements the same :class:`LLMProvider` interface as the real
Gemini provider but makes no network calls, so the entire investigation flow can
be exercised reproducibly.

Two modes:

* **Scripted** — pass a ``script`` of decisions (or callables that build a
  decision from the live request). Gives a test exact control over the flow.
* **Default policy** — with no script, it calls the recipe tool matching the goal
  once, then finishes by promoting that recipe's findings into cited findings.
  This mirrors what a competent model does and lets docs show a real end-to-end
  run without an API key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from prism_eda.assisted_analysis.providers.base import (
    DecisionRequest,
    DraftFinding,
    LLMProvider,
    ProviderDecision,
    ToolInvocation,
)

ScriptStep = ProviderDecision | Callable[[DecisionRequest], ProviderDecision]

_GOAL_TO_TOOL = {
    "classification": "assess_classification",
    "classify": "assess_classification",
    "anomaly_detection": "detect_anomalies",
    "anomaly": "detect_anomalies",
    "outlier_detection": "detect_anomalies",
    "schema_discovery": "discover_schema",
    "discover_schema": "discover_schema",
    "profile": "profile_dataset",
    "minimal_eda": "profile_dataset",
}


class FakeProvider(LLMProvider):
    """An offline provider used for deterministic flows."""

    name = "fake"
    model = "fake-deterministic"

    def __init__(self, *, script: Sequence[ScriptStep] | None = None) -> None:
        self._script = list(script) if script is not None else None
        self._cursor = 0

    def decide(self, request: DecisionRequest) -> ProviderDecision:
        if self._script is not None:
            return self._next_scripted(request)
        return self._default_policy(request)

    def _next_scripted(self, request: DecisionRequest) -> ProviderDecision:
        assert self._script is not None
        if self._cursor >= len(self._script):
            # Script ran out: finish cleanly with whatever evidence exists.
            return ProviderDecision(
                kind="finish",
                summary="Investigation complete.",
                findings=(),
                status="completed",
            )
        step = self._script[self._cursor]
        self._cursor += 1
        return step(request) if callable(step) else step

    def _default_policy(self, request: DecisionRequest) -> ProviderDecision:
        if not request.history:
            tool = _GOAL_TO_TOOL.get(request.goal, "profile_dataset")
            arguments: dict[str, object] = {}
            if tool == "assess_classification" and request.target:
                arguments["target"] = request.target
            elif tool == "detect_anomalies" and request.target:
                arguments["target"] = request.target
            return ProviderDecision(
                kind="call_tool",
                reasoning=f"Gather evidence for goal {request.goal!r}.",
                tool_calls=(ToolInvocation(tool=tool, arguments=arguments),),
            )

        # Promote the recipe's findings into cited findings.
        findings: list[DraftFinding] = []
        for step in request.history:
            for item in step.result.get("findings", []):
                evidence_ids = tuple(item.get("evidence_ids", ()))
                if not evidence_ids:
                    continue
                findings.append(
                    DraftFinding(
                        title=item.get("title", "Finding"),
                        summary=item.get("summary", ""),
                        severity=item.get("severity", "info"),
                        confidence=float(item.get("confidence", 0.7)),
                        evidence_ids=evidence_ids,
                        recommendation=item.get("recommendation"),
                    )
                )

        last = request.history[-1].result
        summary = last.get("summary") or "Investigation complete."
        status = "completed" if findings else "insufficient_evidence"
        return ProviderDecision(
            kind="finish" if findings else "insufficient",
            reasoning="Summarize the gathered evidence.",
            findings=tuple(findings),
            summary=summary,
            status=status,
        )
