"""Provider-neutral interface for AI-assisted investigation.

The deterministic core never imports this module. Providers translate between
Prism's neutral request/decision types and a concrete LLM SDK; the orchestration
graph only ever speaks these neutral types, so swapping Gemini for another
provider changes nothing upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A deterministic tool the model is allowed to call.

    ``parameters`` is a JSON-Schema-style object describing the tool's arguments.
    The model may only ever choose a registered tool; it can never run arbitrary
    code or read raw data.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A model's request to run one tool with concrete arguments."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepRecord:
    """A compact, privacy-safe record of one executed tool call.

    ``result`` holds only aggregate summaries (never raw rows), so it is safe to
    send back to the provider as context for the next decision.
    """

    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DraftFinding:
    """A finding proposed by the model, before evidence-citation validation."""

    title: str
    summary: str
    severity: str = "info"
    confidence: float = 0.7
    evidence_ids: tuple[str, ...] = ()
    recommendation: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Everything a provider needs to choose the next step.

    This is assembled by the orchestration graph and contains only
    privacy-filtered, aggregate context — never raw data or secrets.
    """

    goal: str
    dataset_overview: str
    tools: tuple[ToolSpec, ...]
    history: tuple[StepRecord, ...] = ()
    available_evidence_ids: tuple[str, ...] = ()
    remaining_steps: int = 0
    domain_notes: str | None = None
    assumptions: tuple[str, ...] = ()
    target: str | None = None
    entity_id: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderDecision:
    """One step the provider decided to take.

    ``kind`` is one of:
      * ``"call_tool"`` — run the tools in ``tool_calls`` and decide again;
      * ``"finish"`` — stop and synthesize ``findings`` / ``summary``;
      * ``"insufficient"`` — the model cannot justify a conclusion.
    """

    kind: str
    reasoning: str = ""
    tool_calls: tuple[ToolInvocation, ...] = ()
    findings: tuple[DraftFinding, ...] = ()
    summary: str = ""
    assumptions: tuple[str, ...] = ()
    status: str = "completed"

    def __post_init__(self) -> None:
        if self.kind not in {"call_tool", "finish", "insufficient"}:
            raise ValueError(
                "ProviderDecision.kind must be 'call_tool', 'finish', or "
                f"'insufficient', not {self.kind!r}"
            )


class LLMProvider(ABC):
    """A pluggable language-model backend.

    A provider has exactly one job: given a :class:`DecisionRequest`, return the
    next :class:`ProviderDecision`. How it does that — a real API call, a scripted
    fake — is entirely its own concern.
    """

    #: Human-readable provider name, surfaced in result metadata.
    name: str = "provider"

    #: Identifier of the underlying model, surfaced in result metadata.
    model: str = "unknown"

    @abstractmethod
    def decide(self, request: DecisionRequest) -> ProviderDecision:
        """Return the next step for the given request."""
