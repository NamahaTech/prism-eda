"""LangGraph orchestration for an assisted investigation.

Flow: ``intake -> agent_step (loop) -> finalize``.

* **intake** announces the run.
* **agent_step** asks the provider for the next decision; if it's a tool call it
  runs the (deterministic) tool, banks the evidence, and loops; if it's a finish
  it stores the draft findings. The loop is bounded by a step budget.
* **finalize** validates that every reported finding cites real evidence,
  drops the rest, and builds the outcome.

The deterministic core never imports this module; this module depends on the core
(one-way), plus LangGraph for the state machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from prism_eda.assisted_analysis.providers.base import (
    DecisionRequest,
    LLMProvider,
    ProviderDecision,
    StepRecord,
)
from prism_eda.assisted_analysis.state import GraphState
from prism_eda.assisted_analysis.tools import (
    Tool,
    dataset_overview,
    tool_specs,
)
from prism_eda.dataset import Dataset
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import Finding, sort_findings
from prism_eda.privacy.models import PrivacyPolicy


@dataclass(frozen=True, slots=True)
class GraphDeps:
    """Everything the graph nodes need that isn't part of the mutable state."""

    dataset: Dataset
    privacy: PrivacyPolicy
    provider: LLMProvider
    registry: dict[str, Tool]
    callbacks: tuple[EventCallback, ...]
    max_steps: int


def _build_request(state: GraphState, deps: GraphDeps) -> DecisionRequest:
    evidence_ids = tuple(item.id for item in state.get("evidence", []))
    return DecisionRequest(
        goal=state["goal"],
        dataset_overview=dataset_overview(deps.dataset, deps.privacy),
        tools=tool_specs(deps.registry),
        history=tuple(state.get("history", [])),
        available_evidence_ids=evidence_ids,
        remaining_steps=state.get("remaining_steps", 0),
        domain_notes=state.get("domain_notes"),
        assumptions=tuple(state.get("assumptions", [])),
        target=state.get("target"),
        entity_id=state.get("entity_id"),
        timestamp=state.get("timestamp"),
    )


def _run_tool_calls(
    decision: ProviderDecision, state: GraphState, deps: GraphDeps
) -> None:
    """Execute the model's requested tools, banking evidence into the state."""
    for call in decision.tool_calls:
        tool = deps.registry.get(call.tool)
        if tool is None:
            state.setdefault("history", []).append(
                StepRecord(
                    tool=call.tool,
                    arguments=call.arguments,
                    result={"error": f"Unknown tool {call.tool!r}."},
                )
            )
            emit(
                deps.callbacks,
                Event(
                    EventKind.WARNING_RAISED,
                    f"Model requested unknown tool {call.tool!r}.",
                    stage="agent",
                ),
            )
            continue
        try:
            output = tool.run(deps.dataset, deps.privacy, call.arguments)
        except Exception as error:  # recoverable: feed the error back to the model
            state.setdefault("history", []).append(
                StepRecord(
                    tool=call.tool,
                    arguments=call.arguments,
                    result={"error": str(error)},
                )
            )
            emit(
                deps.callbacks,
                Event(
                    EventKind.METRIC_FAILED,
                    f"Tool {call.tool!r} failed: {error}",
                    stage="agent",
                ),
            )
            continue

        state.setdefault("evidence", []).extend(output.evidence)
        state.setdefault("recipe_findings", []).extend(output.findings)
        state.setdefault("artifacts", []).extend(output.artifacts)
        state.setdefault("history", []).append(
            StepRecord(
                tool=call.tool,
                arguments=call.arguments,
                result=output.summary,
                evidence_ids=tuple(item.id for item in output.evidence),
            )
        )
        for item in output.evidence:
            emit(
                deps.callbacks,
                Event(
                    EventKind.EVIDENCE_CREATED,
                    item.description,
                    stage="agent",
                    data={"evidence_id": item.id, "tool": call.tool},
                ),
            )


def build_graph(deps: GraphDeps):  # noqa: ANN201 - LangGraph's compiled type
    """Compile and return the investigation graph for the given dependencies."""

    def intake(state: GraphState) -> GraphState:
        emit(
            deps.callbacks,
            Event(
                EventKind.RUN_STARTED,
                f"Assisted investigation started for goal {state['goal']!r}.",
                stage="intake",
            ),
        )
        state.setdefault("history", [])
        state.setdefault("evidence", [])
        state.setdefault("recipe_findings", [])
        state.setdefault("artifacts", [])
        state["finished"] = False
        state["converged"] = True
        return state

    def agent_step(state: GraphState) -> GraphState:
        request = _build_request(state, deps)
        decision = deps.provider.decide(request)
        emit(
            deps.callbacks,
            Event(
                EventKind.STAGE_STARTED,
                decision.reasoning or f"Model chose to {decision.kind}.",
                stage="agent",
                data={"decision": decision.kind},
            ),
        )

        if decision.kind == "call_tool":
            if state.get("remaining_steps", 0) <= 0:
                # Out of budget but the model still wants to dig: stop gracefully.
                state["finished"] = True
                state["converged"] = False
                return state
            _run_tool_calls(decision, state, deps)
            state["remaining_steps"] = state.get("remaining_steps", 0) - 1
            return state

        # finish or insufficient
        state["draft_findings"] = list(decision.findings)
        state["model_summary"] = decision.summary
        state["model_assumptions"] = list(decision.assumptions)
        state["status"] = (
            "insufficient_evidence"
            if decision.kind == "insufficient"
            else decision.status
        )
        state["finished"] = True
        return state

    def finalize(state: GraphState) -> GraphState:
        emit(
            deps.callbacks,
            Event(
                EventKind.STAGE_STARTED,
                "Validating evidence citations.",
                stage="synthesis",
            ),
        )
        valid_ids = {item.id for item in state.get("evidence", [])}
        drafts = state.get("draft_findings", [])

        validated: list[Finding] = []
        dropped = 0
        for draft in drafts:
            cited = tuple(eid for eid in draft.evidence_ids if eid in valid_ids)
            if not cited:
                dropped += 1
                continue
            validated.append(
                Finding.create(
                    title=draft.title,
                    summary=draft.summary,
                    severity=draft.severity,
                    confidence=draft.confidence,
                    evidence_ids=cited,
                    recommendation=draft.recommendation,
                )
            )

        converged = state.get("converged", True)
        metadata: dict[str, object] = {
            "engine": "assisted",
            "provider": deps.provider.name,
            "model": deps.provider.model,
            "tool_calls": len(state.get("history", [])),
            "uncited_findings_dropped": dropped,
            "converged": converged,
        }

        if not converged:
            # Model never converged within budget: fall back to the deterministic
            # recipe findings it already gathered, clearly flagged.
            validated = list(state.get("recipe_findings", []))
            status = "completed_with_warnings" if validated else "insufficient_evidence"
            summary = (
                "The investigator did not converge within its step budget; "
                "showing the deterministic findings gathered so far."
            )
        elif not validated:
            status = "insufficient_evidence"
            summary = state.get("model_summary") or (
                "The investigation did not find evidence to support a conclusion."
            )
        else:
            status = state.get("status", "completed")
            if status == "insufficient_evidence":
                status = "completed_with_warnings"
            summary = state.get("model_summary") or "Investigation complete."

        state["findings"] = sort_findings(validated)
        state["status"] = status
        state["summary"] = summary
        state["metadata"] = metadata
        emit(
            deps.callbacks,
            Event(
                EventKind.RUN_COMPLETED,
                summary,
                stage="synthesis",
                progress=1.0,
                data={"status": status, "findings": len(state["findings"])},
            ),
        )
        return state

    def route_after_agent(state: GraphState) -> str:
        return "finalize" if state.get("finished") else "agent_step"

    graph: StateGraph = StateGraph(GraphState)
    graph.add_node("intake", intake)
    graph.add_node("agent_step", agent_step)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "agent_step")
    graph.add_conditional_edges(
        "agent_step",
        route_after_agent,
        {"agent_step": "agent_step", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()
