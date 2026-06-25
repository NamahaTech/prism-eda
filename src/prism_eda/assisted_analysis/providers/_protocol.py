"""Text protocol shared by text-completion providers.

Rather than rely on native function-calling (which varies across models, and is
unreliable on small/open models such as Gemma), Prism uses a portable
prompted-JSON protocol: the model is asked to reply with a single JSON object
describing its next action. This module renders a :class:`DecisionRequest` into a
prompt and parses the model's reply back into a :class:`ProviderDecision`.

Keeping this here (a) lets any text-completion provider reuse it and (b) makes the
parser unit-testable without a network call.
"""

from __future__ import annotations

import json
from typing import Any

from prism_eda.assisted_analysis.providers.base import (
    DecisionRequest,
    DraftFinding,
    ProviderDecision,
    ToolInvocation,
)

_INSTRUCTIONS = """\
You are Prism's data-analysis investigator. You DO NOT have access to the raw \
data. You can only learn about the dataset by calling the deterministic tools \
listed below; each returns trustworthy, pre-computed evidence. Your job is to \
reach the user's goal using the fewest tool calls, then report a short list of \
decision-first findings.

Hard rules:
- Reply with EXACTLY ONE JSON object and nothing else. No prose, no markdown.
- You may only call a tool from the "tools" list, with its documented arguments.
- Every finding you report MUST cite one or more evidence_ids that appeared in a \
tool result. Never invent an evidence id. If you cannot support a claim with \
evidence, do not make the claim.
- If the evidence does not justify any conclusion, finish with \
status "insufficient_evidence" and an empty findings list.
- Prefer signal over noise: report only findings that would actually change a \
decision. A handful of high-value findings beats a long list.

Respond with one of these two shapes.

To call a tool:
{"action": "call_tool", "thought": "<why>", "tool": "<tool_name>", \
"arguments": {<args>}}

To finish:
{"action": "finish", "thought": "<why>", "status": "completed" | \
"insufficient_evidence", "summary": "<one decision-first sentence>", \
"assumptions": ["..."], "findings": [{"title": "...", "summary": "...", \
"severity": "critical|high|medium|low|info", "confidence": 0.0-1.0, \
"evidence_ids": ["ev_..."], "recommendation": "..."}]}
"""


def render_prompt(request: DecisionRequest) -> str:
    """Render a decision request into a single self-contained prompt string.

    Everything is folded into one user turn (no separate system role) so the
    protocol works on models — like Gemma — that don't accept system
    instructions.
    """
    tools_block = json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "arguments": tool.parameters,
            }
            for tool in request.tools
        ],
        indent=2,
    )

    history_block = (
        json.dumps(
            [
                {
                    "tool": step.tool,
                    "arguments": step.arguments,
                    "result": step.result,
                    "evidence_ids": list(step.evidence_ids),
                }
                for step in request.history
            ],
            indent=2,
            default=str,
        )
        if request.history
        else "(no tools called yet)"
    )

    evidence_block = (
        ", ".join(request.available_evidence_ids)
        if request.available_evidence_ids
        else "(none yet — call a tool to gather evidence)"
    )

    sections = [
        _INSTRUCTIONS,
        f"\nGOAL: {request.goal}",
    ]
    if request.target:
        sections.append(f"TARGET COLUMN: {request.target}")
    if request.entity_id:
        sections.append(f"ENTITY ID COLUMN: {request.entity_id}")
    if request.timestamp:
        sections.append(f"TIMESTAMP COLUMN: {request.timestamp}")
    if request.domain_notes:
        sections.append(f"DOMAIN NOTES: {request.domain_notes}")
    if request.assumptions:
        sections.append("ASSUMPTIONS: " + "; ".join(request.assumptions))
    sections.extend(
        [
            f"REMAINING TOOL CALLS: {request.remaining_steps}",
            f"\nDATASET OVERVIEW:\n{request.dataset_overview}",
            f"\nTOOLS:\n{tools_block}",
            f"\nTOOL CALLS SO FAR:\n{history_block}",
            f"\nEVIDENCE IDS AVAILABLE TO CITE:\n{evidence_block}",
            "\nYour single JSON object:",
        ]
    )
    return "\n".join(sections)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of a model reply.

    Tolerates markdown code fences and leading/trailing prose, which small models
    often add despite instructions.
    """
    cleaned = text.strip()
    # Strip a leading ```json / ``` fence if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model reply.")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : index + 1])
    raise ValueError("Unbalanced JSON object in model reply.")


def parse_decision(text: str) -> ProviderDecision:
    """Parse a model reply into a :class:`ProviderDecision`.

    Raises ``ValueError`` if the reply cannot be interpreted; the orchestration
    layer turns that into a recoverable failure rather than a crash.
    """
    payload = _extract_json(text)
    action = str(payload.get("action", "")).lower()
    reasoning = str(payload.get("thought", ""))

    if action == "call_tool":
        tool = payload.get("tool")
        if not tool:
            raise ValueError("call_tool decision is missing a 'tool' name.")
        arguments = payload.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("Tool 'arguments' must be a JSON object.")
        return ProviderDecision(
            kind="call_tool",
            reasoning=reasoning,
            tool_calls=(ToolInvocation(tool=str(tool), arguments=arguments),),
        )

    if action == "finish":
        status = str(payload.get("status", "completed"))
        findings = tuple(
            _parse_finding(item) for item in payload.get("findings", []) or ()
        )
        kind = "insufficient" if status == "insufficient_evidence" else "finish"
        return ProviderDecision(
            kind=kind,
            reasoning=reasoning,
            findings=findings,
            summary=str(payload.get("summary", "")),
            assumptions=tuple(str(a) for a in payload.get("assumptions", []) or ()),
            status=status,
        )

    raise ValueError(f"Unknown action {action!r} in model reply.")


def _parse_finding(item: dict[str, Any]) -> DraftFinding:
    evidence_ids = tuple(str(e) for e in item.get("evidence_ids", []) or ())
    try:
        confidence = float(item.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    return DraftFinding(
        title=str(item.get("title", "Untitled finding")),
        summary=str(item.get("summary", "")),
        severity=str(item.get("severity", "info")).lower(),
        confidence=max(0.0, min(1.0, confidence)),
        evidence_ids=evidence_ids,
        recommendation=(
            str(item["recommendation"]) if item.get("recommendation") else None
        ),
    )
