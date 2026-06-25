"""Graph state for an assisted investigation.

The state holds only privacy-safe, aggregate context — tool summaries and the
deterministic evidence/findings the tools produced. It never contains raw rows or
API keys, so it is safe to inspect, log, or (in future) checkpoint.
"""

from __future__ import annotations

from typing import Any, TypedDict

from prism_eda.artifacts import Artifact
from prism_eda.assisted_analysis.providers.base import DraftFinding, StepRecord
from prism_eda.evidence.models import Evidence, Finding


class GraphState(TypedDict, total=False):
    """Mutable state threaded through the investigation graph."""

    # Inputs
    goal: str
    domain_notes: str | None
    assumptions: list[str]
    target: str | None
    entity_id: str | None
    timestamp: str | None

    # Budget
    remaining_steps: int

    # Accumulated during the tool loop
    history: list[StepRecord]
    evidence: list[Evidence]
    recipe_findings: list[Finding]
    artifacts: list[Artifact]

    # Produced by the model when it finishes
    draft_findings: list[DraftFinding]
    model_summary: str
    model_assumptions: list[str]

    # Control + outcome
    finished: bool
    converged: bool
    status: str
    summary: str
    findings: list[Finding]
    metadata: dict[str, Any]
