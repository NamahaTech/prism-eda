"""Non-mutating transformation plan models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TransformationStep:
    operation: str
    table: str
    columns: tuple[str, ...]
    parameters: dict[str, Any]
    rationale: str
    evidence_ids: tuple[str, ...]
    risk: str = "medium"
    requires_approval: bool = True


@dataclass(frozen=True, slots=True)
class TransformationPlan:
    steps: tuple[TransformationStep, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.steps
