"""Analysis context and execution configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class AnalysisMode(StrEnum):
    """Compute presets used by analysis recipes."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


SamplingMode = Literal["auto", "disabled"]


@dataclass(slots=True)
class AnalysisContext:
    """Domain and task information that changes how evidence is interpreted."""

    goal: str = "profile"
    target: str | None = None
    entity_id: str | None = None
    timestamp: str | None = None
    groups: tuple[str, ...] = ()
    domain_notes: str | None = None
    assumptions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisConfig:
    """Controls reproducibility, compute depth, and evidence thresholds."""

    mode: AnalysisMode | str = AnalysisMode.STANDARD
    sampling: SamplingMode = "auto"
    random_seed: int = 42
    allow_insufficient_evidence: bool = False

    def __post_init__(self) -> None:
        self.mode = AnalysisMode(self.mode)
        if self.sampling not in {"auto", "disabled"}:
            raise ValueError("sampling must be 'auto' or 'disabled'")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
