"""Framework-neutral events for progress, warnings, and future interviews."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    PROGRESS_UPDATED = "progress_updated"
    SAMPLING_SELECTED = "sampling_selected"
    WARNING_RAISED = "warning_raised"
    METRIC_FAILED = "metric_failed"
    QUESTION_ASKED = "question_asked"
    EVIDENCE_CREATED = "evidence_created"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    message: str
    stage: str | None = None
    progress: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class EventCallback(Protocol):
    def __call__(self, event: Event) -> None: ...


def emit(callbacks: tuple[EventCallback, ...], event: Event) -> None:
    """Dispatch an event without allowing observer failures to stop analysis."""
    for callback in callbacks:
        try:
            callback(event)
        except Exception:
            continue
