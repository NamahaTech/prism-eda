"""Structured artifacts rendered by reports and consumed by downstream tools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from prism_eda._serialization import to_jsonable


@dataclass(frozen=True, slots=True)
class Artifact:
    """A typed table, graph, chart, or other result-backed report artifact."""

    id: str
    kind: str
    title: str
    data: dict[str, Any]
    evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        title: str,
        data: dict[str, Any],
        evidence_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        payload = {
            "kind": kind,
            "title": title,
            "data": to_jsonable(data),
            "evidence_ids": evidence_ids,
            "metadata": to_jsonable(metadata or {}),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return cls(
            id=f"artifact_{digest}",
            kind=kind,
            title=title,
            data=to_jsonable(data),
            evidence_ids=evidence_ids,
            metadata=metadata or {},
        )
