"""Structured evidence and finding models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from prism_eda._serialization import to_jsonable


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    table: str | None = None
    columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    kind: str
    scope: EvidenceScope
    value: Any
    method: str
    description: str
    confidence: float = 1.0
    assumptions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        scope: EvidenceScope,
        value: Any,
        method: str,
        description: str,
        confidence: float = 1.0,
        assumptions: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        payload = {
            "kind": kind,
            "scope": to_jsonable(scope),
            "value": to_jsonable(value),
            "method": method,
            "assumptions": assumptions,
            "metadata": to_jsonable(metadata or {}),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return cls(
            id=f"ev_{digest}",
            kind=kind,
            scope=scope,
            value=to_jsonable(value),
            method=method,
            description=description,
            confidence=confidence,
            assumptions=assumptions,
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class Finding:
    id: str
    title: str
    summary: str
    severity: str
    confidence: float
    evidence_ids: tuple[str, ...]
    recommendation: str | None = None

    @classmethod
    def create(
        cls,
        *,
        title: str,
        summary: str,
        severity: str,
        confidence: float,
        evidence_ids: tuple[str, ...],
        recommendation: str | None = None,
    ) -> Finding:
        digest = hashlib.sha256(
            json.dumps(
                {"title": title, "evidence_ids": evidence_ids}, sort_keys=True
            ).encode()
        ).hexdigest()[:16]
        return cls(
            id=f"finding_{digest}",
            title=title,
            summary=summary,
            severity=severity,
            confidence=confidence,
            evidence_ids=evidence_ids,
            recommendation=recommendation,
        )
