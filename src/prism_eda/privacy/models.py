"""Privacy controls for AI-assisted analysis payloads."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PrivacyAction(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    ALIAS = "alias"
    EXCLUDE = "exclude"


@dataclass(frozen=True, slots=True)
class ColumnPolicy:
    action: PrivacyAction | str = PrivacyAction.ALLOW

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", PrivacyAction(self.action))


@dataclass(slots=True)
class PrivacyPolicy:
    """Controls what dataset context may be sent to an AI provider."""

    default: PrivacyAction | str = PrivacyAction.ALLOW
    columns: dict[str, ColumnPolicy] = field(default_factory=dict)
    send_column_names: bool = True
    allow_raw_values: bool = False
    _hmac_key: bytes = field(default_factory=lambda: secrets.token_bytes(32))

    def __post_init__(self) -> None:
        self.default = PrivacyAction(self.default)

    def policy_for(self, column: str) -> ColumnPolicy:
        return self.columns.get(column, ColumnPolicy(self.default))

    def alias(self, value: object, *, prefix: str = "alias") -> str:
        digest = hmac.new(
            self._hmac_key,
            str(value).encode("utf-8", errors="replace"),
            hashlib.sha256,
        ).hexdigest()[:16]
        return f"{prefix}_{digest}"

    def safe_column_name(self, column: str) -> str | None:
        action = self.policy_for(column).action
        if action == PrivacyAction.EXCLUDE:
            return None
        if not self.send_column_names or action == PrivacyAction.ALIAS:
            return self.alias(column, prefix="column")
        if action == PrivacyAction.REDACT:
            return "redacted_column"
        return column

    def describe_column(self, column: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        safe_name = self.safe_column_name(column)
        if safe_name is None:
            return None
        action = self.policy_for(column).action
        sanitized = {
            key: value
            for key, value in payload.items()
            if key not in {"name", "top_values"}
        }
        sanitized["name"] = safe_name
        sanitized["privacy_action"] = action.value
        if self.allow_raw_values and action == PrivacyAction.ALLOW:
            sanitized["top_values"] = payload.get("top_values", ())
        return sanitized


__all__ = ["ColumnPolicy", "PrivacyAction", "PrivacyPolicy"]
