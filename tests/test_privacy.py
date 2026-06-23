"""Tests for the privacy policy controls used by future AI-assisted analysis."""

from __future__ import annotations

from prism_eda.privacy import ColumnPolicy, PrivacyAction, PrivacyPolicy


def test_safe_column_name_applies_each_action() -> None:
    policy = PrivacyPolicy(
        columns={
            "email": ColumnPolicy("exclude"),
            "name": ColumnPolicy("alias"),
            "notes": ColumnPolicy("redact"),
        }
    )

    assert policy.safe_column_name("email") is None
    assert policy.safe_column_name("name").startswith("column_")
    assert policy.safe_column_name("notes") == "redacted_column"
    assert policy.safe_column_name("age") == "age"  # default allow


def test_send_column_names_false_aliases_everything() -> None:
    policy = PrivacyPolicy(send_column_names=False)
    safe = policy.safe_column_name("age")
    assert safe is not None and safe.startswith("column_")


def test_alias_is_stable_within_policy_and_keyed_per_policy() -> None:
    policy = PrivacyPolicy()
    alias = policy.alias("secret-value")
    assert alias == policy.alias("secret-value")  # deterministic within a policy

    other = PrivacyPolicy()
    # Independent random HMAC keys make collisions astronomically unlikely.
    assert other.alias("secret-value") != alias


def test_describe_column_respects_policy() -> None:
    policy = PrivacyPolicy(columns={"email": ColumnPolicy("exclude")})

    assert policy.describe_column("email", {"name": "email"}) is None

    described = policy.describe_column(
        "age", {"name": "age", "dtype": "int64", "top_values": [1, 2, 3]}
    )
    assert described is not None
    assert described["name"] == "age"
    assert described["privacy_action"] == PrivacyAction.ALLOW.value
    # Raw values stay out of the payload unless explicitly allowed.
    assert "top_values" not in described


def test_describe_column_can_include_raw_values_when_opted_in() -> None:
    policy = PrivacyPolicy(allow_raw_values=True)
    described = policy.describe_column("age", {"name": "age", "top_values": [1, 2, 3]})
    assert described is not None
    assert described["top_values"] == [1, 2, 3]
