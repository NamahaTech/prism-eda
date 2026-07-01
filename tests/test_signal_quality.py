"""Regression tests for report signal-to-noise.

These lock in the behavior audited on 2026-06-23: the recipes must surface the
findings that matter (leakage, identifiers) and must not emit noise (numeric
columns flagged as high-cardinality, clean-tail univariate findings, every
pairwise conditional combination, spurious one-to-one relationships).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import prism_eda as pe
from prism_eda.analysis.anomaly import _MAX_CONDITIONAL_FINDINGS
from prism_eda.evidence.models import SEVERITY_RANK


# --------------------------------------------------------------------------- #
# Classification correctness
# --------------------------------------------------------------------------- #
def test_leakage_detected_on_imbalanced_target_via_value_rule() -> None:
    """The original bug: a near-perfect value rule was undetectable when the
    majority rate was high, because the threshold exceeded 1.0."""
    target = [0] * 90 + [1] * 10  # majority rate 0.9
    # "code" predicts the label perfectly but is neither an exact copy nor
    # name-overlapping the target, so only the value-rule screen can catch it.
    code = ["A"] * 90 + ["B"] * 10
    frame = pd.DataFrame({"status": target, "code": code, "noise": list(range(100))})

    result = pe.classification(frame, target="status", sampling="disabled")

    leakage = [
        item
        for item in result.evidence
        if item.kind == "classification_leakage_candidate"
    ]
    assert any(item.scope.columns[0] == "code" for item in leakage)
    leakage_findings = [
        finding
        for finding in result.findings
        if "Potential target leakage" in finding.title
    ]
    assert leakage_findings
    assert any(finding.severity in {"high", "critical"} for finding in leakage_findings)


def test_perfect_predictor_is_not_reported_as_ready_to_model() -> None:
    target = [0] * 90 + [1] * 10
    code = ["A"] * 90 + ["B"] * 10
    frame = pd.DataFrame({"status": target, "code": code})

    result = pe.classification(frame, target="status", sampling="disabled")

    # The leaky feature must be excluded from the probe, so the probe must not
    # certify strong separability driven by the leak.
    probe = [
        item for item in result.evidence if item.kind == "classification_probe_model"
    ]
    for item in probe:
        assert "code" in item.value["excluded_features"]
    assert not any(
        "Strong classification separability" in finding.title
        for finding in result.findings
    )


def test_numeric_columns_are_not_flagged_high_cardinality() -> None:
    frame = pd.DataFrame(
        {
            "y": [0, 1] * 100,
            "amount": [float(index) * 1.5 for index in range(200)],
            "tenure": [index % 70 for index in range(200)],
        }
    )

    result = pe.classification(frame, target="y", sampling="disabled")

    high_card_columns = {
        item.scope.columns[0]
        for item in result.evidence
        if item.kind == "classification_high_cardinality_feature"
    }
    assert "amount" not in high_card_columns
    assert "tenure" not in high_card_columns


def test_identifier_feature_is_flagged_and_excluded_from_probe() -> None:
    frame = pd.DataFrame(
        {
            "y": [0, 1] * 50,
            "user_id": list(range(100)),
            "x": [0.0, 1.0] * 50,
        }
    )

    result = pe.classification(frame, target="y", sampling="disabled")

    identifiers = {
        item.scope.columns[0]
        for item in result.evidence
        if item.kind == "classification_identifier_feature"
    }
    assert "user_id" in identifiers
    assert any(
        "Identifier-like feature" in finding.title for finding in result.findings
    )
    probe = [
        item for item in result.evidence if item.kind == "classification_probe_model"
    ]
    for item in probe:
        assert "user_id" in item.value["excluded_features"]


def test_findings_are_sorted_by_severity() -> None:
    target = [0] * 90 + [1] * 10
    frame = pd.DataFrame({"y": target, "leak": target, "x": list(range(100))})

    result = pe.classification(frame, target="y", sampling="disabled")

    ranks = [SEVERITY_RANK.get(finding.severity, 99) for finding in result.findings]
    assert ranks == sorted(ranks)
    # Leakage (critical) must lead the report.
    assert "leakage" in result.findings[0].title.lower()


# --------------------------------------------------------------------------- #
# Anomaly noise reduction
# --------------------------------------------------------------------------- #
def test_univariate_tail_finding_requires_meaningful_signal() -> None:
    frame = pd.DataFrame(
        {
            # A small, ordinary tail: a couple of values just past the IQR
            # fence with modest robust-z. Evidence yes, finding no.
            "mild": list(range(100)) + [200, -100],
            # One genuinely extreme value: must become a finding.
            "spike": list(range(101)) + [99999],
        }
    )

    result = pe.anomaly_detection(frame, sampling="disabled")

    univariate = {
        item.scope.columns[0]: item
        for item in result.evidence
        if item.kind == "anomaly_univariate_outlier"
    }
    assert univariate["mild"].value["candidate_count"] >= 1
    titles = [finding.title for finding in result.findings]
    assert any("Univariate tail candidates" in t and "spike" in t for t in titles)
    assert not any("Univariate tail candidates" in t and "mild" in t for t in titles)


def test_conditional_anomaly_findings_are_capped() -> None:
    rng = np.random.default_rng(1)
    size = 300
    frame = pd.DataFrame(
        {
            "a": rng.normal(50, 5, size),
            "b": rng.normal(10, 2, size),
            "c": rng.normal(0, 1, size),
        }
    )

    result = pe.anomaly_detection(frame, sampling="disabled")

    conditional_findings = [
        finding
        for finding in result.findings
        if "Conditional anomaly candidates" in finding.title
    ]
    assert len(conditional_findings) <= _MAX_CONDITIONAL_FINDINGS


# --------------------------------------------------------------------------- #
# Row-centric reporting: show the data behind the claim
# --------------------------------------------------------------------------- #
def test_consensus_review_lists_rows_with_values_and_plain_why() -> None:
    """The headline must name the rows, their values, the baseline, and the why."""
    rng = np.random.default_rng(0)
    salary = list(rng.integers(30_000, 90_000, size=40)) + [5_000_000, 7_000_000]
    tenure = list(rng.integers(1, 10, size=40)) + [22, 25]
    frame = pd.DataFrame(
        {"id": range(len(salary)), "salary": salary, "tenure": tenure}
    )

    result = pe.anomaly_detection(frame, sampling="disabled")

    review = [e for e in result.evidence if e.kind == "anomaly_consensus_review"]
    assert review
    value = review[0].value
    assert value["id_column"] == "id"
    assert value["review_row_count"] >= 1
    row = value["rows"][0]
    assert row["values"]  # the actual row is carried, not just an index
    assert "salary" in {c["column"] for c in row["contributors"]}
    assert "typical" in row["why"]  # plain-language baseline, not just a z-score
    assert any("to review" in finding.title for finding in result.findings)


def test_review_rows_carry_per_method_explanations() -> None:
    """A multivariate/conditional tag must be backed by visible evidence.

    The report's whole point is that a row flagged "multivariate outlier" shows
    the joint per-column profile, and a "conditional outlier" shows the peer
    group it stands out from — not a single univariate sigma bar.
    """
    rng = np.random.default_rng(11)
    size = 120
    experience = rng.integers(1, 20, size=size)
    salary = 200_000 + experience * 8_000 + rng.integers(-20_000, 20_000, size=size)
    frame = pd.DataFrame(
        {
            "id": list(range(1, size + 1)),
            "experience": experience.astype(int),
            "salary": salary.astype(int),
        }
    )
    extras = [(901, 6, 6_800_000), (902, 16, 9_300_000), (903, 12, 7_600_000)]
    frame = pd.concat(
        [frame, pd.DataFrame(extras, columns=["id", "experience", "salary"])],
        ignore_index=True,
    )

    result = pe.anomaly_detection(frame, sampling="disabled")
    review = [e for e in result.evidence if e.kind == "anomaly_consensus_review"]
    assert review
    rows = review[0].value["rows"]
    assert rows and all("explanations" in row for row in rows)

    multivariate_rows = [r for r in rows if "multivariate outlier" in r["methods"]]
    assert multivariate_rows
    multivariate = multivariate_rows[0]["explanations"]["multivariate"]
    assert "score" in multivariate and "threshold" in multivariate
    # The full profile spans every numeric column, not just the dominant spike —
    # that is what lets the analyst see the joint (multivariate) picture.
    assert len(multivariate["columns"]) >= 2
    assert {"column", "robust_z", "value", "baseline"} <= set(
        multivariate["columns"][0]
    )

    conditional_rows = [r for r in rows if "conditional outlier" in r["methods"]]
    assert conditional_rows
    conditional = conditional_rows[0]["explanations"]["conditional"]
    assert {
        "condition_column",
        "value_column",
        "peer_q1",
        "peer_q3",
        "peer_median",
        "value",
    } <= set(conditional)


def test_verdict_headline_is_synthesized() -> None:
    """The report metadata carries a plain-language verdict to lead the hero."""
    low = [float(value) for value in np.linspace(10, 30, 25)]
    high = [float(value) for value in np.linspace(500, 520, 15)]
    frame = pd.DataFrame({"salary": low + high, "y": list(range(40))})

    result = pe.anomaly_detection(frame, sampling="disabled")

    verdict = result.metadata.get("verdict")
    assert isinstance(verdict, str) and verdict
    # A clean two-population split is the strongest reframing, so it must lead.
    assert "two distinct populations" in verdict


def test_bimodal_column_is_reported_as_two_populations() -> None:
    low = [float(value) for value in np.linspace(10, 30, 25)]
    high = [float(value) for value in np.linspace(500, 520, 15)]
    frame = pd.DataFrame({"x": low + high, "y": list(range(40))})

    result = pe.anomaly_detection(frame, sampling="disabled")

    shapes = {
        item.value["column"]: item
        for item in result.evidence
        if item.kind == "anomaly_distribution_shape"
    }
    assert shapes["x"].value["modality"]["is_multimodal"]
    assert not shapes["y"].value["modality"]["is_multimodal"]
    assert any("two populations" in finding.title for finding in result.findings)


def test_consensus_is_quiet_on_clean_data() -> None:
    """Clean, unimodal data must not manufacture a review list or a regime split."""
    rng = np.random.default_rng(7)
    size = 200
    x = rng.normal(0.0, 1.0, size)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": x * 2 + rng.normal(0.0, 0.05, size),
            "z": rng.normal(5.0, 1.0, size),
        }
    )

    result = pe.anomaly_detection(frame, sampling="disabled")

    review = [e for e in result.evidence if e.kind == "anomaly_consensus_review"]
    review_rows = review[0].value["review_row_count"] if review else 0
    assert review_rows <= 2  # at most the genuinely extreme tail, never a crowd
    assert not any("two populations" in finding.title for finding in result.findings)


def test_distribution_finding_summary_hides_raw_boundary_values() -> None:
    """Privacy boundary: exact cell values stay in evidence (local report) and must
    not leak into a finding summary, which the assisted investigator forwards to
    the LLM."""
    low = [float(value) for value in np.linspace(10, 30, 25)]
    high = [1_234_567.0 + index for index in range(15)]
    frame = pd.DataFrame({"amount": low + high, "row": list(range(40))})

    result = pe.anomaly_detection(frame, sampling="disabled")

    bimodal = [f for f in result.findings if "two populations" in f.title]
    assert bimodal
    assert "1,234,567" not in bimodal[0].summary
    assert "1234567" not in bimodal[0].summary
    shape = next(
        item
        for item in result.evidence
        if item.kind == "anomaly_distribution_shape"
        and item.value["column"] == "amount"
    )
    assert shape.value["modality"]["clusters"][1]["min"] >= 1_234_567


# --------------------------------------------------------------------------- #
# Schema noise reduction
# --------------------------------------------------------------------------- #
def test_spurious_one_to_one_from_range_overlap_is_suppressed() -> None:
    # customer_id (0..49) is fully contained in order_id (0..199) purely by
    # range overlap; both are unique, so naive inclusion would call it 1:1.
    customers = pd.DataFrame({"customer_id": range(50), "name": list("x" * 50)})
    orders = pd.DataFrame(
        {"order_id": range(200), "customer_id": [index % 50 for index in range(200)]}
    )

    result = pe.discover_schema({"customers": customers, "orders": orders})

    one_to_one = [
        item
        for item in result.evidence
        if item.kind == "candidate_relationship"
        and item.value["cardinality"] == "one_to_one"
        and {item.value["parent_columns"][0], item.value["child_columns"][0]}
        == {"customer_id", "order_id"}
    ]
    assert not one_to_one


def test_relationship_finding_titles_name_the_tables() -> None:
    customers = pd.DataFrame({"customer_id": range(50)})
    orders = pd.DataFrame({"customer_id": [index % 50 for index in range(200)]})

    result = pe.discover_schema({"customers": customers, "orders": orders})

    relationship_findings = [
        finding for finding in result.findings if finding.title.startswith("Candidate")
    ]
    assert relationship_findings
    assert all("→" in finding.title for finding in relationship_findings)
