from __future__ import annotations

import json

import pandas as pd
import pytest

import prism_eda as pe


def test_anomaly_detection_finds_conditional_numeric_candidate(tmp_path) -> None:
    ages = list(range(10, 70)) + [5]
    weights = [20 + age * 1.1 + (index % 3) for index, age in enumerate(ages[:-1])]
    weights.append(100.0)
    frame = pd.DataFrame(
        {
            "age": ages,
            "weight": weights,
            "segment": ["normal"] * 60 + ["rare"],
        }
    )

    result = pe.anomaly_detection(frame, sampling="disabled")

    assert result.status == pe.AnalysisStatus.COMPLETED
    conditional = [
        item for item in result.evidence if item.kind == "anomaly_conditional_outlier"
    ]
    assert conditional
    assert any(
        "Conditional anomaly candidates" in finding.title for finding in result.findings
    )
    assert any(
        artifact.title == "Anomaly candidate signals" for artifact in result.artifacts
    )

    html = result.to_html(tmp_path / "anomaly.html").read_text(encoding="utf-8")
    payload = json.loads(result.to_json(tmp_path / "anomaly.json").read_text())
    assert "Anomaly candidate review" in html
    assert "Anomaly candidate signals" in html
    assert any(
        item["kind"] == "anomaly_conditional_outlier" for item in payload["evidence"]
    )


def test_anomaly_detection_runs_model_backed_ranked_detectors() -> None:
    normal_rows = 120
    frame = pd.DataFrame(
        {
            "x": [index / 20 for index in range(normal_rows)] + [12.0, 13.0, 14.0],
            "y": [index / 22 for index in range(normal_rows)] + [11.5, 12.5, 13.5],
            "segment": ["expected"] * normal_rows + ["review"] * 3,
        }
    )

    result = pe.anomaly_detection(
        frame,
        sampling="disabled",
        expected_contamination=0.03,
    )

    kinds = {item.kind for item in result.evidence}
    assert "anomaly_isolation_forest" in kinds
    assert "anomaly_local_density_outlier" in kinds
    assert "anomaly_detector_agreement" in kinds
    isolation = next(
        item for item in result.evidence if item.kind == "anomaly_isolation_forest"
    )
    assert isolation.value["expected_contamination"] == 0.03
    assert isolation.value["stability"]["seed_count"] == 3
    agreement = next(
        item for item in result.evidence if item.kind == "anomaly_detector_agreement"
    )
    assert agreement.value["candidate_count"] >= 1
    assert all(finding.evidence_ids for finding in result.findings)
    assert result.metadata["expected_contamination"] == 0.03


def test_anomaly_detection_finds_rare_category_combinations() -> None:
    frame = pd.DataFrame(
        {
            "region": ["north"] * 80 + ["south"] * 20 + ["antarctica"],
            "channel": ["web"] * 50 + ["store"] * 50 + ["satellite"],
            "amount": list(range(101)),
        }
    )

    result = pe.anomaly_detection(frame, sampling="disabled")

    rare_pairs = [
        item
        for item in result.evidence
        if item.kind == "anomaly_rare_category_combination"
    ]
    assert rare_pairs
    assert any(
        "Rare category combinations" in finding.title for finding in result.findings
    )


def test_anomaly_detection_rejects_invalid_expected_contamination() -> None:
    with pytest.raises(ValueError, match="expected_contamination"):
        pe.anomaly_detection(
            pd.DataFrame({"x": range(30), "y": range(30)}),
            expected_contamination=0.8,
        )


def test_classification_reports_imbalance_leakage_and_artifacts(tmp_path) -> None:
    labels = [0] * 95 + [1] * 5
    frame = pd.DataFrame(
        {
            "target": labels,
            "leak_target": labels,
            "score": [0.1] * 95 + [0.95] * 5,
            "segment": ["common"] * 90 + ["rare"] * 10,
        }
    )

    result = pe.classification(frame, target="target", sampling="disabled")

    assert result.status == pe.AnalysisStatus.COMPLETED
    assert any(item.kind == "classification_target_summary" for item in result.evidence)
    assert any(
        item.kind == "classification_leakage_candidate" for item in result.evidence
    )
    assert any("Class imbalance" in finding.title for finding in result.findings)
    assert any(
        "Potential target leakage" in finding.title for finding in result.findings
    )
    assert {artifact.title for artifact in result.artifacts} >= {
        "Class balance",
        "Feature-target diagnostic signals",
    }

    html = result.to_html(tmp_path / "classification.html").read_text(encoding="utf-8")
    assert "Classification readiness map" in html
    assert "Class balance" in html


def test_classification_probe_reports_separability_and_excludes_leakage() -> None:
    labels = [0] * 40 + [1] * 40
    frame = pd.DataFrame(
        {
            "target": labels,
            "leak_target": labels,
            "signal": [index / 40 for index in range(40)]
            + [4 + index / 40 for index in range(40)],
            "segment": ["low"] * 40 + ["high"] * 40,
        }
    )

    result = pe.classification(frame, target="target", sampling="disabled")

    probe = next(
        item for item in result.evidence if item.kind == "classification_probe_model"
    )
    assert probe.value["balanced_accuracy"] >= 0.85
    assert "leak_target" in probe.value["excluded_features"]
    assert any(
        "Strong classification separability" in finding.title
        for finding in result.findings
    )
    assert any(
        row.get("signal") == "probe separability"
        for artifact in result.artifacts
        for row in artifact.data.get("rows", ())
    )


def test_classification_probe_surfaces_hard_examples() -> None:
    labels = [index % 2 for index in range(80)]
    frame = pd.DataFrame(
        {
            "target": labels,
            "weak_signal": [index / 10 for index in range(80)],
            "group": ["a" if index < 40 else "b" for index in range(80)],
        }
    )

    result = pe.classification(frame, target="target", sampling="disabled")

    hard_examples = [
        item for item in result.evidence if item.kind == "classification_hard_examples"
    ]
    assert hard_examples
    assert hard_examples[0].value["examples"]
    assert any("Probe hard examples" in finding.title for finding in result.findings)


def test_classification_uses_context_for_split_guidance() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 40,
            "customer_id": [index // 2 for index in range(80)],
            "observed_at": pd.date_range("2025-01-01", periods=80, freq="D"),
            "signal": list(range(80)),
        }
    )

    result = pe.classification(
        frame,
        target="target",
        context={"entity_id": "customer_id", "timestamp": "observed_at"},
        sampling="disabled",
    )

    split_guidance = [
        item for item in result.evidence if item.kind == "classification_split_guidance"
    ]
    assert split_guidance
    risk_kinds = {risk["kind"] for risk in split_guidance[0].value["risks"]}
    assert "group_split_recommended" in risk_kinds
    assert "time_split_recommended" in risk_kinds
    assert any("validation recommended" in finding.title for finding in result.findings)


def test_classification_requires_target_column() -> None:
    result = pe.classification(pd.DataFrame({"feature": [1, 2, 3]}))

    assert result.status == pe.AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert any(
        warning.code == "classification_target_required" for warning in result.warnings
    )


def test_dataset_analyze_dispatches_new_goals() -> None:
    dataset = pe.load(
        pd.DataFrame(
            {
                "target": ["no", "no", "yes", "no"],
                "feature": [1.0, 1.2, 9.0, 1.1],
            }
        )
    )

    anomaly = dataset.analyze("anomaly_detection", sampling="disabled")
    classification = dataset.analyze(
        "classification", target="target", sampling="disabled"
    )

    assert anomaly.goal == "anomaly_detection"
    assert classification.goal == "classification"
