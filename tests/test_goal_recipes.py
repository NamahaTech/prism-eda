from __future__ import annotations

import json

import pandas as pd

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
