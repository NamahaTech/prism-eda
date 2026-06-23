"""Edge-case inputs must not crash the recipes."""

from __future__ import annotations

import numpy as np
import pandas as pd

import prism_eda as pe


def test_profile_handles_all_null_column() -> None:
    frame = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
    result = pe.profile(frame)
    assert result.status in {
        pe.AnalysisStatus.COMPLETED,
        pe.AnalysisStatus.COMPLETED_WITH_WARNINGS,
    }


def test_profile_handles_single_row() -> None:
    result = pe.profile(pd.DataFrame({"a": [1], "b": ["x"]}))
    assert result.status in {
        pe.AnalysisStatus.COMPLETED,
        pe.AnalysisStatus.COMPLETED_WITH_WARNINGS,
    }


def test_profile_handles_mixed_dtype_object_column() -> None:
    frame = pd.DataFrame({"mixed": [1, "two", 3.0, None, True], "n": range(5)})
    result = pe.profile(frame)
    assert result.status in {
        pe.AnalysisStatus.COMPLETED,
        pe.AnalysisStatus.COMPLETED_WITH_WARNINGS,
    }


def test_classification_single_class_is_insufficient_evidence() -> None:
    frame = pd.DataFrame({"y": [1, 1, 1, 1], "x": [1, 2, 3, 4]})
    result = pe.classification(frame, target="y", sampling="disabled")
    assert result.status == pe.AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_anomaly_detection_handles_all_null_numeric_column() -> None:
    frame = pd.DataFrame(
        {
            "all_null": [np.nan] * 30,
            "x": list(range(30)),
            "y": list(range(30)),
        }
    )
    result = pe.anomaly_detection(frame, sampling="disabled")
    assert result.status in {
        pe.AnalysisStatus.COMPLETED,
        pe.AnalysisStatus.COMPLETED_WITH_WARNINGS,
        pe.AnalysisStatus.NO_MEANINGFUL_STRUCTURE,
    }


def test_classification_export_roundtrip_is_json_serializable() -> None:
    import json

    target = [0] * 90 + [1] * 10
    frame = pd.DataFrame({"y": target, "leak": target, "x": list(range(100))})
    result = pe.classification(frame, target="y", sampling="disabled")
    payload = json.loads(json.dumps(result.to_dict(), sort_keys=True))
    assert payload["status"] in {"completed", "completed_with_warnings"}
    finding_severities = {finding["severity"] for finding in payload["findings"]}
    assert "critical" in finding_severities  # exact-copy leak escalates to critical
