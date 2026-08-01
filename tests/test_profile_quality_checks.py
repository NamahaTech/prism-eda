"""Data-quality detectors: each fixture carries one known, deliberate defect."""

from __future__ import annotations

import pandas as pd
import pytest

import prism_eda as pe
from prism_eda.analysis.quality_checks import detect_quality_issues
from prism_eda.evidence.models import OBSERVATION, QUALITY_ISSUE, split_findings


def _issues(frame: pd.DataFrame, **kwargs: object) -> dict[str, list]:
    """Run the detectors over a one-table frame, keyed by evidence kind."""
    table = pe.load(frame).catalog().tables[0]
    found: dict[str, list] = {}
    for issue in detect_quality_issues(frame, table, **kwargs):  # type: ignore[arg-type]
        found.setdefault(issue.kind, []).append(issue)
    return found


def test_clean_data_raises_nothing() -> None:
    frame = pd.DataFrame(
        {
            "id": range(1, 21),
            "region": ["north", "south"] * 10,
            "amount": [float(value) for value in range(1, 21)],
            "seen_at": pd.to_datetime(["2024-01-01"] * 20),
        }
    )

    assert detect_quality_issues(frame, pe.load(frame).catalog().tables[0]) == []


def test_case_and_whitespace_variants_are_one_finding() -> None:
    frame = pd.DataFrame({"country": ["USA", "usa", " USA ", "India", "India"]})

    issue = _issues(frame)["quality_inconsistent_formatting"][0]

    assert issue.value["variant_group_count"] == 1
    assert issue.value["affected_row_count"] == 3
    assert issue.value["padded_value_count"] == 1
    variants = {item["value"] for item in issue.value["examples"][0]["variants"]}
    assert variants == {"USA", "usa", " USA "}
    # The raw values are banked as evidence, never quoted in the summary that
    # the assisted-analysis layer forwards to a model.
    assert "USA" not in issue.summary


def test_placeholder_tokens_are_not_reported_as_case_variants() -> None:
    frame = pd.DataFrame({"country": ["N/A", "n/a", "NA", "India", "India"]})

    found = _issues(frame)

    assert "quality_inconsistent_formatting" not in found
    assert found["quality_disguised_missing"][0].value["hidden_missing_count"] == 3


def test_disguised_missing_reports_the_true_missing_rate() -> None:
    frame = pd.DataFrame({"note": ["unknown", "?", "real", "real", None]})

    issue = _issues(frame)["quality_disguised_missing"][0]

    assert issue.value["declared_missing_count"] == 1
    assert issue.value["hidden_missing_count"] == 2
    assert issue.value["effective_missing_rate"] == pytest.approx(0.6)


def test_numeric_sentinel_needs_repetition_and_a_non_negative_column() -> None:
    flagged = pd.DataFrame({"score": [-999.0, -999.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    assert "quality_disguised_missing" in _issues(flagged)

    # A single occurrence is an outlier, not a code.
    once = pd.DataFrame({"score": [-999.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert "quality_disguised_missing" not in _issues(once)

    # In a column that genuinely goes negative, -999 may be a real measurement.
    negatives = pd.DataFrame({"delta": [-999.0, -999.0, -5.0, 2.0, 3.0, 4.0, 5.0]})
    assert "quality_disguised_missing" not in _issues(negatives)


def test_numbers_stored_as_text_is_detected() -> None:
    frame = pd.DataFrame({"amount": ["10", "20", "30", "40", "50"]})

    issue = _issues(frame)["quality_numeric_as_text"][0]

    assert issue.value["parsed_rate"] == pytest.approx(1.0)


def test_mixed_date_layouts_beat_the_single_layout_finding() -> None:
    mixed = pd.DataFrame(
        {
            "signup": [
                "2024-01-01",
                "02/03/2024",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",
                "2024-01-09",
                "x",
            ]
        }
    )

    issue = _issues(mixed)["quality_date_text_format"][0]

    assert issue.severity == "high"
    assert len(issue.value["layouts"]) == 2
    assert issue.value["unmatched_count"] == 1

    single = pd.DataFrame(
        {
            "signup": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        }
    )
    only = _issues(single)["quality_date_text_format"][0]
    assert only.severity == "medium"
    assert only.title == "Dates stored as text"


def test_ambiguous_day_month_order_is_called_out() -> None:
    frame = pd.DataFrame(
        {
            "day": [
                "13/01/2024",
                "01/13/2024",
                "05/06/2024",
                "07/08/2024",
                "09/10/2024",
                "11/12/2024",
            ]
        }
    )

    issue = _issues(frame)["quality_date_text_format"][0]

    assert issue.value["ambiguous_day_month"] is True
    assert issue.severity == "high"


def test_mixed_python_types_are_detected() -> None:
    frame = pd.DataFrame({"mixed": [1, "2", 3, "4"]})

    issue = _issues(frame)["quality_mixed_types"][0]

    assert issue.value["minority_count"] == 2
    assert {item["type"] for item in issue.value["type_counts"]} == {"int", "str"}


def test_date_sentinels_future_and_implausible_dates() -> None:
    frame = pd.DataFrame(
        {
            "seen_at": pd.to_datetime(
                [
                    "1970-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2030-01-01",
                    "1500-01-01",
                ]
            )
        }
    )

    found = _issues(frame, now=pd.Timestamp("2025-01-01"))

    assert found["quality_date_sentinel"][0].value["affected_row_count"] == 1
    assert found["quality_future_date"][0].value["affected_row_count"] == 1
    assert found["quality_implausible_date"][0].value["affected_row_count"] == 1


def test_future_dates_do_not_change_the_evidence_id_by_the_hour() -> None:
    """The reference clock stays out of the evidence value, so the ID is stable."""
    frame = pd.DataFrame({"due_at": pd.to_datetime(["2030-01-01", "2024-01-01"])})

    morning = _issues(frame, now=pd.Timestamp("2025-01-01 09:00"))
    evening = _issues(frame, now=pd.Timestamp("2025-01-01 21:00"))

    assert (
        morning["quality_future_date"][0].value
        == evening["quality_future_date"][0].value
    )


def test_reversed_date_ranges_are_found_by_column_name() -> None:
    frame = pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "end_date": pd.to_datetime(["2023-01-01", "2024-02-01"]),
        }
    )

    issue = _issues(frame)["quality_reversed_date_range"][0]

    assert issue.value["affected_row_count"] == 1
    assert issue.columns == ("start_date", "end_date")


def test_structural_defects_and_alias_suppression() -> None:
    frame = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3],
            "country": ["USA", "usa", "India", "India"],
        }
    )
    frame["country_copy"] = frame["country"]

    found = _issues(frame)

    assert found["quality_unnamed_columns"][0].value["count"] == 1
    assert found["quality_aliased_columns"][0].value["columns"] == [
        "country",
        "country_copy",
    ]
    # The copy does not repeat its original's defect.
    formatting = found["quality_inconsistent_formatting"]
    assert [issue.columns for issue in formatting] == [("country",)]


def test_mangled_duplicate_headers_are_detected() -> None:
    frame = pd.DataFrame({"amount": [1, 2], "amount.1": [3, 4]})

    issue = _issues(frame)["quality_duplicate_headers"][0]

    assert issue.value["columns"] == [{"column": "amount.1", "base": "amount"}]


def test_profile_promotes_quality_issues_with_lineage() -> None:
    frame = pd.DataFrame(
        {"country": ["USA", "usa", "India", "India"], "value": [1, 2, 3, 4]}
    )

    result = pe.profile(frame)

    evidence_ids = {item.id for item in result.evidence}
    issues, _ = split_findings(result.findings)
    assert issues
    assert any("Inconsistent value formatting" in item.title for item in issues)
    for finding in result.findings:
        assert finding.category in {QUALITY_ISSUE, OBSERVATION}
        assert set(finding.evidence_ids) <= evidence_ids
    assert any(
        step.operation == "normalize_text_values"
        for step in result.transformation_plan.steps
    )


def test_a_failing_detector_does_not_abort_the_profile(monkeypatch) -> None:
    """Optional metrics may fail; the catalog-level profile still completes."""
    import prism_eda.analysis.profile as profile_module

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(profile_module, "build_quality_findings", explode)
    result = pe.profile(pd.DataFrame({"value": [1, 2, 3]}))

    assert result.status == pe.AnalysisStatus.COMPLETED_WITH_WARNINGS
    assert any("detector exploded" in failure.message for failure in result.failures)
