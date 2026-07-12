from __future__ import annotations

import pandas as pd

import prism_eda as pe
from prism_eda.events import EventKind


def test_profile_is_non_mutating_and_evidence_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "person_id": [1, 2, 2, 4],
            "age": [5, 32, 32, 60],
            "weight": [100.0, 75.0, 75.0, None],
            "constant": ["x", "x", "x", "x"],
        }
    )
    original = frame.copy(deep=True)
    dataset = pe.load(frame)

    first = dataset.profile()
    second = dataset.profile()

    pd.testing.assert_frame_equal(frame, original)
    assert first.status == pe.AnalysisStatus.COMPLETED
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    assert first.catalog.fingerprint == second.catalog.fingerprint
    assert any("Duplicate rows" in finding.title for finding in first.findings)
    duplicate_finding = next(
        finding for finding in first.findings if "Duplicate rows" in finding.title
    )
    assert duplicate_finding.summary == "1 row (25.0%) is an exact duplicate."
    assert any("Constant column" in finding.title for finding in first.findings)
    evidence_ids = {item.id for item in first.evidence}
    assert all(set(finding.evidence_ids) <= evidence_ids for finding in first.findings)
    assert not first.transformation_plan.is_empty


def test_empty_dataset_returns_insufficient_evidence() -> None:
    result = pe.profile(pd.DataFrame(columns=["id", "value"]))

    assert result.status == pe.AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert result.warnings[0].code == "insufficient_rows"


def test_best_effort_empty_dataset_is_visibly_qualified() -> None:
    result = pe.profile(pd.DataFrame(columns=["id"]), allow_insufficient_evidence=True)

    assert result.status == pe.AnalysisStatus.COMPLETED_WITH_WARNINGS
    assert result.warnings


def test_callbacks_receive_events_and_callback_errors_are_isolated() -> None:
    events = []

    def collect(event: pe.Event) -> None:
        events.append(event)

    def broken_callback(event: pe.Event) -> None:
        raise RuntimeError("observer failure")

    result = pe.profile(
        pd.DataFrame({"value": [1, 2, 3]}),
        callbacks=[broken_callback, collect],
    )

    assert result.status == pe.AnalysisStatus.COMPLETED
    assert events[0].kind == EventKind.RUN_STARTED
    assert events[-1].kind == EventKind.RUN_COMPLETED
    assert any(event.kind == EventKind.EVIDENCE_CREATED for event in events)


def test_catalog_infers_basic_semantic_roles() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "created_at": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "segment": ["a", "a", "b"],
        }
    )

    columns = {
        column.name: column for column in pe.load(frame).catalog().tables[0].columns
    }

    assert "identifier_candidate" in columns["customer_id"].roles
    assert columns["created_at"].semantic_type == "datetime"
    assert "timestamp_candidate" in columns["created_at"].roles
    assert columns["segment"].semantic_type == "categorical"


def _string_column(unique: int, repeats: int) -> list[str]:
    return [f"value_{index:04d}" for index in range(unique)] * repeats


def test_high_cardinality_text_is_not_categorical() -> None:
    # 250 uniques over 6,000 rows: ratio 4.2% passes the 5% rule, but the
    # absolute cap must demote the column to text.
    frame = pd.DataFrame({"agent_name": _string_column(250, 24)})

    agent = pe.load(frame).catalog().tables[0].columns[0]

    assert agent.semantic_type == "text"
    assert "free_text_candidate" in agent.roles
    assert "min_length" in agent.statistics


def test_categorical_cap_boundaries() -> None:
    at_cap = pd.DataFrame({"code": _string_column(200, 30)})
    above_cap = pd.DataFrame({"code": _string_column(201, 30)})
    small_high_ratio = pd.DataFrame({"code": _string_column(40, 2)})

    at_cap_col = pe.load(at_cap).catalog().tables[0].columns[0]
    above_cap_col = pe.load(above_cap).catalog().tables[0].columns[0]
    small_col = pe.load(small_high_ratio).catalog().tables[0].columns[0]

    assert at_cap_col.semantic_type == "categorical"
    assert above_cap_col.semantic_type == "text"
    # <= 50 uniques stays categorical regardless of unique ratio (here 50%).
    assert small_col.semantic_type == "categorical"


def test_high_cardinality_categorical_gets_warning() -> None:
    frame = pd.DataFrame(
        {
            "many_codes": _string_column(150, 40),
            "few_codes": _string_column(30, 200),
        }
    )

    columns = {
        column.name: column for column in pe.load(frame).catalog().tables[0].columns
    }
    assert columns["many_codes"].semantic_type == "categorical"
    assert any(
        "High cardinality" in warning for warning in columns["many_codes"].warnings
    )
    assert not any(
        "High cardinality" in warning for warning in columns["few_codes"].warnings
    )


def test_explicit_categorical_dtype_kept_but_warned() -> None:
    frame = pd.DataFrame({"declared": pd.Categorical(_string_column(250, 4))})

    column = pe.load(frame).catalog().tables[0].columns[0]

    assert column.semantic_type == "categorical"
    assert any("High cardinality" in warning for warning in column.warnings)
