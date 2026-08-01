"""Associations, observations, interactions, missing structure, and samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prism_eda as pe
from prism_eda.analysis._limits import ProfileLimits
from prism_eda.analysis.associations import (
    _correlation_ratio,
    _cramers_v,
    build_observations,
)
from prism_eda.evidence.models import OBSERVATION, split_findings


def _observations(frame: pd.DataFrame, detail: str = "standard"):
    table = pe.load(frame).catalog().tables[0]
    column_evidence = {column.name: f"ev_{column.name}" for column in table.columns}
    return build_observations(
        frame,
        table,
        column_evidence,
        limits=ProfileLimits.for_detail(detail),
        seed=42,
    )


def _by_kind(evidence) -> dict[str, list]:
    found: dict[str, list] = {}
    for item in evidence:
        found.setdefault(item.kind, []).append(item)
    return found


def test_cramers_v_is_one_for_a_perfect_mapping_and_low_for_independence() -> None:
    left = pd.Series(["a", "b", "c"] * 20)
    perfect = left.map({"a": "x", "b": "y", "c": "z"})
    assert _cramers_v(left, perfect) == pytest.approx(1.0, abs=1e-6)

    rng = np.random.default_rng(4)
    independent = pd.Series(rng.choice(["x", "y", "z"], size=60))
    value = _cramers_v(left, independent)
    assert value is not None and value < 0.4


def test_correlation_ratio_measures_variance_explained_by_group() -> None:
    groups = pd.Series(["a"] * 30 + ["b"] * 30)
    separated = pd.Series([0.0] * 30 + [10.0] * 30)
    assert _correlation_ratio(groups, separated) == pytest.approx(1.0, abs=1e-6)

    rng = np.random.default_rng(5)
    unrelated = pd.Series(rng.normal(size=60))
    value = _correlation_ratio(groups, unrelated)
    assert value is not None and value < 0.5


def test_each_pair_records_the_statistic_used() -> None:
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {
            "height": rng.normal(170, 10, 120),
            "weight": rng.normal(70, 8, 120),
            "region": rng.choice(["north", "south"], 120),
            "tier": rng.choice(["gold", "silver"], 120),
        }
    )

    matrix = _by_kind(_observations(frame)[0])["profile_association_matrix"][0]

    methods = {
        (pair["left"], pair["right"]): pair["method"] for pair in matrix.value["pairs"]
    }
    assert methods[("height", "weight")] == "spearman"
    assert methods[("region", "tier")] == "cramers_v"
    assert methods[("height", "region")] == "correlation_ratio"
    assert all(0.0 <= pair["strength"] <= 1.0 for pair in matrix.value["pairs"])


def test_strong_pairs_become_observations_not_issues() -> None:
    base = np.arange(100, dtype=float)
    frame = pd.DataFrame({"a": base, "b": base * 2 + 1, "noise": base % 7})

    _, findings, _, _ = _observations(frame)

    strong = [item for item in findings if "strongly associated" in item.title]
    assert strong
    for finding in strong:
        assert finding.category == OBSERVATION
        assert finding.severity == "info"
    assert "effectively one variable" in strong[0].summary


def test_weak_associations_are_not_reported() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})

    _, findings, _, _ = _observations(frame)

    assert not [item for item in findings if "strongly associated" in item.title]


def test_identity_cardinality_and_dominance_observations() -> None:
    frame = pd.DataFrame(
        {
            "row_id": range(200),
            "wide": [f"label_{index}" for index in range(200)],
            "flag": ["yes"] * 195 + ["no"] * 5,
        }
    )

    _, findings, _, _ = _observations(frame)
    titles = " | ".join(item.title for item in findings)

    assert "has all-unique values" in titles
    assert "is dominated by one label" in titles


def test_time_coverage_gaps_and_ordering_are_observed() -> None:
    stamps = list(pd.date_range("2024-01-01", periods=20, freq="D"))
    stamps += list(pd.date_range("2024-06-01", periods=20, freq="D"))
    frame = pd.DataFrame({"seen_at": stamps[::-1]})

    evidence, findings, _, _ = _observations(frame)
    coverage = _by_kind(evidence)["profile_time_coverage"][0]
    titles = " | ".join(item.title for item in findings)

    assert coverage.value["gap_count"] == 1
    assert coverage.value["rows_in_file_order_are_sorted"] is False
    assert "Gaps in" in titles
    assert "is not in time order" in titles


def test_scatter_evidence_splits_highlights_from_the_explorer() -> None:
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({f"n{index}": rng.normal(size=80) for index in range(5)})

    scatters = _by_kind(_observations(frame)[0])["profile_scatter"]

    roles = [item.value["role"] for item in scatters]
    assert roles.count("highlight") == 6  # the standard highlight budget
    assert "explorer" in roles
    # Explorer scatters carry fewer points on purpose: dozens are embedded.
    limits = ProfileLimits.for_detail("standard")
    for item in scatters:
        budget = (
            limits.scatter_points
            if item.value["role"] == "highlight"
            else limits.explorer_points
        )
        assert item.value["point_count"] <= budget
        assert set(item.value["points"][0]) == {"x", "y", "flagged"}


def test_scatter_matches_the_shape_the_renderer_consumes() -> None:
    from prism_eda.reporting.charts import scatter_svg

    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})

    scatter = _by_kind(_observations(frame)[0])["profile_scatter"][0]

    assert "<svg" in scatter_svg(scatter.value)


def test_co_missingness_finds_columns_that_go_missing_together() -> None:
    values = [1.0] * 30 + [None] * 10
    frame = pd.DataFrame(
        {
            "always_together_a": values,
            "always_together_b": values,
            "independent": [None] * 10 + [2.0] * 30,
        }
    )

    evidence = _by_kind(_observations(frame)[0])
    pairs = evidence["profile_co_missingness"][0].value["pairs"]

    together = next(
        pair
        for pair in pairs
        if {pair["left"], pair["right"]} == {"always_together_a", "always_together_b"}
    )
    assert together["always_together"] is True
    assert together["both_missing"] == 10


def test_sample_rows_carry_head_tail_and_duplicate_groups() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2, 3, 4] * 6, "b": ["x", "x", "y", "z", "w"] * 6})

    evidence = _by_kind(_observations(frame)[0])
    sample = evidence["profile_sample_rows"][0].value

    assert len(sample["head"]) == 10
    assert len(sample["tail"]) == 10
    assert sample["columns"] == ["a", "b"]
    assert evidence["profile_duplicate_groups"][0].value["groups"]


def test_sample_columns_are_capped_and_the_cut_is_recorded() -> None:
    frame = pd.DataFrame({f"c{index}": [1, 2, 3] for index in range(40)})

    sample = _by_kind(_observations(frame)[0])["profile_sample_rows"][0].value

    assert len(sample["columns"]) == 30
    assert sample["hidden_column_count"] == 10


def test_large_tables_are_sampled_for_associations_and_say_so() -> None:
    rng = np.random.default_rng(8)
    size = 60_000
    frame = pd.DataFrame({"a": rng.normal(size=size), "b": rng.normal(size=size)})

    _, _, warnings, sampling = _observations(frame)

    assert any(warning.code == "sampled_associations" for warning in warnings)
    assert sampling[0].sampled_rows == 50_000
    assert sampling[0].operation == "profile_associations"


def test_profile_separates_issues_from_observations_end_to_end() -> None:
    base = np.arange(60, dtype=float)
    frame = pd.DataFrame(
        {
            "row_id": range(60),
            "a": base,
            "b": base * 3,
            "dirty": (["USA", "usa"] * 30),
        }
    )

    result = pe.profile(frame)
    issues, observations = split_findings(result.findings)

    assert any("Inconsistent value formatting" in item.title for item in issues)
    assert any("strongly associated" in item.title for item in observations)
    assert result.metadata["issue_count"] == len(issues)
    assert result.metadata["observation_count"] == len(observations)
    assert "observation(s)" in result.summary
