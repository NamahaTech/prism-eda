"""Distribution shape labels and family fitting, checked against known draws."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prism_eda as pe
from prism_eda.analysis.distributions import (
    category_frequency_evidence,
    datetime_timeline_evidence,
    numeric_distribution_evidence,
)


def _distribution(values: np.ndarray | list[float], column: str = "value") -> dict:
    evidence = numeric_distribution_evidence(
        pd.Series(values), table="t", column=column, fit_rows=20_000
    )
    assert evidence is not None
    return evidence.value


@pytest.mark.parametrize(
    ("family", "draw"),
    [
        ("norm", lambda rng: rng.normal(50, 10, 800)),
        ("lognorm", lambda rng: rng.lognormal(1.0, 0.6, 800)),
        ("expon", lambda rng: rng.exponential(3.0, 800)),
        ("uniform", lambda rng: rng.uniform(0, 100, 800)),
        ("poisson", lambda rng: rng.poisson(4.0, 800).astype(float)),
    ],
)
def test_known_draws_recover_their_own_family(family, draw) -> None:
    values = draw(np.random.default_rng(7))

    fit = _distribution(values)["fit"]

    assert fit["family"] == family
    assert fit["quality"] in {"close", "approximate"}
    # A distance is reported; a p-value deliberately is not.
    assert "ks_distance" in fit
    assert "p_value" not in fit


def test_fitting_abstains_rather_than_naming_the_least_bad_family() -> None:
    rng = np.random.default_rng(7)
    bimodal = np.concatenate([rng.normal(10, 1, 400), rng.normal(60, 1, 400)])

    fit = _distribution(bimodal)["fit"]

    assert fit["family"] is None
    assert fit["reason"] == "no_family_fits_well"
    # The ranked candidates are still available for the analyst to judge.
    assert fit["ranked"]


def test_fitting_does_not_run_on_too_few_rows() -> None:
    fit = _distribution([float(value) for value in range(12)])["fit"]

    assert fit["family"] is None
    assert fit["reason"] == "too_few_values"


def test_shape_labels_describe_what_the_analyst_would_see() -> None:
    rng = np.random.default_rng(11)

    assert _distribution(rng.normal(0, 1, 600))["shape"]["label"] == "bell-shaped"
    assert _distribution(rng.uniform(0, 1, 600))["shape"]["label"] == "uniform"
    assert _distribution(rng.exponential(2, 600))["shape"]["label"] == "right-skewed"

    bimodal = np.concatenate([rng.normal(0, 1, 300), rng.normal(40, 1, 300)])
    assert _distribution(bimodal)["shape"]["label"] == "bimodal"

    assert _distribution([3.0] * 60)["shape"]["label"] == "constant"


def test_zero_inflation_and_count_like_are_called_out() -> None:
    rng = np.random.default_rng(3)
    values = np.concatenate([np.zeros(300), rng.exponential(5, 300)])

    descriptors = " ".join(_distribution(values)["shape"]["descriptors"])

    assert "zero-inflated" in descriptors

    counts = rng.poisson(3, 400).astype(float)
    assert "count-like" in " ".join(_distribution(counts)["shape"]["descriptors"])


def test_histogram_matches_the_shape_the_renderer_consumes() -> None:
    from prism_eda.reporting.charts import histogram_svg

    value = _distribution(np.random.default_rng(5).normal(0, 1, 400))

    assert len(value["histogram"]["edges"]) == len(value["histogram"]["counts"]) + 1
    assert value["flagged_values"] == []
    assert "<svg" in histogram_svg(value)


def test_infinities_are_excluded_from_the_summary_not_hidden() -> None:
    values = [1.0, 2.0, 3.0, float("inf"), 5.0, 6.0, 7.0, 8.0, 9.0]

    box = _distribution(values)["box"]

    assert box["max"] == 9.0
    assert np.isfinite(box["mean"])


def test_category_frequency_buckets_the_long_tail() -> None:
    labels = [f"cat_{index}" for index in range(40)] * 2 + ["cat_0"] * 20
    evidence = category_frequency_evidence(pd.Series(labels), table="t", column="label")

    assert evidence is not None
    assert evidence.value["distinct_count"] == 40
    assert len(evidence.value["counts"]) == 15
    assert evidence.value["other_count"] > 0
    assert evidence.value["other_category_count"] == 25


def test_timeline_granularity_follows_the_covered_range() -> None:
    daily = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    yearly = pd.Series(pd.date_range("1990-01-01", periods=30, freq="YS"))

    day_evidence = datetime_timeline_evidence(daily, table="t", column="at")
    year_evidence = datetime_timeline_evidence(yearly, table="t", column="at")

    assert day_evidence is not None and day_evidence.value["granularity"] == "day"
    assert year_evidence is not None and year_evidence.value["granularity"] == "year"


def test_profile_banks_chart_evidence_and_reports_the_chart_cap() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({f"c{index}": rng.normal(size=40) for index in range(70)})

    result = pe.profile(frame)

    kinds = [item.kind for item in result.evidence]
    assert kinds.count("profile_distribution") == 60
    assert any(warning.code == "chart_columns_capped" for warning in result.warnings)

    full = pe.profile(frame, detail="full")
    assert [item.kind for item in full.evidence].count("profile_distribution") == 70
    assert not [
        warning for warning in full.warnings if warning.code == "chart_columns_capped"
    ]


def test_detail_must_be_a_known_level() -> None:
    with pytest.raises(ValueError, match="detail must be"):
        pe.profile(pd.DataFrame({"a": [1, 2, 3]}), detail="everything")
