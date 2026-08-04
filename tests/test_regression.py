"""Regression readiness diagnostics.

The tests that matter most here are not the ones proving a detector fires. They
are the ones proving a detector stays *quiet*: a recipe that reports something
about every dataset is noise wearing a lab coat. So the first test asserts a
clean regression produces zero issues, and several others assert that a true
property of the data (skew, scale differences between groups) never gets filed
as a defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prism_eda as pe
from examples.sample_data import subscriptions
from prism_eda.evidence.models import OBSERVATION, split_findings
from prism_eda.results import AnalysisStatus


def _clean_frame(rows: int = 400, seed: int = 3) -> pd.DataFrame:
    """A well-specified regression: real signal, constant noise, no pathology."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50.0, 12.0, size=rows)
    x2 = rng.normal(20.0, 4.0, size=rows)
    group = rng.choice(["a", "b", "c"], size=rows)
    y = 2.0 * x1 + 3.0 * x2 + rng.normal(0.0, 6.0, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "group": group, "y": y})


def _run(frame: pd.DataFrame, target: str = "y", **kwargs):
    return pe.load({"t": frame}).regression(target, **kwargs)


def _kinds(result) -> set[str]:
    return {item.kind for item in result.evidence}


def _evidence(result, kind):
    return next(item for item in result.evidence if item.kind == kind)


# --------------------------------------------------------------------------
# Silence on clean data
# --------------------------------------------------------------------------


def test_clean_regression_reports_no_issues() -> None:
    """The whole product thesis: nothing wrong means nothing reported."""
    result = _run(_clean_frame())
    issues, _ = split_findings(result.findings)
    assert result.status is AnalysisStatus.COMPLETED
    assert issues == [], [finding.title for finding in issues]
    assert "looks ready" in result.summary


def test_clean_regression_still_produces_evidence() -> None:
    """Silence in the findings is not silence in the evidence."""
    result = _run(_clean_frame())
    assert "regression_target_summary" in _kinds(result)
    assert "regression_probe" in _kinds(result)
    probe = _evidence(result, "regression_probe")
    # Signal is 2*x1 + 3*x2 against noise of 6, so a linear probe must find it.
    assert probe.value["best_r_squared"] > 0.9


def test_homoscedastic_data_does_not_report_uneven_spread() -> None:
    result = _run(_clean_frame())
    titles = [finding.title for finding in result.findings]
    assert "Error spread changes across the prediction range" not in titles


def test_symmetric_target_produces_no_shape_finding() -> None:
    result = _run(_clean_frame())
    shape = _evidence(result, "regression_target_shape")
    assert shape.value["shape"] == "symmetric"
    assert not any("skew" in finding.title.lower() for finding in result.findings)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_affine_copy_of_target_is_a_critical_leak() -> None:
    frame = _clean_frame()
    # Annualized revenue: an affine copy, sharing no name token with the target.
    frame["annual_total"] = frame["y"] * 12.0
    result = _run(frame)
    leaks = [
        finding
        for finding in result.findings
        if finding.title.startswith("Potential target leakage")
    ]
    assert len(leaks) == 1
    assert "annual_total" in leaks[0].title
    assert leaks[0].severity == "critical"


def test_leaking_feature_is_excluded_from_the_probe() -> None:
    """A probe trained on the leak would score ~1.0 and bury the finding."""
    frame = _clean_frame()
    frame["annual_total"] = frame["y"] * 12.0
    result = _run(frame)
    probe = _evidence(result, "regression_probe")
    assert "annual_total" in probe.value["excluded_features"]
    assert "annual_total" not in probe.value["numeric_features"]


def test_leakage_finding_cites_real_evidence() -> None:
    frame = _clean_frame()
    frame["annual_total"] = frame["y"] * 12.0
    result = _run(frame)
    ids = {item.id for item in result.evidence}
    for finding in result.findings:
        assert finding.evidence_ids
        assert set(finding.evidence_ids) <= ids


def test_short_target_name_does_not_match_by_substring() -> None:
    """A target called ``y`` must not "leak" into every column containing a y.

    Raw substring matching finds ``y`` inside ``x1_copy``. Name evidence is
    taken from whole tokens, and only tokens long enough to carry meaning.
    """
    frame = _clean_frame()
    rng = np.random.default_rng(2)
    frame["x1_copy"] = frame["x1"] + rng.normal(0.0, 0.01, size=len(frame))
    result = _run(frame)
    assert not any("leakage" in finding.title.lower() for finding in result.findings), [
        finding.title for finding in result.findings
    ]


def test_shared_name_token_still_raises_a_leak_candidate() -> None:
    frame = _clean_frame()
    frame = frame.rename(columns={"y": "revenue"})
    frame["revenue_forecast"] = frame["revenue"] * 0.5 + 3.0
    result = _run(frame, target="revenue")
    assert any("revenue_forecast" in finding.title for finding in result.findings), [
        finding.title for finding in result.findings
    ]


def test_identifier_column_is_flagged_and_excluded() -> None:
    frame = _clean_frame()
    frame["account_id"] = np.arange(len(frame))
    result = _run(frame)
    titles = [finding.title for finding in result.findings]
    assert "Identifier-like feature: account_id" in titles
    probe = _evidence(result, "regression_probe")
    assert "account_id" in probe.value["excluded_features"]


# --------------------------------------------------------------------------
# Shape is a property, not a defect
# --------------------------------------------------------------------------


def test_skew_is_an_alert_never_an_issue() -> None:
    """A skewed target is not broken, and must not sit beside a leak."""
    rng = np.random.default_rng(11)
    rows = 400
    x1 = rng.normal(50.0, 12.0, size=rows)
    frame = pd.DataFrame(
        {"x1": x1, "y": np.exp(rng.normal(0.0, 1.0, size=rows)) * 10.0 + x1}
    )
    result = _run(frame)
    skew_findings = [
        finding for finding in result.findings if "skew" in finding.title.lower()
    ]
    assert skew_findings, "a strongly skewed target should still be reported"
    for finding in skew_findings:
        assert finding.category == OBSERVATION
        assert finding.severity == "low"


def test_transformation_candidate_is_measured_not_asserted() -> None:
    rng = np.random.default_rng(11)
    rows = 400
    frame = pd.DataFrame({"x1": rng.normal(size=rows)})
    frame["y"] = np.exp(rng.normal(0.0, 1.0, size=rows)) * 10.0
    shape = _evidence(_run(frame), "regression_target_shape")
    best = shape.value["best_candidate"]
    assert best is not None
    # The recommendation must be backed by a measured reduction on this data.
    assert abs(shape.value["best_skewness_after"]) < abs(shape.value["skewness"])
    chosen = next(
        item for item in shape.value["candidates"] if item["transform"] == best
    )
    assert chosen["skew_reduction"] >= 0.3


def test_no_transformation_is_recommended_for_a_symmetric_target() -> None:
    shape = _evidence(_run(_clean_frame()), "regression_target_shape")
    assert shape.value["best_candidate"] is None


# --------------------------------------------------------------------------
# Censoring, spikes, heaping
# --------------------------------------------------------------------------


def test_censored_target_is_reported_as_an_issue() -> None:
    frame = _clean_frame()
    ceiling = float(np.quantile(frame["y"], 0.85))
    frame["y"] = frame["y"].clip(upper=ceiling)
    result = _run(frame)
    issues, _ = split_findings(result.findings)
    spike_titles = [finding.title for finding in issues if "pile up" in finding.title]
    assert spike_titles or any("capped" in finding.title for finding in issues)
    spikes = _evidence(result, "regression_target_spikes")
    assert spikes.value["spikes"][0]["position"] == "at_maximum"


def test_spike_detector_ignores_a_genuinely_discrete_target() -> None:
    """A 1-5 rating repeats by construction; that is not censoring."""
    rng = np.random.default_rng(5)
    rows = 300
    frame = pd.DataFrame({"x1": rng.normal(size=rows)})
    frame["y"] = rng.integers(1, 6, size=rows).astype(float)
    result = _run(frame)
    assert "regression_target_spikes" not in _kinds(result)
    assert "regression_target_heaping" not in _kinds(result)


def test_discrete_target_is_an_alert_pointing_at_classification() -> None:
    rng = np.random.default_rng(5)
    rows = 300
    frame = pd.DataFrame({"x1": rng.normal(size=rows)})
    frame["y"] = rng.integers(1, 6, size=rows).astype(float)
    result = _run(frame)
    matches = [
        finding for finding in result.findings if "few distinct values" in finding.title
    ]
    assert len(matches) == 1
    assert matches[0].category == OBSERVATION


def test_heaping_does_not_fire_on_unrounded_values() -> None:
    assert "regression_target_heaping" not in _kinds(_run(_clean_frame()))


# --------------------------------------------------------------------------
# Weak support: a thin tail is not a hole
# --------------------------------------------------------------------------


def test_thin_tail_of_a_normal_target_is_not_weak_support() -> None:
    """Equal-width bins always leave thin bins in a bell curve's tails.

    Treating those as missing support would fire on every normally distributed
    target ever analysed.
    """
    result = _run(_clean_frame())
    titles = [finding.title for finding in result.findings]
    assert "Parts of the range have almost no support" not in titles


def test_genuine_hole_in_the_target_range_is_reported() -> None:
    """Two separated populations leave a real gap, with mass on both sides."""
    rng = np.random.default_rng(9)
    low = rng.normal(20.0, 4.0, size=200)
    high = rng.normal(120.0, 4.0, size=200)
    x = np.concatenate([low, high])
    frame = pd.DataFrame(
        {
            "x": x + rng.normal(0.0, 1.0, size=400),
            "y": 2.0 * x + rng.normal(0.0, 2.0, 400),
        }
    )
    result = _run(frame)
    support = _evidence(result, "regression_weak_support")
    assert support.value["sparse_target_bins"]
    for item in support.value["sparse_target_bins"]:
        assert item["mass_below"] >= 0.05
        assert item["mass_above"] >= 0.05


# --------------------------------------------------------------------------
# Residual diagnostics
# --------------------------------------------------------------------------


def test_heteroscedastic_data_reports_uneven_spread() -> None:
    rng = np.random.default_rng(7)
    rows = 500
    x1 = rng.uniform(1.0, 100.0, size=rows)
    # Noise scale grows with x1, so the spread genuinely fans out.
    frame = pd.DataFrame({"x1": x1, "y": 3.0 * x1 + rng.normal(0.0, x1 * 0.9)})
    result = _run(frame)
    titles = [finding.title for finding in result.findings]
    assert "Error spread changes across the prediction range" in titles
    spread = _evidence(result, "regression_heteroscedasticity")
    assert spread.value["spread_ratio"] >= 3.0
    assert spread.value["breusch_pagan_statistic"] is not None


def test_probe_reports_no_signal_when_features_are_noise() -> None:
    rng = np.random.default_rng(13)
    rows = 300
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "y": rng.normal(size=rows),
        }
    )
    result = _run(frame)
    titles = [finding.title for finding in result.findings]
    assert any("carry little signal" in title for title in titles)


def test_residual_shape_reports_a_distance_never_a_p_value() -> None:
    result = _run(_clean_frame())
    shape = _evidence(result, "regression_residual_shape")
    assert "normality_distance" in shape.value
    assert not any("p_value" in key for key in shape.value)


def test_residual_diagnostics_are_labelled_model_conditional() -> None:
    probe = _evidence(_run(_clean_frame()), "regression_probe")
    assert any("diagnostic" in item for item in probe.assumptions)


# --------------------------------------------------------------------------
# Redundancy and non-linearity
# --------------------------------------------------------------------------


def test_redundant_pair_is_an_alert_with_vif_reported() -> None:
    frame = _clean_frame()
    rng = np.random.default_rng(2)
    frame["x1_copy"] = frame["x1"] + rng.normal(0.0, 0.01, size=len(frame))
    result = _run(frame)
    matches = [
        finding for finding in result.findings if "Redundant features" in finding.title
    ]
    assert len(matches) == 1
    assert matches[0].category == OBSERVATION
    redundancy = _evidence(result, "regression_redundancy")
    assert redundancy.value["redundant_pairs"]
    assert redundancy.value["variance_inflation"]


def test_non_linear_alert_is_not_repeated_for_a_redundant_twin() -> None:
    """One relationship, reported once — not once per interchangeable column."""
    rng = np.random.default_rng(4)
    rows = 400
    # A parabola, so the curve is non-monotone and a straight line finds nothing.
    x1 = rng.uniform(-5.0, 5.0, size=rows)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x1_copy": x1 + rng.normal(0.0, 0.005, size=rows),
            "y": x1**2 + rng.normal(0.0, 1.0, size=rows),
        }
    )
    result = _run(frame)
    nonlinear = [
        finding for finding in result.findings if "non-linearly" in finding.title
    ]
    assert len(nonlinear) == 1


def test_non_linear_relationship_is_detected_at_all() -> None:
    rng = np.random.default_rng(4)
    rows = 400
    x1 = rng.uniform(-5.0, 5.0, size=rows)
    # A parabola: zero linear correlation, strong binned signal.
    frame = pd.DataFrame({"x1": x1, "y": x1**2 + rng.normal(0.0, 1.0, size=rows)})
    result = _run(frame)
    association = _evidence(result, "regression_feature_association")
    assert association.value["is_nonlinear"]
    assert (
        association.value["binned_eta_squared"] > association.value["linear_r_squared"]
    )


# --------------------------------------------------------------------------
# Subgroup error must not be a scale artifact
# --------------------------------------------------------------------------


def test_group_with_larger_values_is_not_flagged_for_larger_errors() -> None:
    """Ten times the magnitude, ten times the error, equally well predicted.

    The feature is supplied already scaled, so one additive linear fit serves
    both groups equally well in relative terms. Raw mean absolute error still
    differs by roughly ten times between them, so a diagnostic that ranked on
    raw error would flag the large group every run and say nothing.
    """
    rng = np.random.default_rng(21)
    rows = 600
    group = rng.choice(["small", "large"], size=rows)
    scale = np.where(group == "large", 10.0, 1.0)
    base = scale * rng.normal(50.0, 12.0, size=rows)
    y = 2.0 * base + rng.normal(0.0, 6.0, size=rows) * scale
    frame = pd.DataFrame({"base": base, "group": group, "y": y})
    result = _run(frame)

    concentration = _evidence(result, "regression_error_concentration")
    by_level = {item["level"]: item for item in concentration.value["groups"]}
    raw_gap = (
        by_level["large"]["mean_absolute_error"]
        / by_level["small"]["mean_absolute_error"]
    )
    # The naive metric would have fired hard; the scaled one must not.
    assert raw_gap > 5.0
    assert by_level["large"]["error_ratio"] < 1.5

    titles = [finding.title for finding in result.findings]
    assert not any("Error concentrates" in title for title in titles), titles


def test_genuinely_worse_served_group_is_reported() -> None:
    rng = np.random.default_rng(22)
    rows = 600
    group = rng.choice(["clean", "noisy"], size=rows)
    x1 = rng.normal(50.0, 12.0, size=rows)
    # Same magnitude in both groups; one is simply unpredictable.
    noise = np.where(group == "noisy", 60.0, 3.0)
    frame = pd.DataFrame(
        {"x1": x1, "group": group, "y": 2.0 * x1 + rng.normal(0.0, noise)}
    )
    result = _run(frame)
    concentration = _evidence(result, "regression_error_concentration")
    assert concentration.value["max_error_ratio"] > 1.5
    assert concentration.value["groups"][0]["level"] == "noisy"


# --------------------------------------------------------------------------
# Influence and review rows
# --------------------------------------------------------------------------


def test_influential_rows_are_surfaced_for_review() -> None:
    result = _run(subscriptions(), "monthly_revenue")
    rows = _evidence(result, "regression_review_rows").value["rows"]
    assert rows
    # The two planted data-entry errors must both make the list.
    flagged = {row["row_index"] for row in rows}
    assert {"7", "123"} <= flagged
    worst = rows[0]
    assert worst["reasons"]
    assert worst["cooks_distance"] is not None


def test_clean_data_produces_no_review_rows() -> None:
    """The 4/n screen flags a few rows in *any* dataset; that is not a finding.

    Roughly one row in a clean 400-row fit clears a 3-sigma residual by chance.
    Publishing it would put a review list on every well-behaved regression, so
    the list requires at least one decisively influential row to exist at all.
    """
    result = _run(_clean_frame())
    assert "regression_review_rows" not in _kinds(result)
    # The underlying counts are still recorded as evidence, just not promoted.
    influence = _evidence(result, "regression_influence")
    assert influence.value["decisive_row_count"] == 0
    assert "cooks_distance_threshold" in influence.value


def test_review_rows_carry_their_own_explanation() -> None:
    result = _run(subscriptions(), "monthly_revenue")
    rows = _evidence(result, "regression_review_rows").value["rows"]
    top = next(row for row in rows if row["row_index"] == "7")
    assert "extreme feature values" in top["reasons"]
    assert any(item["column"] == "seats" for item in top["extreme_features"])


def test_residual_scatter_marks_the_review_rows() -> None:
    result = _run(subscriptions(), "monthly_revenue")
    scatter = _evidence(result, "regression_residual_scatter")
    assert scatter.value["points"]
    assert any(point["flagged"] for point in scatter.value["points"])


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


def test_analysis_does_not_mutate_the_caller_frame() -> None:
    frame = subscriptions()
    before = frame.copy(deep=True)
    pe.load({"subscriptions": frame}).regression("monthly_revenue")
    pd.testing.assert_frame_equal(frame, before)


def test_repeated_runs_are_byte_identical() -> None:
    first = _run(subscriptions(), "monthly_revenue")
    second = _run(subscriptions(), "monthly_revenue")
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    assert first.summary == second.summary
    assert first.to_dict() == second.to_dict()


def test_missing_target_is_insufficient_evidence() -> None:
    result = _run(_clean_frame(), target="nope")
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert any(
        warning.code == "regression_target_not_found" for warning in result.warnings
    )


def test_text_target_is_refused_with_a_pointer_to_classification() -> None:
    frame = _clean_frame()
    frame["label"] = "yes"
    result = _run(frame, target="label")
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    warning = next(
        item for item in result.warnings if item.code == "regression_target_not_numeric"
    )
    assert "classification()" in warning.message


def test_ambiguous_target_across_tables_asks_for_a_table() -> None:
    frame = _clean_frame(rows=60)
    result = pe.load({"a": frame, "b": frame.copy()}).regression("y")
    assert any(
        warning.code == "regression_target_ambiguous" for warning in result.warnings
    )


def test_tiny_table_reports_insufficient_evidence() -> None:
    frame = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    result = _run(frame)
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_constant_target_reports_insufficient_evidence() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x1": rng.normal(size=100), "y": np.full(100, 5.0)})
    result = _run(frame)
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_all_null_target_reports_insufficient_evidence() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x1": rng.normal(size=100), "y": np.full(100, np.nan)})
    result = _run(frame)
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_partially_missing_target_is_reported_as_an_issue() -> None:
    frame = _clean_frame()
    frame.loc[frame.index[:120], "y"] = np.nan
    result = _run(frame)
    issues, _ = split_findings(result.findings)
    assert any("Missing target values" in finding.title for finding in issues)


def test_small_table_is_not_sampled() -> None:
    result = _run(_clean_frame(rows=200), mode="quick")
    assert result.sampling == ()
    assert result.warnings == ()


def test_sampling_is_deterministic_and_disclosed() -> None:
    """Over the mode budget, the cut must be recorded, stated, and repeatable."""
    frame = _clean_frame(rows=26_000)
    first = _run(frame, mode="quick")
    second = _run(frame, mode="quick")

    record = next(
        item for item in first.sampling if item.operation == "regression_analysis"
    )
    assert record.source_rows == 26_000
    assert record.sampled_rows == 25_000
    assert record.seed == 42
    assert record.limitations
    # A truncation the reader cannot see is worse than a large report.
    assert any(
        warning.code == "sampled_regression_analysis" for warning in first.warnings
    )
    assert first.status is AnalysisStatus.COMPLETED_WITH_WARNINGS
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]


def test_sampling_can_be_disabled() -> None:
    result = _run(_clean_frame(rows=26_000), mode="quick", sampling="disabled")
    assert result.sampling == ()


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------


def test_exports_round_trip(tmp_path) -> None:
    result = _run(subscriptions(), "monthly_revenue")
    html = result.to_html(tmp_path / "regression.html")
    payload = result.to_json(tmp_path / "regression.json")
    assert html.exists() and payload.exists()
    text = html.read_text(encoding="utf-8")
    assert "Regression readiness" in text
    assert "Rows to review" in text
    assert "http://" not in text.replace("http://www.w3.org", "")


def test_report_names_the_residual_sections() -> None:
    text = _run(subscriptions(), "monthly_revenue").render_html()
    for anchor in ("section-rows", "section-residuals", "section-target"):
        assert f'id="{anchor}"' in text


def test_api_and_dataset_entry_points_agree() -> None:
    frame = subscriptions()
    direct = pe.regression({"subscriptions": frame}, "monthly_revenue")
    session = pe.load({"subscriptions": frame}).regression("monthly_revenue")
    assert direct.summary == session.summary


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(TypeError):
        pe.load({"t": _clean_frame(rows=60)}).analyze(
            "regression", target="y", nonsense=1
        )
