"""Tree-based feature importance, shared by the regression and classification
recipes.

The tests that matter most here are the ones proving silence. A forest ranks
whatever it is given, so the first three tests assert that pure noise produces
no section at all, that clean well-specified data promotes no finding, and that
an honest leader among honest features is not reported as a soft leak. Only
after those do the tests check that each detector fires when it should.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prism_eda as pe
from examples.sample_data import feature_signal
from prism_eda.analysis.feature_importance import EVIDENCE_KIND
from prism_eda.results import AnalysisStatus

# --------------------------------------------------------------------------
# Frames, each built for one question
# --------------------------------------------------------------------------


def _noise_frame(rows: int = 400, seed: int = 1) -> pd.DataFrame:
    """No relationship of any kind. The model must find nothing."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"x{index}": rng.normal(size=rows) for index in range(5)})
    frame["y"] = rng.normal(size=rows)
    return frame


def _clean_frame(rows: int = 400, seed: int = 3) -> pd.DataFrame:
    """Real signal, unequal coefficients, no pathology.

    ``x1`` carries roughly four times ``x2``'s importance purely because of its
    coefficient and spread. That is honest data, and nothing here may be filed
    as a defect — the case that caught a share-based dominance rule.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50.0, 12.0, size=rows)
    x2 = rng.normal(20.0, 4.0, size=rows)
    group = rng.choice(["a", "b", "c"], size=rows)
    y = 2.0 * x1 + 3.0 * x2 + rng.normal(0.0, 6.0, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "group": group, "y": y})


def _decoy_frame(rows: int = 800, seed: int = 3) -> pd.DataFrame:
    """One weak real driver beside two wide decoys.

    Impurity bias is a phenomenon of *weak* models: when the real signal is
    strong, impurity concentrates on it and the decoys never climb. So the
    target here is deliberately noisy, which is the regime the finding exists
    for and the one ``feature_signal()`` is not.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "x1": x1,
            "wide_code": rng.normal(size=rows) * 100.0,
            "region_code": rng.integers(0, 40, size=rows).astype(str),
            "flag": rng.integers(0, 2, size=rows),
            "y": 1.2 * x1 + rng.normal(0.0, 2.2, size=rows),
        }
    )


def _proxy_frame(rows: int = 700, seed: int = 11) -> pd.DataFrame:
    """A soft leak: one column reproduces the target well short of the 98% bar."""
    rng = np.random.default_rng(seed)
    core = rng.normal(100.0, 20.0, size=rows)
    return pd.DataFrame(
        {
            "estimate": core + rng.normal(0.0, 8.0, size=rows),
            "region": rng.choice(list("abcd"), size=rows),
            "seats": rng.integers(1, 9, size=rows),
            "noise": rng.normal(size=rows),
            "y": core,
        }
    )


def _label_frame(rows: int = 800, seed: int = 9) -> pd.DataFrame:
    """A classification target driven by a threshold on one feature."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    logit = np.where(a > 0.4, 2.5, -2.0) + 0.8 * b
    churned = (rng.uniform(size=rows) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "a": a,
            "b": b,
            "junk": rng.normal(size=rows),
            "code": rng.integers(0, 50, size=rows).astype(str),
            "churned": churned,
        }
    )


def _run(frame: pd.DataFrame, target: str = "y", **kwargs):
    return pe.load({"t": frame}).regression(target, **kwargs)


def _importance(result):
    return next((item for item in result.evidence if item.kind == EVIDENCE_KIND), None)


def _titles(result) -> list[str]:
    return [finding.title for finding in result.findings]


def _steps(result, operation: str):
    return [
        step for step in result.transformation_plan.steps if step.operation == operation
    ]


def _columns(result, operation: str) -> set[str]:
    return {column for step in _steps(result, operation) for column in step.columns}


# --------------------------------------------------------------------------
# Gate 1: a model with no skill produces no ranking
# --------------------------------------------------------------------------


def test_pure_noise_produces_no_importance_evidence() -> None:
    """A forest ranks anything. On noise the ranking must not be reported."""
    result = _run(_noise_frame())
    assert _importance(result) is None


def test_pure_noise_records_why_it_stayed_silent() -> None:
    """Silence has to be visible, or the reader cannot tell it from a bug."""
    result = _run(_noise_frame())
    codes = [warning.code for warning in result.warnings]
    assert "feature_importance_no_signal" in codes
    message = next(
        warning.message
        for warning in result.warnings
        if warning.code == "feature_importance_no_signal"
    )
    assert "ordering of noise" in message


def test_pure_noise_renders_no_drivers_section() -> None:
    from prism_eda.reporting.sections import report_sections

    result = _run(_noise_frame())
    assert "drivers" not in report_sections(result)
    assert "section-drivers" not in result.render_html()


# --------------------------------------------------------------------------
# Silence on clean data
# --------------------------------------------------------------------------


def test_clean_data_promotes_no_importance_finding() -> None:
    """Unequal but honest coefficients are not a defect and not a soft leak."""
    result = _run(_clean_frame())
    importance = _importance(result)
    assert importance is not None, "a clean fit should still produce the ranking"
    assert not any(row["cardinality_biased"] for row in importance.value["features"])
    assert "is almost the whole model" not in " ".join(_titles(result))


def test_clean_data_does_not_claim_the_tree_beat_the_linear_probe() -> None:
    """On a linear relationship the two scores match; the gap is not a finding."""
    result = _run(_clean_frame())
    importance = _importance(result)
    assert importance is not None
    assert importance.value["tree_advantage"] < 0.10
    assert "A tree finds signal the linear probe missed" not in _titles(result)


def test_clean_data_still_banks_the_evidence() -> None:
    """Silence in the findings is not silence in the evidence."""
    value = _importance(_run(_clean_frame())).value
    assert value["feature_count"] == 3
    assert value["model_score"] > 0.5
    assert len(value["sentinels"]) == 3


# --------------------------------------------------------------------------
# The measured noise floor
# --------------------------------------------------------------------------


def test_sentinels_are_measured_not_assumed() -> None:
    """Three manufactured columns, one of them shaped like a real one."""
    value = _importance(_run(_clean_frame())).value
    kinds = {sentinel["kind"] for sentinel in value["sentinels"]}
    assert kinds == {"gaussian_noise", "uniform_noise", "shuffled_copy"}
    shuffled = next(
        sentinel
        for sentinel in value["sentinels"]
        if sentinel["kind"] == "shuffled_copy"
    )
    assert shuffled["source"] in {"x1", "x2", "group"}
    assert value["noise_floor"] >= 0.0


def test_sentinels_never_reach_the_feature_ranking() -> None:
    """They calibrate the floor; they are not columns the user has."""
    value = _importance(_run(_clean_frame())).value
    names = {row["feature"] for row in value["features"]}
    assert not any(name.startswith("__prism_sentinel") for name in names)
    assert names == {"x1", "x2", "group"}


def test_real_drivers_clear_the_floor_and_noise_does_not() -> None:
    value = _importance(_run(_clean_frame())).value
    by_name = {row["feature"]: row for row in value["features"]}
    assert by_name["x1"]["below_noise_floor"] is False
    assert by_name["x2"]["below_noise_floor"] is False
    assert by_name["group"]["below_noise_floor"] is True


# --------------------------------------------------------------------------
# The three findings, each on the data it exists for
# --------------------------------------------------------------------------


def test_wide_decoys_are_reported_as_high_in_sample_only() -> None:
    result = _run(_decoy_frame())
    value = _importance(result).value
    biased = {row["feature"] for row in value["features"] if row["cardinality_biased"]}
    assert {"wide_code", "region_code"} <= biased
    assert "flag" not in biased, "an ordinary weak column is not a cardinality story"
    assert any("ranks high in sample but not out of it" in t for t in _titles(result))


def test_decoy_finding_states_the_measurement_not_a_cause() -> None:
    """A decoy and a masked real driver are indistinguishable, so say what was
    measured rather than asserting why."""
    result = _run(_decoy_frame())
    finding = next(item for item in result.findings if "wide_code" in item.title)
    assert "impurity importance" in finding.summary
    assert "held-out rows" in finding.summary
    assert "masked by" in finding.recommendation


def test_decoys_are_reviewed_rather_than_dropped() -> None:
    result = _run(_decoy_frame())
    dropped = _columns(result, "drop_uninformative_features")
    reviewed = _columns(result, "review_biased_importance")
    assert "wide_code" in reviewed
    assert "wide_code" not in dropped


def test_soft_leak_is_reported_when_one_column_is_the_whole_model() -> None:
    result = _run(_proxy_frame())
    value = _importance(result).value
    assert value["top_feature_solo_reliance"] >= 0.90
    assert "estimate is almost the whole model" in _titles(result)
    assert _steps(result, "review_dominant_feature")[0].risk == "high"


def test_non_linear_target_reports_the_tree_beating_the_linear_probe() -> None:
    result = pe.load({"feature_signal": feature_signal()}).regression("renewal_value")
    value = _importance(result).value
    assert value["linear_score"] < 0.4, "a line should struggle on a non-monotonic step"
    assert value["model_score"] > 0.9
    assert "A tree finds signal the linear probe missed" in _titles(result)


def test_tree_beating_linear_is_an_alert_not_a_defect() -> None:
    """The data being curved is true, not broken."""
    from prism_eda.evidence.models import OBSERVATION

    result = pe.load({"feature_signal": feature_signal()}).regression("renewal_value")
    finding = next(
        item
        for item in result.findings
        if item.title == "A tree finds signal the linear probe missed"
    )
    assert finding.category == OBSERVATION


# --------------------------------------------------------------------------
# The drop list, and the correlated-pair guard on it
# --------------------------------------------------------------------------


def test_below_floor_columns_reach_the_plan_not_the_findings() -> None:
    """Uninformative is a suggestion for the next pass, not a defect."""
    result = pe.load({"feature_signal": feature_signal()}).regression("renewal_value")
    steps = _steps(result, "drop_uninformative_features")
    assert steps and steps[0].risk == "low"
    assert steps[0].requires_approval is True
    assert not any("carry no measured signal" in title for title in _titles(result))


def test_redundant_partners_are_never_recommended_for_removal() -> None:
    """Permutation splits their credit, so dropping both would lose the signal."""
    result = pe.load({"feature_signal": feature_signal()}).regression("renewal_value")
    value = _importance(result).value
    by_name = {row["feature"]: row for row in value["features"]}
    assert by_name["seats"]["has_redundant_partner"] is True
    assert by_name["seats_billed"]["has_redundant_partner"] is True

    dropped = _columns(result, "drop_uninformative_features")
    assert "seats" not in dropped
    assert "seats_billed" not in dropped


def test_the_plan_says_why_a_redundant_column_was_held_back() -> None:
    result = pe.load({"feature_signal": feature_signal()}).regression("renewal_value")
    step = _steps(result, "drop_uninformative_features")[0]
    assert "seats" in step.rationale
    assert "splits credit" in step.rationale
    assert step.parameters["held_back_redundant"]


# --------------------------------------------------------------------------
# The screened feature set is shared with the recipe's own probe
# --------------------------------------------------------------------------


def test_leaks_and_identifiers_never_enter_the_ranking() -> None:
    """A forest trained on the leak would rank the leak first."""
    result = pe.load({"feature_signal": feature_signal()}).regression("renewal_value")
    names = {row["feature"] for row in _importance(result).value["features"]}
    assert "account_id" not in names, "identifiers label rows, they do not explain"
    assert "renewal_value" not in names


def test_ranking_matches_the_linear_probe_feature_set() -> None:
    result = pe.load({"feature_signal": feature_signal()}).regression("renewal_value")
    probe = next(item for item in result.evidence if item.kind == "regression_probe")
    expected = set(probe.value["numeric_features"]) | set(
        probe.value["categorical_features"]
    )
    ranked = {row["feature"] for row in _importance(result).value["features"]}
    assert ranked == expected


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def test_classification_produces_the_same_ranking_shape() -> None:
    result = pe.load({"t": _label_frame()}).classification("churned")
    value = _importance(result).value
    assert value["task"] == "classification"
    assert value["metric"] == "balanced_accuracy"
    assert value["model"] == "random_forest_classifier"
    assert value["baseline_model"] == "majority_baseline"
    assert value["linear_model"] == "logistic_regression"


def test_classification_ranking_is_scored_against_a_majority_baseline() -> None:
    """Balanced accuracy floors at 1/classes, not zero, so the gate is a lift."""
    value = _importance(pe.load({"t": _label_frame()}).classification("churned")).value
    assert value["model_score"] > value["baseline_score"]
    assert value["features"][0]["feature"] == "a"


def test_classification_noise_target_produces_no_ranking() -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=400),
            "b": rng.normal(size=400),
            "label": rng.integers(0, 2, size=400),
        }
    )
    result = pe.load({"t": frame}).classification("label")
    assert _importance(result) is None
    assert "feature_importance_no_signal" in [w.code for w in result.warnings]


def test_classification_section_appears_after_the_findings() -> None:
    from prism_eda.reporting.sections import report_sections

    result = pe.load({"t": _label_frame()}).classification("churned")
    order = [section.id for section in report_sections(result)]
    assert "drivers" in order
    assert order.index("drivers") > order.index("issues")


# --------------------------------------------------------------------------
# Contracts the rest of the library relies on
# --------------------------------------------------------------------------


def test_sentinels_do_not_mutate_the_callers_frame() -> None:
    """Three columns are injected for the fit; none may survive into the input."""
    frame = _clean_frame()
    before = frame.copy(deep=True)
    _run(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert list(frame.columns) == ["x1", "x2", "group", "y"]


def test_the_ranking_is_deterministic() -> None:
    """Identical input must mint an identical evidence ID, run after run."""
    frame = _clean_frame()
    first = _importance(_run(frame))
    second = _importance(_run(frame))
    assert first.id == second.id
    assert first.value == second.value


def test_a_different_seed_is_still_internally_consistent() -> None:
    value = _importance(_run(_clean_frame(), random_seed=99)).value
    assert value["features"][0]["feature"] == "x1"
    assert value["permutation_repeats"] == 5


@pytest.mark.parametrize("mode,repeats,trees", [("quick", 3, 100), ("deep", 10, 300)])
def test_compute_budget_scales_with_mode(mode: str, repeats: int, trees: int) -> None:
    value = _importance(_run(_clean_frame(), mode=mode)).value
    assert value["permutation_repeats"] == repeats
    assert value["tree_count"] == trees


def test_short_tables_are_skipped_rather_than_guessed_at() -> None:
    """A 75/25 split of 60 rows leaves a holdout too small to mean anything."""
    result = _run(_clean_frame(rows=60))
    assert _importance(result) is None


def test_importance_survives_json_export() -> None:
    result = _run(_proxy_frame())
    payload = result.to_dict()
    banked = [item for item in payload["evidence"] if item["kind"] == EVIDENCE_KIND]
    assert len(banked) == 1
    assert banked[0]["value"]["features"][0]["feature"] == "estimate"
    assert banked[0]["value"]["sentinels"]


def test_importance_renders_into_the_html_report() -> None:
    result = _run(_proxy_frame())
    html = result.render_html()
    assert 'id="section-drivers"' in html
    assert "What drives the target" in html
    assert "noise floor" in html
    assert "sentinel &mdash; manufactured noise" in html or "sentinel" in html
    assert result.status is AnalysisStatus.COMPLETED


def test_report_shows_every_feature_even_when_the_chart_caps_them() -> None:
    """A cap may shorten the chart; it may not drop a column from the page."""
    rng = np.random.default_rng(21)
    rows = 600
    frame = pd.DataFrame({f"f{index}": rng.normal(size=rows) for index in range(22)})
    frame["y"] = 3.0 * frame["f0"] + 2.0 * frame["f1"] + rng.normal(0.0, 1.0, rows)
    result = _run(frame)
    value = _importance(result).value
    assert value["feature_count"] > value["chart_feature_count"] == 15
    html = result.render_html()
    for row in value["features"]:
        assert row["feature"] in html
