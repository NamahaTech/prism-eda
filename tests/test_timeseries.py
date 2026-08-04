"""Time-series forecasting readiness.

As with the regression suite, the tests that carry the most weight are the ones
proving a detector stays *quiet*. Time-series diagnostics are unusually prone to
reporting on everything: binary segmentation chops a smooth trend into a
staircase of "regime changes", an outlier rule calibrated on a robust remainder
flags a steady trickle of ordinary days, and almost every real series with a
trend is non-stationary. Each of those is checked here against data with nothing
wrong in it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prism_eda as pe
from examples.sample_data import daily_orders, daily_orders_single
from prism_eda.analysis._timeseries import infer_frequency
from prism_eda.evidence.models import OBSERVATION, split_findings
from prism_eda.results import AnalysisStatus

WEEKEND_LIFT = 20.0
WEEKDAY_DIP = -8.0


def _clean_series(rows: int = 730, seed: int = 5, slope: float = 0.05) -> pd.DataFrame:
    """A well-behaved daily series: real trend, real weekly cycle, no pathology."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    weekly = np.where(index.dayofweek >= 5, WEEKEND_LIFT, WEEKDAY_DIP)
    values = 100 + np.arange(rows) * slope + weekly + rng.normal(0.0, 7.0, rows)
    return pd.DataFrame({"day": index, "units": values})


def _run(frame: pd.DataFrame, value: str = "units", **kwargs):
    return pe.load({"t": frame}).time_series(value, **kwargs)


def _kinds(result) -> set[str]:
    return {item.kind for item in result.evidence}


def _evidence(result, kind):
    return next(item for item in result.evidence if item.kind == kind)


def _titles(result) -> list[str]:
    return [finding.title for finding in result.findings]


# --------------------------------------------------------------------------
# Silence on a clean series
# --------------------------------------------------------------------------


def test_clean_series_reports_no_issues() -> None:
    result = _run(_clean_series())
    issues, _ = split_findings(result.findings)
    assert result.status is AnalysisStatus.COMPLETED
    assert issues == [], [finding.title for finding in issues]
    assert "looks forecastable" in result.summary


def test_smooth_trend_is_not_a_staircase_of_level_shifts() -> None:
    """Binary segmentation will invent regime changes in any trending series."""
    result = _run(_clean_series(slope=0.12))
    changes = [
        item for item in result.evidence if item.kind == "time_series_change_points"
    ]
    for item in changes:
        assert item.value["level_shifts"] == [], item.value["level_shifts"]


def test_clean_series_reports_no_outliers() -> None:
    result = _run(_clean_series())
    outliers = [item for item in result.evidence if item.kind == "time_series_outliers"]
    total = sum(item.value["outlier_count"] for item in outliers)
    # A handful of points in 730 can clear any fence by chance; a *list* of them
    # is what would be noise.
    assert total <= 2, total


def test_clean_series_still_produces_structural_evidence() -> None:
    result = _run(_clean_series())
    assert "time_series_decomposition" in _kinds(result)
    assert "time_series_autocorrelation" in _kinds(result)
    assert "time_series_stationarity" in _kinds(result)
    assert "time_series_validation_plan" in _kinds(result)


# --------------------------------------------------------------------------
# Structure is described, never filed as a defect
# --------------------------------------------------------------------------


def test_seasonality_and_trend_are_alerts_not_issues() -> None:
    result = _run(_clean_series())
    _, alerts = split_findings(result.findings)
    labels = [finding.title for finding in alerts]
    assert any("seasonal" in title.lower() for title in labels), labels
    for finding in alerts:
        assert finding.category == OBSERVATION


def test_non_stationarity_is_an_alert_not_an_issue() -> None:
    """Almost every series with a trend is non-stationary. That is not a defect."""
    result = _run(_clean_series(slope=0.3))
    stationarity = _evidence(result, "time_series_stationarity")
    assert stationarity.value["verdict"] != "stationary"
    matching = [
        finding
        for finding in result.findings
        if finding.evidence_ids == (stationarity.id,)
    ]
    assert matching
    for finding in matching:
        assert finding.category == OBSERVATION


def test_stationarity_reports_both_tests_and_their_agreement() -> None:
    stationarity = _evidence(_run(_clean_series()), "time_series_stationarity")
    assert set(stationarity.value["adf"]) >= {"statistic", "p_value", "rejects_null"}
    assert set(stationarity.value["kpss"]) >= {"statistic", "p_value", "rejects_null"}
    assert isinstance(stationarity.value["tests_agree"], bool)
    assert stationarity.value["explanation"]


def test_weekly_seasonal_shape_matches_the_data() -> None:
    decomposition = _evidence(_run(_clean_series()), "time_series_decomposition")
    profile = {
        item["position"]: item["effect"]
        for item in decomposition.value["seasonal_profile"]
    }
    assert decomposition.value["period"] == 7
    # Saturday (5) and Sunday (6) were built to be the busy days.
    assert profile[5] > 0 and profile[6] > 0
    assert profile[0] < 0 and profile[2] < 0


def test_autocorrelation_finds_the_weekly_period() -> None:
    autocorrelation = _evidence(_run(_clean_series()), "time_series_autocorrelation")
    periods = [item["period"] for item in autocorrelation.value["seasonal_candidates"]]
    assert 7 in periods


# --------------------------------------------------------------------------
# Time-axis defects
# --------------------------------------------------------------------------


def test_unrecorded_block_is_reported_as_a_block() -> None:
    frame = _clean_series()
    outage = pd.date_range("2024-06-01", periods=9, freq="D")
    frame = frame[~frame["day"].isin(outage)]
    result = _run(frame)
    gaps = _evidence(result, "time_series_gaps")
    assert gaps.value["unrecorded_period_count"] == 9
    assert gaps.value["unrecorded_block_count"] == 1
    assert gaps.value["longest_unrecorded_block"] == 9
    assert any("never recorded" in title for title in _titles(result))


def test_absent_rows_and_blank_values_are_counted_separately() -> None:
    """Two different failures that a single missingness percentage would merge."""
    frame = _clean_series()
    frame = frame[~frame["day"].isin(pd.date_range("2024-06-01", periods=4, freq="D"))]
    frame.loc[
        frame["day"].isin(pd.date_range("2024-09-01", periods=3, freq="D")), "units"
    ] = np.nan
    gaps = _evidence(_run(frame), "time_series_gaps")
    assert gaps.value["unrecorded_period_count"] == 4
    assert gaps.value["blank_period_count"] == 3
    assert gaps.value["unrecorded_block_count"] == 1
    assert gaps.value["blank_block_count"] == 1


def test_duplicate_timestamps_are_reported() -> None:
    frame = _clean_series()
    frame = pd.concat([frame, frame.head(3)], ignore_index=True)
    result = _run(frame)
    duplicates = _evidence(result, "time_series_duplicate_timestamps")
    assert duplicates.value["duplicated_timestamp_count"] == 3
    assert any("Duplicate timestamps" in title for title in _titles(result))


def test_duplicate_detection_is_entity_aware() -> None:
    """In a panel every date appears once per entity; that is not duplication."""
    result = _run(daily_orders(), "orders", entity_id="store")
    duplicates = _evidence(result, "time_series_duplicate_timestamps")
    # The fixture plants exactly four double-reported days for one store.
    assert duplicates.value["duplicated_timestamp_count"] == 4
    assert duplicates.value["duplicated_row_count"] == 8


def test_panel_without_entity_column_would_look_entirely_duplicated() -> None:
    """The control for the test above: entity awareness is doing real work."""
    result = _run(daily_orders(), "orders")
    duplicates = _evidence(result, "time_series_duplicate_timestamps")
    assert duplicates.value["duplicated_row_count"] > 1000


def test_future_timestamps_are_flagged() -> None:
    frame = _clean_series()
    frame.loc[frame.index[-1], "day"] = pd.Timestamp.now() + pd.Timedelta(days=400)
    result = _run(frame)
    index = _evidence(result, "time_series_index")
    assert index.value["future_timestamp_count"] == 1
    assert any("future" in title.lower() for title in _titles(result))


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def test_panel_coverage_names_the_short_entity() -> None:
    result = _run(daily_orders(), "orders", entity_id="store")
    panel = _evidence(result, "time_series_panel_coverage")
    assert panel.value["entity_count"] == 3
    imbalanced = {item["entity"] for item in panel.value["imbalanced_entities"]}
    assert imbalanced == {"harbour"}
    assert any("less history" in title for title in _titles(result))


def test_changing_panel_membership_is_reported_as_an_issue() -> None:
    """A total whose membership changes produces level shifts that are not real."""
    result = _run(daily_orders(), "orders", entity_id="store")
    composition = _evidence(result, "time_series_panel_composition")
    assert composition.value["min_active_entities"] == 2
    assert composition.value["max_active_entities"] == 3
    issues, _ = split_findings(result.findings)
    assert any("not made of the same series" in f.title for f in issues), [
        f.title for f in issues
    ]


def test_balanced_panel_reports_no_composition_change() -> None:
    frame = _clean_series(rows=200)
    panel = pd.concat(
        [frame.assign(store=name) for name in ("a", "b")], ignore_index=True
    )
    result = _run(panel, entity_id="store")
    assert "time_series_panel_composition" not in _kinds(result)


def test_panel_analysis_discloses_that_it_aggregated() -> None:
    result = _run(daily_orders(), "orders", entity_id="store")
    assert any(
        warning.code == "time_series_panel_aggregated" for warning in result.warnings
    )


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


def test_real_level_shift_is_detected() -> None:
    frame = _clean_series()
    frame.loc[frame.index[400:], "units"] += 45.0
    result = _run(frame)
    changes = _evidence(result, "time_series_change_points")
    assert changes.value["level_shift_count"] == 1
    shift = changes.value["level_shifts"][0]
    assert shift["direction"] == "up"
    assert shift["step_in_noise"] > 1.0


def test_promotion_spikes_are_surfaced_as_outliers() -> None:
    result = _run(daily_orders_single(), "orders")
    outliers = _evidence(result, "time_series_outliers")
    flagged = {row["timestamp"][:10] for row in outliers.value["outliers"]}
    assert {"2025-06-14", "2025-11-28"} <= flagged


def test_interpolated_periods_are_never_flagged_as_outliers() -> None:
    """Prism must not report its own reconstruction as an anomaly."""
    result = _run(daily_orders_single(), "orders")
    outliers = _evidence(result, "time_series_outliers")
    flagged = {row["timestamp"][:10] for row in outliers.value["outliers"]}
    outage = {
        stamp.strftime("%Y-%m-%d")
        for stamp in pd.date_range("2025-04-07", periods=9, freq="D")
    }
    assert not (flagged & outage), flagged & outage


def test_changing_spread_suppresses_the_outlier_list() -> None:
    """Heteroscedasticity is not a run of anomalies, and is not reported as one."""
    rng = np.random.default_rng(6)
    rows = 730
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    weekly = np.where(index.dayofweek >= 5, WEEKEND_LIFT, WEEKDAY_DIP)
    values = 100 + weekly + rng.normal(0.0, 7.0, rows)
    values[500:] = 100 + weekly[500:] + rng.normal(0.0, 28.0, rows - 500)
    result = _run(pd.DataFrame({"day": index, "units": values}))
    outliers = _evidence(result, "time_series_outliers")
    assert outliers.value["suppressed_as_changing_spread"]
    assert outliers.value["outliers"] == []
    assert any("Spread changes" in title for title in _titles(result))


def test_intermittent_series_is_classified() -> None:
    rng = np.random.default_rng(1)
    rows = 500
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    values = np.where(rng.random(rows) < 0.65, 0.0, rng.gamma(2.0, 8.0, rows))
    result = _run(pd.DataFrame({"day": index, "units": values}))
    intermittency = _evidence(result, "time_series_intermittency")
    assert intermittency.value["pattern"] in {"intermittent", "lumpy"}
    assert intermittency.value["zero_rate"] > 0.5
    assert intermittency.value["longest_zero_run"] >= 2


def test_continuous_series_gets_no_intermittency_evidence() -> None:
    assert "time_series_intermittency" not in _kinds(_run(_clean_series()))


# --------------------------------------------------------------------------
# Horizon and validation
# --------------------------------------------------------------------------


def test_horizon_is_optional() -> None:
    result = _run(_clean_series())
    history = _evidence(result, "time_series_history")
    assert history.value["horizon"] is None
    assert history.value["adequate"]


def test_long_horizon_against_short_history_is_an_issue() -> None:
    result = _run(_clean_series(rows=120), horizon=90)
    history = _evidence(result, "time_series_history")
    assert not history.value["adequate"]
    assert history.value["reasons"]
    issues, _ = split_findings(result.findings)
    assert any("Not enough history" in finding.title for finding in issues)


def test_comfortable_horizon_raises_no_history_finding() -> None:
    result = _run(_clean_series(rows=730), horizon=14)
    history = _evidence(result, "time_series_history")
    assert history.value["adequate"]
    assert not any("Not enough history" in title for title in _titles(result))


def test_validation_plan_is_sized_to_the_horizon() -> None:
    plan = _evidence(_run(_clean_series(), horizon=28), "time_series_validation_plan")
    assert plan.value["feasible"]
    assert plan.value["test_periods_per_fold"] == 28
    assert plan.value["fold_count"] >= 1
    assert plan.value["strategy"] == "expanding_window_backtest"


# --------------------------------------------------------------------------
# Frequency inference
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "label", "period"),
    [("D", "daily", 7), ("h", "hourly", 24), ("MS", "monthly", 12)],
)
def test_frequency_inference(freq: str, label: str, period: int) -> None:
    stamps = pd.Series(pd.date_range("2020-01-01", periods=80, freq=freq))
    inferred = infer_frequency(stamps)
    assert inferred is not None
    assert inferred.label == label
    assert inferred.seasonal_period == period


def test_frequency_inference_survives_a_gap() -> None:
    """``pd.infer_freq`` returns None here; the modal spacing does not."""
    stamps = pd.date_range("2024-01-01", periods=100, freq="D")
    stamps = stamps.delete(range(40, 49))
    inferred = infer_frequency(pd.Series(stamps))
    assert inferred is not None and inferred.alias == "D"
    assert inferred.regularity < 1.0


def test_irregular_spacing_is_reported() -> None:
    rng = np.random.default_rng(3)
    base = pd.Timestamp("2024-01-01")
    offsets = np.cumsum(rng.integers(1, 5, size=300))
    frame = pd.DataFrame(
        {
            "day": [base + pd.Timedelta(days=int(x)) for x in offsets],
            "units": rng.normal(100.0, 8.0, 300),
        }
    )
    result = _run(frame)
    assert "time_series_irregular_spacing" in _kinds(result)
    assert any("not evenly spaced" in title for title in _titles(result))


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


def test_analysis_does_not_mutate_the_caller_frame() -> None:
    frame = daily_orders()
    before = frame.copy(deep=True)
    pe.load({"daily_orders": frame}).time_series("orders", entity_id="store")
    pd.testing.assert_frame_equal(frame, before)


def test_repeated_runs_are_byte_identical() -> None:
    first = _run(daily_orders(), "orders", entity_id="store", horizon=28)
    second = _run(daily_orders(), "orders", entity_id="store", horizon=28)
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    assert first.to_dict() == second.to_dict()


def test_interpolation_is_disclosed_as_sampling() -> None:
    frame = _clean_series()
    frame = frame[~frame["day"].isin(pd.date_range("2024-06-01", periods=5, freq="D"))]
    result = _run(frame)
    assert any(
        record.operation == "time_series_regularization" for record in result.sampling
    )
    assert any(
        warning.code == "time_series_interpolated_for_analysis"
        for warning in result.warnings
    )
    decomposition = _evidence(result, "time_series_decomposition")
    assert decomposition.value["computed_on_interpolated_series"]


def test_timestamp_is_inferred_when_unambiguous() -> None:
    result = _run(_clean_series())
    assert result.metadata["timestamp"] == "day"


def test_two_datetime_columns_ask_which_one() -> None:
    frame = _clean_series()
    frame["shipped"] = frame["day"] + pd.Timedelta(days=2)
    result = _run(frame)
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert any(
        warning.code == "time_series_timestamp_ambiguous" for warning in result.warnings
    )


def test_explicit_timestamp_resolves_the_ambiguity() -> None:
    frame = _clean_series()
    frame["shipped"] = frame["day"] + pd.Timedelta(days=2)
    result = _run(frame, timestamp="day")
    assert result.status in {
        AnalysisStatus.COMPLETED,
        AnalysisStatus.COMPLETED_WITH_WARNINGS,
    }


def test_missing_value_column_is_insufficient_evidence() -> None:
    result = _run(_clean_series(), value="nope")
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert any(
        warning.code == "time_series_value_not_found" for warning in result.warnings
    )


def test_text_value_column_is_refused() -> None:
    frame = _clean_series()
    frame["label"] = "x"
    result = _run(frame, value="label")
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert any(
        warning.code == "time_series_value_not_numeric" for warning in result.warnings
    )


def test_table_without_a_datetime_column_is_refused() -> None:
    frame = pd.DataFrame({"a": range(50), "units": range(50)})
    result = _run(frame)
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert any(
        warning.code == "time_series_timestamp_not_found" for warning in result.warnings
    )


def test_too_few_observations_is_insufficient_evidence() -> None:
    frame = _clean_series(rows=8)
    result = _run(frame)
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(TypeError):
        pe.load({"t": _clean_series(rows=60)}).analyze(
            "time_series", value="units", nonsense=1
        )


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------


def test_exports_round_trip(tmp_path) -> None:
    result = _run(daily_orders(), "orders", entity_id="store", horizon=28)
    html = result.to_html(tmp_path / "series.html")
    payload = result.to_json(tmp_path / "series.json")
    assert html.exists() and payload.exists()
    text = html.read_text(encoding="utf-8")
    assert "Time series" in text
    assert "Coverage" in text
    assert "http://" not in text.replace("http://www.w3.org", "")


def test_report_names_the_time_series_sections() -> None:
    text = _run(daily_orders_single(), "orders").render_html()
    for anchor in ("section-series", "section-structure", "section-memory"):
        assert f'id="{anchor}"' in text


def test_api_and_dataset_entry_points_agree() -> None:
    frame = daily_orders_single()
    direct = pe.time_series({"t": frame}, "orders", horizon=14)
    session = pe.load({"t": frame}).time_series("orders", horizon=14)
    assert direct.summary == session.summary


def test_context_supplies_the_timestamp_and_entity() -> None:
    result = pe.load({"t": daily_orders()}).time_series(
        "orders", context={"timestamp": "order_date", "entity_id": "store"}
    )
    assert result.metadata["timestamp"] == "order_date"
    assert result.metadata["entity_id"] == "store"
