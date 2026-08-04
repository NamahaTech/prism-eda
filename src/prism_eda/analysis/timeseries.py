"""Goal-aware deterministic diagnostics for time-series EDA.

The recipe answers two questions at once — *can this be forecast* and *what is
actually in it* — and keeps them apart in the report, because they have
different consequences. A nine-day outage is a problem to fix before modelling.
A strong weekly cycle is not a problem at all; it is the most useful thing in
the series.

That split is the same issue-versus-alert decision the profile and regression
recipes already make, applied to the place it is most often got wrong.
Non-stationarity is the clearest case: nearly every real series with a trend is
non-stationary, so reporting it as a defect would put a finding on almost every
report ever run while telling the reader nothing they can act on. It is an
observation with a modelling consequence — difference it, or pick a model that
handles a trend — and it is filed as one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from prism_eda.analysis._timeseries import (
    MIN_OBSERVATIONS,
    MIN_SEASONAL_CYCLES,
    Frequency,
    RegularSeries,
    analysis_series,
    evidence_by_kind,
    infer_frequency,
    observed_series,
    regular_series,
    resolve_series,
    series_points,
)
from prism_eda.analysis.timeseries_events import (
    CHANGE_POINT_GUARD_CYCLES,
    change_point_evidence,
    intermittency_evidence,
    outlier_evidence,
)
from prism_eda.analysis.timeseries_index import (
    composition_evidence,
    duplicate_evidence,
    gap_evidence,
    index_evidence,
    irregularity_evidence,
    panel_evidence,
)
from prism_eda.analysis.timeseries_structure import (
    SEASONAL_STRONG,
    autocorrelation_evidence,
    decomposition_evidence,
    stationarity_evidence,
)
from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import (
    OBSERVATION,
    Evidence,
    EvidenceScope,
    Finding,
    sort_findings,
    split_findings,
)
from prism_eda.results import (
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)
from prism_eda.transformations.models import TransformationPlan, TransformationStep

#: An unrecorded run this long is an outage rather than a stray missing reading.
OUTAGE_BLOCK = 3

#: Share of periods missing above which coverage itself is the problem.
COVERAGE_HIGH = 0.1
COVERAGE_MEDIUM = 0.02

#: Cycles of history a forecast horizon wants behind it. Below the first, the
#: series cannot show a seasonal pattern the horizon will run through.
HISTORY_CYCLES_MIN = 2
HISTORY_CYCLES_COMFORTABLE = 3

#: History-to-horizon ratio below which the horizon is long relative to what is
#: known. Forecasting a year from two years of history is a stretch; forecasting
#: a year from six months is not forecasting.
HORIZON_RATIO_MIN = 3.0

#: Lags searched when relating another column to the target series.
MAX_CROSS_LAG = 14
CROSS_CORRELATION_MIN = 0.3
MAX_CROSS_COLUMNS = 8

#: Backtest folds recommended at most.
MAX_BACKTEST_FOLDS = 5

#: Titles for the four-way stationarity outcome. Each names the *consequence*
#: rather than the test result, because "non-stationary" alone tells a reader
#: nothing they can act on and de-trending and differencing are different fixes.
_STATIONARITY_TITLES = {
    "stationary": "The level is stable over time",
    "non_stationary": "The level moves over time",
    "trend_stationary": "Stable around a trend — de-trend rather than difference",
    "difference_stationary": "Stable around a shifting level — difference it",
}


def _history_evidence(
    series: pd.Series,
    frequency: Frequency,
    *,
    table: str,
    value: str,
    horizon: int | None,
) -> Evidence:
    """Whether there is enough history to forecast the requested distance."""
    period = frequency.seasonal_period
    observations = int(len(series))
    cycles = (observations / period) if period else None

    ratio = (observations / horizon) if horizon else None
    horizon_cycles = (horizon / period) if (horizon and period) else None
    adequate = True
    reasons: list[str] = []
    if period and cycles is not None and cycles < HISTORY_CYCLES_MIN:
        adequate = False
        reasons.append(
            f"only {cycles:.1f} complete {frequency.seasonal_label} cycle(s) of "
            f"history; a seasonal pattern needs at least {HISTORY_CYCLES_MIN}"
        )
    if horizon and ratio is not None and ratio < HORIZON_RATIO_MIN:
        adequate = False
        reasons.append(
            f"history is only {ratio:.1f}x the requested horizon of {horizon} "
            f"{frequency.label} period(s)"
        )
    if horizon and horizon_cycles is not None and horizon_cycles > (cycles or 0):
        adequate = False
        reasons.append(
            "the horizon spans more seasonal cycles than the history contains"
        )

    return Evidence.create(
        kind="time_series_history",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "observation_count": observations,
            "frequency_label": frequency.label,
            "seasonal_period": period,
            "seasonal_label": frequency.seasonal_label,
            "complete_cycles": cycles,
            "horizon": horizon,
            "history_to_horizon_ratio": ratio,
            "horizon_in_cycles": horizon_cycles,
            "adequate": adequate,
            "reasons": reasons,
            "comfortable_cycles": HISTORY_CYCLES_COMFORTABLE,
        },
        method="history_versus_horizon_v1",
        description=f"History adequacy for {table}.{value}.",
        confidence=0.9,
        assumptions=(
            "Adequacy is measured in seasonal cycles and in multiples of the "
            "requested horizon, not in rows.",
            "Enough history is necessary for a forecast, never sufficient.",
        ),
    )


def _validation_plan_evidence(
    series: pd.Series,
    frequency: Frequency,
    *,
    table: str,
    value: str,
    horizon: int | None,
) -> Evidence:
    """A backtest that respects time order, sized to this series."""
    period = frequency.seasonal_period or 1
    observations = len(series)
    step = horizon or max(period, observations // 10 or 1)
    minimum_train = max(period * MIN_SEASONAL_CYCLES, observations // 3)
    available = observations - minimum_train
    folds = int(max(0, available // step)) if step else 0
    folds = min(folds, MAX_BACKTEST_FOLDS)

    return Evidence.create(
        kind="time_series_validation_plan",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "strategy": "expanding_window_backtest",
            "fold_count": folds,
            "test_periods_per_fold": step,
            "minimum_training_periods": minimum_train,
            "observation_count": observations,
            "first_test_start": (
                series.index[minimum_train].isoformat()
                if minimum_train < observations
                else None
            ),
            "frequency_label": frequency.label,
            "feasible": folds >= 1,
        },
        method="expanding_window_backtest_plan_v1",
        description=f"Temporal validation plan for {table}.{value}.",
        confidence=0.86,
        assumptions=(
            "A random train/test split on a time series trains on the future and "
            "reports an accuracy that cannot be reproduced in production.",
            "Each fold trains on everything before its test window and nothing "
            "after it.",
        ),
    )


def _cross_correlation_evidence(
    frame: pd.DataFrame,
    aligned: pd.Series,
    frequency: Frequency,
    *,
    table: str,
    timestamp: str,
    value: str,
    entity_id: str | None,
) -> Evidence | None:
    """Other columns that lead the series, restricted to lags a forecast can use.

    Directionality is the whole point. A column measured at the same instant as
    the target is not a predictor unless it is known in advance, and a column
    that only correlates at lag zero is very often the target wearing another
    name. Only strictly past lags are offered as usable.
    """
    numeric = [
        column
        for column in frame.columns
        if column not in {timestamp, value, entity_id}
        and ptypes.is_numeric_dtype(frame[column].dtype)
        and not ptypes.is_bool_dtype(frame[column].dtype)
    ][:MAX_CROSS_COLUMNS]
    if not numeric:
        return None

    times = pd.to_datetime(frame[timestamp], errors="coerce")
    rows: list[dict[str, Any]] = []
    for column in numeric:
        other = pd.Series(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(times),
        )
        other = other[other.index.notna()].groupby(level=0).mean()
        other = other.reindex(aligned.index).interpolate(limit_direction="both")
        if other.notna().sum() < MIN_OBSERVATIONS or other.nunique() < 2:
            continue
        best: dict[str, Any] | None = None
        for lag in range(0, min(MAX_CROSS_LAG, len(aligned) // 4) + 1):
            shifted = other.shift(lag)
            pair = pd.DataFrame({"a": aligned, "b": shifted}).dropna()
            if len(pair) < MIN_OBSERVATIONS:
                continue
            correlation = pair["a"].corr(pair["b"])
            if pd.isna(correlation):
                continue
            if best is None or abs(correlation) > abs(best["correlation"]):
                best = {"lag": lag, "correlation": float(correlation)}
        if best is None or abs(best["correlation"]) < CROSS_CORRELATION_MIN:
            continue
        rows.append(
            {
                "column": column,
                "best_lag": best["lag"],
                "correlation": best["correlation"],
                "usable_for_forecasting": best["lag"] >= 1,
                "note": (
                    "Leads the series, so its past values are usable as features."
                    if best["lag"] >= 1
                    else "Moves at the same time, so it is only usable if it is "
                    "known in advance."
                ),
            }
        )
    if not rows:
        return None
    rows.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return Evidence.create(
        kind="time_series_cross_correlation",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "columns": rows,
            "max_lag_searched": min(MAX_CROSS_LAG, len(aligned) // 4),
            "usable_count": sum(1 for item in rows if item["usable_for_forecasting"]),
            "frequency_label": frequency.label,
        },
        method="lagged_cross_correlation_v1",
        description=f"Columns related to {table}.{value} at a lag.",
        confidence=0.72,
        assumptions=(
            "Correlation at a lag is not causation, and a shared trend produces "
            "high lagged correlation between series that have nothing to do with "
            "each other.",
            "Only strictly positive lags are usable at forecast time; a lag-zero "
            "relationship needs the other column to be known in advance.",
        ),
    )


def _findings_and_steps(
    evidence: list[Evidence],
) -> tuple[list[Finding], list[TransformationStep]]:
    """Turn evidence into a short, prioritized, decision-first list."""
    findings: list[Finding] = []
    steps: list[TransformationStep] = []

    for item in evidence:
        value = item.value
        table = item.scope.table or ""

        if item.kind == "time_series_index" and value["future_timestamp_count"]:
            findings.append(
                Finding.create(
                    title="Timestamps in the future",
                    summary=(
                        f"{value['future_timestamp_count']:,} row(s) are dated "
                        "after the current time."
                    ),
                    severity="medium",
                    confidence=1.0,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Check the clock and timezone of whatever wrote these "
                        "rows before training on them."
                    ),
                )
            )

        elif item.kind == "time_series_duplicate_timestamps":
            conflicting = value["conflicting_value_count"]
            findings.append(
                Finding.create(
                    title="Duplicate timestamps in the series",
                    summary=(
                        f"{value['duplicated_row_count']:,} row(s) across "
                        f"{value['duplicated_timestamp_count']:,} timestamp(s) "
                        "are recorded more than once for the same series"
                        + (
                            f", and {conflicting} of them disagree about the value."
                            if conflicting
                            else "."
                        )
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Decide whether these are repeated readings or genuine "
                        "separate events. Any resample silently averages or sums "
                        "them until you do."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="resolve_duplicate_timestamps",
                    table=table,
                    columns=item.scope.columns,
                    parameters={
                        "duplicated_timestamp_count": value[
                            "duplicated_timestamp_count"
                        ]
                    },
                    rationale=(
                        "A duplicated timestamp makes the series ambiguous at that "
                        "point in time."
                    ),
                    evidence_ids=(item.id,),
                    risk="high",
                )
            )

        elif item.kind == "time_series_gaps":
            longest = value["longest_unrecorded_block"]
            rate = value["unrecorded_period_rate"]
            if value["unrecorded_period_count"]:
                severity = (
                    "high"
                    if longest >= OUTAGE_BLOCK or rate > COVERAGE_HIGH
                    else "medium"
                    if rate > COVERAGE_MEDIUM
                    else "low"
                )
                blocks = value["unrecorded_blocks"]
                block = blocks[0] if blocks else None
                where = (
                    f" The longest runs {longest} {value['frequency_label']} "
                    f"period(s) from {block['start'][:10]}."
                    if block
                    else ""
                )
                findings.append(
                    Finding.create(
                        title=(
                            f"{value['unrecorded_period_count']:,} period(s) were "
                            "never recorded"
                        ),
                        summary=(
                            f"{value['unrecorded_block_count']} block(s) of the "
                            f"time axis have no row at all ({rate:.1%} of the "
                            f"span).{where}"
                        ),
                        severity=severity,
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "A run of absent rows is a collection failure, not a "
                            "quiet period. Confirm which before filling it."
                        ),
                    )
                )
                steps.append(
                    TransformationStep(
                        operation="review_unrecorded_periods",
                        table=table,
                        columns=item.scope.columns,
                        parameters={"blocks": value["unrecorded_blocks"]},
                        rationale=(
                            "Interpolating across a collection failure invents "
                            "demand that was never observed."
                        ),
                        evidence_ids=(item.id,),
                        risk="high",
                    )
                )
            if value["blank_period_count"]:
                findings.append(
                    Finding.create(
                        title=(
                            f"{value['blank_period_count']:,} period(s) recorded "
                            "with no value"
                        ),
                        summary=(
                            f"{value['blank_block_count']} block(s) have a row but "
                            "a blank value — a measurement failure rather than a "
                            "collection failure."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "These rows were written, so the pipeline ran; the "
                            "value did not arrive. That is a different fault from "
                            "an absent row."
                        ),
                    )
                )

        elif item.kind == "time_series_irregular_spacing":
            findings.append(
                Finding.create(
                    title="Observations are not evenly spaced",
                    summary=(
                        f"Only {value['on_grid_rate']:.0%} of gaps match the "
                        f"dominant {value['frequency_label']} spacing, across "
                        f"{value['distinct_gap_count']:,} distinct gap length(s)."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Every structural result below is computed on a "
                        "regularized reconstruction. Decide whether that "
                        "reconstruction is faithful before relying on them."
                    ),
                )
            )

        elif item.kind == "time_series_panel_coverage":
            imbalanced = value["imbalanced_entities"]
            if imbalanced:
                worst = imbalanced[0]
                findings.append(
                    Finding.create(
                        title=(
                            f"{value['imbalanced_count']} entit"
                            + (
                                "y has"
                                if value["imbalanced_count"] == 1
                                else "ies have"
                            )
                            + " far less history than the rest"
                        ),
                        summary=(
                            f"{worst['entity']} has {worst['period_count']:,} "
                            f"period(s), {worst['history_share']:.0%} of the "
                            f"longest entity's history."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "A model pooled across the panel will be dominated by "
                            "the long entities. Forecast the short ones separately "
                            "or accept that they are being extrapolated."
                        ),
                    )
                )
            if value["short_history_count"]:
                findings.append(
                    Finding.create(
                        title=(
                            f"{value['short_history_count']} entit"
                            f"{'y' if value['short_history_count'] == 1 else 'ies'}"
                            " cannot show a seasonal cycle"
                        ),
                        summary=(
                            "Their history is shorter than "
                            f"{value['required_periods_for_seasonality']} period(s), "
                            f"so no {value['seasonal_label']} pattern is observable "
                            "for them."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Borrow the seasonal shape from the panel rather than "
                            "estimating it per entity."
                        ),
                    )
                )

        elif item.kind == "time_series_panel_composition":
            first = value["changes"][0] if value["changes"] else None
            when = f" first at {first['timestamp'][:10]}." if first else "."
            findings.append(
                Finding.create(
                    title="The aggregate is not made of the same series throughout",
                    summary=(
                        f"The number of contributing {value['entity_id']} values "
                        f"moves between {value['min_active_entities']} and "
                        f"{value['max_active_entities']} across the history, "
                        f"changing {value['change_count']} time(s){when}"
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Any level shift found in the total may be an entity "
                        "joining or leaving rather than a change in demand. "
                        "Restrict the aggregate to a window where the membership "
                        "is constant, or forecast the entities separately."
                    ),
                )
            )

        elif item.kind == "time_series_history" and not value["adequate"]:
            findings.append(
                Finding.create(
                    title="Not enough history for the requested forecast",
                    summary="; ".join(value["reasons"]).capitalize() + ".",
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Shorten the horizon, gather more history, or forecast "
                        "without a seasonal component and say so."
                    ),
                )
            )

        elif item.kind == "time_series_validation_plan" and not value["feasible"]:
            findings.append(
                Finding.create(
                    title="Too short to backtest",
                    summary=(
                        f"{value['observation_count']:,} period(s) leave no room "
                        "for a training window plus even one test fold."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Any accuracy figure from this series is unvalidated. "
                        "Treat a forecast from it as a guess with a number "
                        "attached."
                    ),
                )
            )

        # ---- Observations: true, useful, and not defects ------------------
        elif item.kind == "time_series_decomposition":
            if value["has_seasonality"]:
                findings.append(
                    Finding.create(
                        title=(
                            f"Clear {value['seasonal_label']} seasonality"
                            if value["seasonal_strength"] >= SEASONAL_STRONG
                            else f"Some {value['seasonal_label']} seasonality"
                        ),
                        summary=(
                            f"A {value['seasonal_label']} cycle explains "
                            f"{value['seasonal_strength']:.0%} of what is left "
                            "after the trend, with a peak-to-trough swing of "
                            f"{value['seasonal_peak_to_trough']:,.4g}."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Use a model with a seasonal term, and never validate "
                            "on a window shorter than one full cycle."
                        ),
                        category=OBSERVATION,
                    )
                )
            if value["has_trend"]:
                findings.append(
                    Finding.create(
                        title=f"The level is {value['trend_direction']}",
                        summary=(
                            f"Trend explains {value['trend_strength']:.0%} of the "
                            "non-seasonal variation, moving the level by "
                            f"{value['trend_change']:,.4g} across the history."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "A model without a trend term will drift steadily "
                            "wrong as the horizon lengthens."
                        ),
                        category=OBSERVATION,
                    )
                )

        elif item.kind == "time_series_stationarity":
            findings.append(
                Finding.create(
                    title=_STATIONARITY_TITLES.get(
                        value["verdict"], "Stationarity is unclear"
                    ),
                    summary=(
                        value["explanation"]
                        + (
                            ""
                            if value["tests_agree"]
                            else " The two tests disagree, which is itself the "
                            "informative result here."
                        )
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Non-stationarity is a property of the series, not a "
                        "defect — most series with a trend are non-stationary."
                    ),
                    category=OBSERVATION,
                )
            )

        elif item.kind == "time_series_change_points":
            if value["level_shifts"]:
                latest = value["level_shifts"][-1]
                relative = latest["relative_change"]
                move = f" ({relative:+.0%})" if relative is not None else ""
                findings.append(
                    Finding.create(
                        title=(
                            f"{value['level_shift_count']} level shift"
                            f"{'' if value['level_shift_count'] == 1 else 's'} in "
                            "the series"
                        ),
                        summary=(
                            f"The most recent is {latest['timestamp'][:10]}, where "
                            f"the level stepped {latest['direction']}{move} and "
                            "stayed there."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Training across a real level shift teaches a model "
                            "the average of two regimes that never existed. "
                            "Consider training only on the most recent one."
                        ),
                        category=OBSERVATION,
                    )
                )
            if value["variance_shifts"]:
                latest = value["variance_shifts"][-1]
                findings.append(
                    Finding.create(
                        title="The series became harder to predict",
                        summary=(
                            f"Around {latest['timestamp'][:10]} the spread went "
                            f"{latest['direction']}, from "
                            f"{latest['std_before']:,.4g} to "
                            f"{latest['std_after']:,.4g}."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Prediction intervals fitted on the whole history will "
                            "be wrong on both sides of this point."
                        ),
                        category=OBSERVATION,
                    )
                )

        elif item.kind == "time_series_outliers":
            if value["suppressed_as_changing_spread"]:
                findings.append(
                    Finding.create(
                        title="Spread changes too much for a single outlier rule",
                        summary=(
                            f"{value['outlier_rate']:.0%} of periods fall outside "
                            "a fence built from the whole history, which means the "
                            "spread is not constant rather than that the series is "
                            "full of anomalies."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Read the variance-shift finding instead; a single "
                            "outlier threshold is the wrong instrument here."
                        ),
                        category=OBSERVATION,
                    )
                )
            elif value["outliers"]:
                top = value["outliers"][0]
                findings.append(
                    Finding.create(
                        title=(
                            f"{value['outlier_count']} one-off "
                            f"{'spike' if value['outlier_count'] == 1 else 'spikes'} "
                            "in the series"
                        ),
                        summary=(
                            f"The largest is {top['timestamp'][:10]}, "
                            f"{top['direction']} and well outside the local range."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Decide per point whether it was a real event worth "
                            "modelling or an error worth removing. They look "
                            "identical here."
                        ),
                        category=OBSERVATION,
                    )
                )

        elif item.kind == "time_series_intermittency":
            findings.append(
                Finding.create(
                    title=f"Demand pattern is {value['pattern']}",
                    summary=(
                        f"{value['zero_rate']:.0%} of periods are zero, with runs "
                        f"up to {value['longest_zero_run']} period(s) long."
                    ),
                    severity="medium" if value["pattern"] == "lumpy" else "low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Squared-error forecasting optimizes toward an average "
                        "this series never takes. Consider Croston-style methods "
                        "or forecasting the interval and the size separately."
                    ),
                    category=OBSERVATION,
                )
            )

        elif item.kind == "time_series_cross_correlation" and value["usable_count"]:
            usable = [
                entry for entry in value["columns"] if entry["usable_for_forecasting"]
            ]
            top = usable[0]
            findings.append(
                Finding.create(
                    title=f"{top['column']} leads the series by {top['best_lag']}",
                    summary=(
                        f"Its value {top['best_lag']} period(s) earlier correlates "
                        f"{top['correlation']:+.2f} with {value['value']}."
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "A lagged predictor is usable at forecast time. Confirm it "
                        "is not simply sharing a trend."
                    ),
                    category=OBSERVATION,
                )
            )

    return sort_findings(findings), steps


def _artifacts(evidence: list[Evidence]) -> tuple[Artifact, ...]:
    artifacts: list[Artifact] = []

    panel = evidence_by_kind(evidence, "time_series_panel_coverage")
    if panel is not None:
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Coverage by entity",
                data={
                    "columns": [
                        {"key": "entity", "label": "Entity"},
                        {"key": "periods", "label": "Periods"},
                        {"key": "share", "label": "Share of longest"},
                        {"key": "start", "label": "First"},
                        {"key": "end", "label": "Last"},
                    ],
                    "rows": [
                        {
                            "entity": row["entity"],
                            "periods": f"{row['period_count']:,}",
                            "share": f"{row['history_share']:.0%}",
                            "start": row["start"][:10],
                            "end": row["end"][:10],
                        }
                        for row in panel.value["entities"]
                    ],
                },
                evidence_ids=(panel.id,),
                metadata={
                    "description": (
                        "History per entity. A panel is almost never balanced, and "
                        "the aggregate hides which entities cannot be forecast."
                    )
                },
            )
        )

    plan = evidence_by_kind(evidence, "time_series_validation_plan")
    history = evidence_by_kind(evidence, "time_series_history")
    if plan is not None:
        rows = [
            {"item": "Strategy", "detail": "Expanding-window backtest"},
            {"item": "Folds", "detail": str(plan.value["fold_count"])},
            {
                "item": "Test window per fold",
                "detail": (
                    f"{plan.value['test_periods_per_fold']} "
                    f"{plan.value['frequency_label']} period(s)"
                ),
            },
            {
                "item": "Minimum training window",
                "detail": f"{plan.value['minimum_training_periods']:,} period(s)",
            },
        ]
        if plan.value["first_test_start"]:
            rows.append(
                {
                    "item": "First test window starts",
                    "detail": plan.value["first_test_start"][:10],
                }
            )
        if history is not None and history.value["horizon"]:
            rows.append(
                {
                    "item": "Requested horizon",
                    "detail": (
                        f"{history.value['horizon']} "
                        f"{history.value['frequency_label']} period(s)"
                    ),
                }
            )
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Validation plan",
                data={
                    "columns": [
                        {"key": "item", "label": "Item"},
                        {"key": "detail", "label": "Detail"},
                    ],
                    "rows": rows,
                },
                evidence_ids=(plan.id,),
                metadata={
                    "description": (
                        "A random split trains on the future. These folds keep "
                        "time order, so the reported accuracy is one production "
                        "could reproduce."
                    )
                },
            )
        )
    return tuple(artifacts)


def _summary(
    table: str,
    value: str,
    findings: list[Finding],
    *,
    has_warnings: bool,
) -> str:
    suffix = " Sampling or recoverable caveats apply." if has_warnings else ""
    issues, observations = split_findings(findings)
    alerts = (
        f" {len(observations)} alert(s) describe what is in the series."
        if observations
        else ""
    )
    if not issues:
        return (
            f"{table}.{value} looks forecastable: no blocking time-axis problems "
            f"were found.{alerts}{suffix}"
        )
    order = ("critical", "high", "medium", "low")
    counts = dict.fromkeys(order, 0)
    for finding in issues:
        if finding.severity in counts:
            counts[finding.severity] += 1
    breakdown = ", ".join(f"{counts[key]} {key}" for key in order if counts[key])
    top = issues[0]
    lead = (
        "not ready to forecast"
        if top.severity in {"critical", "high"}
        else "review before forecasting"
    )
    return (
        f"{table}.{value}: {lead}. Top issue — {top.title}. "
        f"{len(issues)} prioritized issue(s) ({breakdown}).{alerts}{suffix}"
    )


def time_series_dataset(
    tables: Mapping[str, pd.DataFrame],
    catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
    value: str | None,
    timestamp: str | None = None,
    entity_id: str | None = None,
    horizon: int | None = None,
    table: str | None = None,
    callbacks: tuple[EventCallback, ...] = (),
) -> AnalysisResult:
    """Run deterministic time-series diagnostics for one value column."""
    emit(
        callbacks,
        Event(
            EventKind.RUN_STARTED,
            "Time-series diagnostics started.",
            stage="time_series",
        ),
    )
    warnings: list[AnalysisWarning] = []
    sampling: list[SamplingRecord] = []
    resolved_entity = entity_id if entity_id is not None else context.entity_id
    table_catalog, resolved_timestamp, resolved_value, resolved_entity = resolve_series(
        catalog,
        tables,
        table=table,
        timestamp=timestamp or context.timestamp,
        value=value,
        entity_id=resolved_entity,
        warnings=warnings,
    )

    evidence: list[Evidence] = []
    findings: list[Finding] = []
    steps: list[TransformationStep] = []
    usable = False
    frequency: Frequency | None = None

    if (
        table_catalog is not None
        and resolved_timestamp is not None
        and resolved_value is not None
    ):
        frame = tables[table_catalog.name]
        frequency = infer_frequency(frame[resolved_timestamp])
        observed = observed_series(frame, resolved_timestamp, resolved_value)

        evidence.append(
            index_evidence(
                observed,
                frequency,
                table=table_catalog.name,
                timestamp=resolved_timestamp,
                value=resolved_value,
            )
        )
        duplicates = duplicate_evidence(
            frame,
            table=table_catalog.name,
            timestamp=resolved_timestamp,
            value=resolved_value,
            entity_id=resolved_entity,
        )
        if duplicates is not None:
            evidence.append(duplicates)

        if resolved_entity is not None and frequency is not None:
            panel = panel_evidence(
                frame,
                frequency,
                table=table_catalog.name,
                timestamp=resolved_timestamp,
                value=resolved_value,
                entity_id=resolved_entity,
            )
            if panel is not None:
                evidence.append(panel)
            composition = composition_evidence(
                frame,
                table=table_catalog.name,
                timestamp=resolved_timestamp,
                value=resolved_value,
                entity_id=resolved_entity,
            )
            if composition is not None:
                evidence.append(composition)
            # Structural analysis runs on the aggregate: bounded cost, and it
            # answers "is this panel forecastable" without emitting one report
            # per entity.
            observed = observed.groupby(level=0).sum(min_count=1)
            warnings.append(
                AnalysisWarning(
                    code="time_series_panel_aggregated",
                    message=(
                        f"{resolved_entity} identifies "
                        f"{frame[resolved_entity].nunique():,} series; trend, "
                        "seasonality, and stationarity describe their total. "
                        "Coverage is reported per entity."
                    ),
                    table=table_catalog.name,
                    column=resolved_entity,
                )
            )

        regular: RegularSeries | None = (
            regular_series(observed, frequency) if frequency is not None else None
        )
        if regular is not None and frequency is not None:
            gaps = gap_evidence(
                observed,
                regular,
                table=table_catalog.name,
                timestamp=resolved_timestamp,
                value=resolved_value,
            )
            if gaps is not None:
                evidence.append(gaps)
            irregular = irregularity_evidence(
                observed,
                frequency,
                table=table_catalog.name,
                timestamp=resolved_timestamp,
            )
            if irregular is not None:
                evidence.append(irregular)

            interpolated = regular.missing_periods > 0
            analysis = analysis_series(
                regular,
                warnings=warnings,
                sampling=sampling,
                table=table_catalog.name,
                column=resolved_value,
            )
            if analysis is not None and len(analysis) >= MIN_OBSERVATIONS:
                usable = True
                evidence.append(
                    _history_evidence(
                        analysis,
                        frequency,
                        table=table_catalog.name,
                        value=resolved_value,
                        horizon=horizon,
                    )
                )
                decomposition = decomposition_evidence(
                    analysis,
                    frequency,
                    table=table_catalog.name,
                    value=resolved_value,
                    interpolated=interpolated,
                )
                adjusted = analysis
                remainder: pd.Series | None = None
                if decomposition is not None:
                    evidence.append(decomposition)
                    from statsmodels.tsa.seasonal import STL

                    fitted = STL(
                        analysis, period=frequency.seasonal_period, robust=True
                    ).fit()
                    adjusted = analysis - pd.Series(
                        fitted.seasonal, index=analysis.index
                    )
                    remainder = pd.Series(fitted.resid, index=analysis.index)

                autocorrelation = autocorrelation_evidence(
                    analysis,
                    frequency,
                    table=table_catalog.name,
                    value=resolved_value,
                )
                if autocorrelation is not None:
                    evidence.append(autocorrelation)
                stationarity = stationarity_evidence(
                    analysis, table=table_catalog.name, value=resolved_value
                )
                if stationarity is not None:
                    evidence.append(stationarity)

                change_points: list[pd.Timestamp] = []
                if remainder is not None:
                    noise = float(remainder.std(ddof=1) or 0.0)
                    changes = change_point_evidence(
                        adjusted,
                        frequency,
                        table=table_catalog.name,
                        value=resolved_value,
                        noise_scale=noise,
                    )
                    if changes is not None:
                        evidence.append(changes)
                        change_points = [
                            pd.Timestamp(entry["timestamp"])
                            for entry in changes.value["level_shifts"]
                        ]
                    outliers = outlier_evidence(
                        remainder,
                        analysis,
                        table=table_catalog.name,
                        value=resolved_value,
                        recorded=regular.values.notna(),
                        change_points=change_points,
                        guard_periods=(frequency.seasonal_period or 1)
                        * CHANGE_POINT_GUARD_CYCLES,
                    )
                    if outliers is not None:
                        evidence.append(outliers)

                intermittency = intermittency_evidence(
                    analysis, table=table_catalog.name, value=resolved_value
                )
                if intermittency is not None:
                    evidence.append(intermittency)

                if resolved_entity is None:
                    cross = _cross_correlation_evidence(
                        frame,
                        analysis,
                        frequency,
                        table=table_catalog.name,
                        timestamp=resolved_timestamp,
                        value=resolved_value,
                        entity_id=resolved_entity,
                    )
                    if cross is not None:
                        evidence.append(cross)

                evidence.append(
                    _validation_plan_evidence(
                        analysis,
                        frequency,
                        table=table_catalog.name,
                        value=resolved_value,
                        horizon=horizon,
                    )
                )
                evidence.append(
                    Evidence.create(
                        kind="time_series_chart",
                        scope=EvidenceScope(
                            table=table_catalog.name, columns=(resolved_value,)
                        ),
                        value={
                            "value": resolved_value,
                            "points": series_points(analysis),
                            "recorded_points": series_points(regular.values),
                            "frequency_label": frequency.label,
                            "change_points": [
                                stamp.isoformat() for stamp in change_points
                            ],
                        },
                        method="series_chart_data_v1",
                        description=(
                            f"Chart data for {table_catalog.name}.{resolved_value}."
                        ),
                        confidence=1.0,
                    )
                )
        elif frequency is None:
            warnings.append(
                AnalysisWarning(
                    code="time_series_frequency_not_inferable",
                    message=(
                        "The spacing between observations is too irregular or too "
                        "sparse to infer a frequency, so no structural analysis "
                        "was run."
                    ),
                    table=table_catalog.name,
                    column=resolved_timestamp,
                )
            )
        findings, steps = _findings_and_steps(evidence)

    for item in evidence:
        emit(
            callbacks,
            Event(
                EventKind.EVIDENCE_CREATED,
                item.description,
                stage="evidence",
                data={"evidence_id": item.id, "kind": item.kind},
            ),
        )

    if table_catalog is None or resolved_timestamp is None or resolved_value is None:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = (
            "Time-series analysis needs a numeric value column and a time column "
            "in one table."
        )
    elif not usable:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = (
            f"{table_catalog.name}.{resolved_value} does not have enough regularly "
            f"spaced observations for time-series diagnostics (at least "
            f"{MIN_OBSERVATIONS} are needed)."
        )
    elif warnings:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        summary = _summary(
            table_catalog.name, resolved_value, findings, has_warnings=True
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = _summary(
            table_catalog.name, resolved_value, findings, has_warnings=False
        )

    result = AnalysisResult(
        goal="time_series",
        status=status,
        summary=summary,
        catalog=catalog,
        findings=tuple(findings),
        evidence=tuple(evidence),
        artifacts=_artifacts(evidence),
        assumptions=context.assumptions,
        warnings=tuple(warnings),
        sampling=tuple(sampling),
        transformation_plan=TransformationPlan(tuple(steps)),
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
            "selected_table": table_catalog.name if table_catalog else table,
            "timestamp": resolved_timestamp,
            "value": resolved_value,
            "entity_id": resolved_entity,
            "horizon": horizon,
            "frequency": frequency.label if frequency else None,
        },
    )
    emit(
        callbacks,
        Event(
            EventKind.RUN_COMPLETED,
            result.summary,
            stage="time_series",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
