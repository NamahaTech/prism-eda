"""What happened along the way?

Three things that a trend-and-seasonality view averages away: the day the level
permanently moved, the day that was simply unlike its neighbours, and the
possibility that the series is mostly zeros and should not be forecast with
continuous methods at all.

The distinction between the first two carries most of the value. A spike is one
bad day; a change point is every day afterwards. Confusing them is expensive in
both directions — deleting a change point as an outlier throws away the most
important fact about the series, and treating a promotion spike as a regime
change re-baselines a forecast onto a day that will never repeat. So they are
detected separately, by different methods, and reported as different things.

Both run on the **seasonally adjusted** series. Detecting a level shift on raw
data with a weekly cycle finds a "change point" every weekend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from prism_eda.analysis._timeseries import Frequency, to_timestamp
from prism_eda.evidence.models import Evidence, EvidenceScope

#: Change points reported. Beyond a handful the series is better described as
#: drifting than as having discrete regimes.
MAX_CHANGE_POINTS = 5

#: Normalized shift statistic a split must clear. This is a CUSUM-style score in
#: standard-deviation units, so it is comparable across series.
CHANGE_STATISTIC = 5.0

#: And the shift must also be *large*, not merely well-evidenced: a tiny step in
#: a very long, very quiet series clears any statistical bar without mattering.
CHANGE_EFFECT = 0.5

#: The step must clear this many standard deviations of the series' own noise.
#:
#: Without it, binary segmentation chops a slow trend into a staircase of
#: "level shifts" — each one statistically overwhelming, because a long trending
#: series gives any split enormous evidence, and each one meaningless, because a
#: two-percent drift is not a regime change. Measuring the step against the
#: remainder's scale rather than the segment's own spread is what separates the
#: two: a trend inflates the segment spread it is measured against, but not the
#: noise around itself.
CHANGE_NOISE_MULTIPLE = 1.0

#: Tukey fence multiplier for temporal outliers.
#:
#: A z-score is the obvious choice here and the wrong one. A robust STL fit
#: deliberately concentrates the remainder, leaving its MAD several times
#: tighter than its standard deviation, so a MAD-scaled z is inflated by a
#: factor that varies with how robust the fit happened to be — and any fixed
#: threshold then fires on ordinary days in one series and misses real spikes in
#: another. The interquartile fence is scale-free and self-calibrating: at this
#: multiplier a clean seasonal series with a trend yields no outliers at all,
#: while genuine one-off events are still caught comfortably.
OUTLIER_FENCE = 4.5
MAX_OUTLIERS = 15

#: Above this share of flagged points, the series has a changing spread rather
#: than outliers, and a single fence is the wrong instrument. The rate is still
#: reported; the list is not.
OUTLIER_RATE_CEILING = 0.02

#: A variance shift must change the spread by at least this ratio. Rolling
#: dispersion wanders even at constant variance, so segmentation alone finds
#: "shifts" in a perfectly homoscedastic series.
VARIANCE_RATIO = 1.5

#: Seasonal cycles either side of a change point excluded from outlier scoring.
#:
#: STL's trend cannot turn instantly, so it overshoots on both sides of a real
#: step and the remainder spikes there. Those spikes are the change point being
#: detected twice, not separate anomalies.
CHANGE_POINT_GUARD_CYCLES = 1

#: Zero share below which "intermittent demand" is not the right frame and the
#: classification would be noise.
INTERMITTENT_MIN_ZERO_RATE = 0.05

#: Syntetos–Boylan cut-offs for classifying demand patterns.
ADI_CUT = 1.32
CV2_CUT = 0.49


def _binary_segmentation(
    values: np.ndarray, min_segment: int, depth: int = 0
) -> list[tuple[int, float, float]]:
    """Recursively split on the strongest mean shift, deepest-first.

    Returns (position, statistic, effect) for each accepted split. The statistic
    is the standard CUSUM-style score, which weights a split by how much data
    sits on each side, so a shift near the boundary needs to be larger to count.
    """
    if depth >= MAX_CHANGE_POINTS or len(values) < 2 * min_segment:
        return []
    total = len(values)
    spread = float(np.std(values, ddof=1))
    if spread <= 0:
        return []

    positions = np.arange(min_segment, total - min_segment)
    if positions.size == 0:
        return []
    # Prefix sums make every candidate split O(1) instead of O(n).
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    left_n = positions.astype("float64")
    right_n = total - left_n
    left_mean = cumulative[positions] / left_n
    right_mean = (cumulative[total] - cumulative[positions]) / right_n
    difference = np.abs(left_mean - right_mean)
    statistic = difference * np.sqrt(left_n * right_n / total) / spread

    best = int(np.argmax(statistic))
    position = int(positions[best])
    score = float(statistic[best])
    effect = float(difference[best] / spread)
    if score < CHANGE_STATISTIC or effect < CHANGE_EFFECT:
        return []

    found = [(position, score, effect)]
    for offset, segment in (
        (0, values[:position]),
        (position, values[position:]),
    ):
        found.extend(
            (offset + pos, stat, eff)
            for pos, stat, eff in _binary_segmentation(segment, min_segment, depth + 1)
        )
    return found


def change_point_evidence(
    adjusted: pd.Series,
    frequency: Frequency,
    *,
    table: str,
    value: str,
    noise_scale: float,
) -> Evidence | None:
    """Level and variance shifts: the days after which the series is different."""
    period = frequency.seasonal_period or 1
    min_segment = max(2 * period, 10)
    if len(adjusted) < 4 * min_segment:
        return None
    values = adjusted.to_numpy(dtype="float64")

    # Segment on a linearly detrended copy. Steady drift is otherwise chopped
    # into a staircase of "level shifts", each statistically overwhelming and
    # none of them real. A straight line is the right thing to remove here and
    # STL's trend is not: a flexible trend absorbs a genuine step as a fast ramp
    # and the shift disappears along with the false ones.
    #
    # The line has to be fitted robustly. Least squares is pulled by the very
    # features being looked for — on a series with one real step and two spikes
    # it returns a *negative* slope for a series that is genuinely rising, and
    # detrending by that wrong line then hides the step it was meant to expose.
    # Theil-Sen takes the median of pairwise slopes and is unmoved by both.
    steps = np.arange(len(values), dtype="float64")
    fit = stats.theilslopes(values, steps)
    detrended = values - (float(fit[0]) * steps + float(fit[1]))

    splits = sorted(_binary_segmentation(detrended, min_segment))[:MAX_CHANGE_POINTS]
    floor = CHANGE_NOISE_MULTIPLE * noise_scale if noise_scale > 0 else 0.0

    level: list[dict[str, Any]] = []
    for position, statistic, effect in splits:
        before = values[max(0, position - 4 * period) : position]
        after = values[position : position + 4 * period]
        if before.size == 0 or after.size == 0:
            continue
        step = float(abs(after.mean() - before.mean()))
        # The statistical bar is trivially cleared by any split of a long
        # trending series. This is the bar that actually decides.
        if step < floor:
            continue
        level.append(
            {
                "timestamp": adjusted.index[position].isoformat(),
                "statistic": statistic,
                "effect_in_std": effect,
                "step": step,
                "step_in_noise": step / noise_scale if noise_scale > 0 else None,
                "mean_before": float(before.mean()),
                "mean_after": float(after.mean()),
                "direction": "up" if after.mean() > before.mean() else "down",
                "relative_change": (
                    float((after.mean() - before.mean()) / abs(before.mean()))
                    if before.mean()
                    else None
                ),
            }
        )

    # Variance shifts are found the same way, on the rolling dispersion rather
    # than the level: a series can hold its average and become far harder to
    # predict, which changes the interval a forecast should quote.
    window = max(period, 7)
    dispersion = (
        pd.Series(detrended, index=adjusted.index)
        .rolling(window, min_periods=window)
        .std()
        .dropna()
    )
    variance: list[dict[str, Any]] = []
    if len(dispersion) >= 4 * min_segment:
        for position, statistic, effect in sorted(
            _binary_segmentation(dispersion.to_numpy(dtype="float64"), min_segment)
        )[:MAX_CHANGE_POINTS]:
            spread_before = float(dispersion.to_numpy()[:position].mean())
            spread_after = float(dispersion.to_numpy()[position:].mean())
            ratio = spread_after / spread_before if spread_before else None
            if ratio is None or (VARIANCE_RATIO > ratio > 1.0 / VARIANCE_RATIO):
                continue
            variance.append(
                {
                    "timestamp": to_timestamp(dispersion.index[position]).isoformat(),
                    "statistic": statistic,
                    "std_before": spread_before,
                    "std_after": spread_after,
                    "ratio": ratio,
                    "direction": (
                        "wider" if spread_after > spread_before else "tighter"
                    ),
                }
            )

    if not level and not variance:
        return None
    return Evidence.create(
        kind="time_series_change_points",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "level_shifts": level,
            "level_shift_count": len(level),
            "variance_shifts": variance[:MAX_CHANGE_POINTS],
            "variance_shift_count": len(variance),
            "most_recent_level_shift": level[-1]["timestamp"] if level else None,
            "detection_window": min_segment,
            "minimum_step": floor,
        },
        method="binary_segmentation_on_adjusted_series_v2",
        description=f"Level and variance shifts in {table}.{value}.",
        confidence=0.76,
        assumptions=(
            "Detected on the seasonally adjusted series, so a regular weekly or "
            "yearly cycle is not mistaken for a regime change.",
            "A shift must move the level by more than the series' own noise, so "
            "a slow trend is not reported as a staircase of regime changes.",
            "A change point is a candidate boundary, not a confirmed event. It "
            "marks where the series behaves differently, not why.",
            "Training a forecast across a real level shift teaches it an average "
            "of two regimes that never existed.",
        ),
    )


def outlier_evidence(
    remainder: pd.Series,
    observed: pd.Series,
    *,
    table: str,
    value: str,
    recorded: pd.Series | None = None,
    change_points: list[pd.Timestamp] | None = None,
    guard_periods: int = 0,
) -> Evidence | None:
    """Points unlike their neighbours, scored against the local remainder.

    Scored on the STL remainder rather than the raw level, so a busy Saturday in
    a strongly seasonal series is not called an outlier for being busy, and a
    high day early in a rising trend is not called an outlier for being early.

    Two exclusions keep the list honest. Interpolated periods are reconstructions
    Prism itself manufactured to make the decomposition run, and either end of a
    filled gap sits at a discontinuity, so scoring them reliably "discovers"
    anomalies that are pure artifact. And the neighbourhood of a detected change
    point is excluded, because a smooth trend cannot turn instantly and its
    overshoot on both sides of a real step would otherwise be reported as a
    cluster of separate anomalies.
    """
    finite = remainder.dropna()
    if recorded is not None:
        genuine = recorded.reindex(finite.index).fillna(False).astype(bool)
        finite = finite[genuine]
    if change_points and guard_periods > 0:
        keep = pd.Series(True, index=finite.index)
        index = pd.DatetimeIndex(finite.index)
        for change in change_points:
            found = index.get_indexer(pd.DatetimeIndex([change]), method="nearest")
            if len(found) == 0 or found[0] < 0:
                continue
            low = max(0, int(found[0]) - guard_periods)
            high = min(len(finite), int(found[0]) + guard_periods + 1)
            keep.iloc[low:high] = False
        finite = finite[keep]
    if len(finite) < 20:
        return None

    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None
    lower = q1 - OUTLIER_FENCE * iqr
    upper = q3 + OUTLIER_FENCE * iqr
    flagged = finite[(finite < lower) | (finite > upper)]
    if flagged.empty:
        return None

    rate = float(len(flagged) / len(finite))
    # Past this rate the series does not have outliers; it has a spread that
    # changes over time, and one fence built from the whole history sits in the
    # wrong place for most of it. Listing hundreds of "anomalies" would be an
    # artifact of the wrong tool, so the rate is reported and the list is not.
    # The variance-shift evidence is where that story actually belongs.
    suppressed = rate > OUTLIER_RATE_CEILING

    def beyond(item: float) -> float:
        return (item - upper) / iqr if item > upper else (lower - item) / iqr

    ranked = flagged.map(beyond).sort_values(ascending=False).head(MAX_OUTLIERS)
    rows: list[dict[str, Any]] = []
    if not suppressed:
        for label, score in zip(ranked.index, ranked.to_numpy(), strict=True):
            when = to_timestamp(label)
            remainder_value = float(flagged.loc[label])
            recorded_value = observed.get(label)
            rows.append(
                {
                    "timestamp": when.isoformat(),
                    "observed": (
                        float(recorded_value)
                        if recorded_value is not None and pd.notna(recorded_value)
                        else None
                    ),
                    "remainder": remainder_value,
                    "iqr_beyond_fence": float(score),
                    "direction": "high" if remainder_value > upper else "low",
                }
            )
    return Evidence.create(
        kind="time_series_outliers",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "outliers": rows,
            "outlier_count": int(len(flagged)),
            "outlier_rate": rate,
            "suppressed_as_changing_spread": suppressed,
            "fence_multiplier": OUTLIER_FENCE,
            "lower_fence": lower,
            "upper_fence": upper,
            "scored_period_count": int(len(finite)),
        },
        method="tukey_fence_remainder_outlier_scan_v2",
        description=f"Temporal outliers in {table}.{value}.",
        confidence=0.76,
        assumptions=(
            "Scored on the seasonally adjusted remainder, so an ordinary busy "
            "weekend is not flagged for being busy.",
            "Interpolated periods and the neighbourhood of a detected change "
            "point are excluded, because both produce artificial spikes.",
            "A temporal outlier is a candidate for review. A promotion and a "
            "data-entry error look identical here and need opposite treatment.",
        ),
    )


def intermittency_evidence(
    series: pd.Series,
    *,
    table: str,
    value: str,
) -> Evidence | None:
    """Zero runs, burstiness, and the demand pattern the series actually has.

    Only meaningful when the series contains zeros. Continuous demand gets no
    evidence here rather than a classification that would describe nothing.
    """
    finite = series.dropna()
    if len(finite) < 20:
        return None
    values = finite.to_numpy(dtype="float64")
    zero_mask = values == 0
    zero_rate = float(zero_mask.mean())
    if zero_rate < INTERMITTENT_MIN_ZERO_RATE:
        return None

    non_zero = values[~zero_mask]
    positions = np.flatnonzero(~zero_mask)
    intervals = np.diff(positions) if positions.size > 1 else np.array([])
    adi = float(intervals.mean()) if intervals.size else float(len(values))
    mean_demand = float(non_zero.mean()) if non_zero.size else 0.0
    cv_squared = (
        float((non_zero.std(ddof=1) / mean_demand) ** 2)
        if non_zero.size > 1 and mean_demand
        else 0.0
    )

    if adi < ADI_CUT and cv_squared < CV2_CUT:
        pattern = "smooth"
    elif adi >= ADI_CUT and cv_squared < CV2_CUT:
        pattern = "intermittent"
    elif adi < ADI_CUT:
        pattern = "erratic"
    else:
        pattern = "lumpy"

    # Longest run of consecutive zeros.
    longest = 0
    current = 0
    for flag in zero_mask:
        current = current + 1 if flag else 0
        longest = max(longest, current)

    return Evidence.create(
        kind="time_series_intermittency",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "zero_count": int(zero_mask.sum()),
            "zero_rate": zero_rate,
            "longest_zero_run": longest,
            "average_demand_interval": adi,
            "cv_squared_of_non_zero": cv_squared,
            "pattern": pattern,
            "non_zero_mean": mean_demand,
            "observation_count": int(len(values)),
        },
        method="syntetos_boylan_demand_classification_v1",
        description=f"Intermittent demand profile for {table}.{value}.",
        confidence=0.8,
        assumptions=(
            "The pattern uses the Syntetos-Boylan cut-offs (ADI 1.32, squared "
            "coefficient of variation 0.49) on non-zero demand.",
            "Squared-error forecasting on an intermittent series optimizes "
            "toward its average, which is a quantity the series never takes.",
        ),
    )
