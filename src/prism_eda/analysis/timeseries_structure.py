"""What shape does the series have?

Trend, seasonality, memory, and stationarity — the four properties that decide
which forecasting approaches are even applicable.

The load-bearing decision in this module is that **none of these are defects**.
Almost every real series with a trend is non-stationary; that is what a trend
is. Reporting "your series is non-stationary" as a problem to fix would be the
time-series equivalent of filing a skewed target as a data-quality issue, and
this library has already decided that is wrong. So everything here is an
observation about the data with a modelling consequence attached, and only
genuine failures — too little history to see a cycle you intend to forecast —
become issues.

The second decision concerns the stationarity tests. ADF and KPSS have opposite
null hypotheses, so they can and do disagree, and that disagreement is
informative rather than embarrassing: it distinguishes a trend-stationary series
from a difference-stationary one, and it distinguishes both from "this series is
too short to tell". Prism reports the four-way outcome rather than picking the
test that gives a cleaner answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prism_eda.analysis._timeseries import (
    MIN_SEASONAL_CYCLES,
    Frequency,
    as_float,
    series_points,
)
from prism_eda.evidence.models import Evidence, EvidenceScope

#: Lags reported from the autocorrelation function.
MAX_ACF_LAGS = 40

#: Seasonal-strength bands. Below the first, a seasonal component explains so
#: little of the remainder's variance that naming it would overstate it.
SEASONAL_WEAK = 0.3
SEASONAL_STRONG = 0.6
TREND_WEAK = 0.3

#: Conventional significance level for the stationarity pair. It drives the
#: classification below, never a finding on its own.
ALPHA = 0.05

#: Candidate seasonal periods ranked from the autocorrelation function.
MAX_SEASONAL_CANDIDATES = 3


def decomposition_evidence(
    series: pd.Series,
    frequency: Frequency,
    *,
    table: str,
    value: str,
    interpolated: bool,
) -> Evidence | None:
    """STL trend, season, and remainder, with the strength of each.

    Strength is the standard Wang–Smith–Hyndman formulation: how much of the
    variance the component removes from what is left after the others. It is
    bounded in [0, 1] and comparable across series, unlike the raw amplitude of
    a seasonal swing, which is only comparable to itself.
    """
    period = frequency.seasonal_period
    if period is None or len(series) < period * MIN_SEASONAL_CYCLES:
        return None
    try:
        from statsmodels.tsa.seasonal import STL

        # A robust fit stops a promotion spike or an outage from being absorbed
        # into the trend, which is exactly what would hide it from the change
        # point and outlier checks downstream.
        result = STL(series, period=period, robust=True).fit()
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate
        return None

    trend = pd.Series(result.trend, index=series.index)
    seasonal = pd.Series(result.seasonal, index=series.index)
    remainder = pd.Series(result.resid, index=series.index)

    def strength(component: pd.Series) -> float:
        combined = float(np.var(component + remainder, ddof=1))
        residual = float(np.var(remainder, ddof=1))
        if combined <= 0:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - residual / combined)))

    seasonal_strength = strength(seasonal)
    trend_strength = strength(trend)
    peak = (
        seasonal.groupby(pd.DatetimeIndex(seasonal.index).dayofweek).mean()
        if period == 7
        else None
    )
    profile: list[dict[str, Any]] = (
        [
            {"position": int(position), "effect": float(effect)}
            for position, effect in zip(
                peak.index.to_numpy(), peak.to_numpy(), strict=True
            )
        ]
        if peak is not None
        else []
    )

    return Evidence.create(
        kind="time_series_decomposition",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "period": period,
            "seasonal_label": frequency.seasonal_label,
            "seasonal_strength": seasonal_strength,
            "trend_strength": trend_strength,
            "has_seasonality": seasonal_strength >= SEASONAL_WEAK,
            "has_trend": trend_strength >= TREND_WEAK,
            "trend_direction": (
                "rising"
                if float(trend.iloc[-1]) > float(trend.iloc[0])
                else "falling"
                if float(trend.iloc[-1]) < float(trend.iloc[0])
                else "flat"
            ),
            "trend_change": float(trend.iloc[-1] - trend.iloc[0]),
            "seasonal_peak_to_trough": float(seasonal.max() - seasonal.min()),
            "remainder_std": float(remainder.std(ddof=1)),
            "observed_points": series_points(series),
            "trend_points": series_points(trend),
            "seasonal_profile": profile,
            "computed_on_interpolated_series": interpolated,
        },
        method="stl_decomposition_v1",
        description=f"Trend and seasonal decomposition of {table}.{value}.",
        confidence=0.82 if not interpolated else 0.74,
        assumptions=(
            "STL is fitted robustly, so isolated spikes land in the remainder "
            "rather than bending the trend.",
            "Component strength is the share of variance the component removes, "
            "bounded in [0, 1]; it is not the size of the seasonal swing.",
            "Trend and seasonality are properties of the series, not defects.",
        ),
    )


def autocorrelation_evidence(
    series: pd.Series,
    frequency: Frequency,
    *,
    table: str,
    value: str,
) -> Evidence | None:
    """How far back the series remembers, and at which lags."""
    usable = int(min(MAX_ACF_LAGS, max(1, len(series) // 3)))
    if usable < 2:
        return None
    try:
        from statsmodels.tsa.stattools import acf, pacf

        acf_values = acf(series, nlags=usable, fft=True)
        pacf_values = pacf(series, nlags=min(usable, len(series) // 2 - 1))
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate
        return None

    # The standard large-sample band. Anything inside it is indistinguishable
    # from no correlation at that lag.
    band = 1.96 / np.sqrt(len(series))
    lags: list[dict[str, Any]] = [
        {
            "lag": index,
            "acf": float(acf_values[index]),
            "pacf": float(pacf_values[index]) if index < len(pacf_values) else None,
            "significant": bool(abs(acf_values[index]) > band),
        }
        for index in range(1, len(acf_values))
    ]
    significant = [item for item in lags if item["significant"]]

    # Seasonal candidates are lags whose autocorrelation is a local peak: a
    # weekly cycle in daily data shows as a spike at 7, 14, 21 rather than a
    # smooth decay.
    candidates: list[dict[str, Any]] = []
    for item in lags[1:-1]:
        index = int(item["lag"])
        previous, following = acf_values[index - 1], acf_values[index + 1]
        if (
            acf_values[index] > previous
            and acf_values[index] > following
            and acf_values[index] > band
        ):
            candidates.append({"period": index, "acf": float(acf_values[index])})
    candidates.sort(key=lambda item: item["acf"], reverse=True)

    return Evidence.create(
        kind="time_series_autocorrelation",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "lags": lags,
            "confidence_band": float(band),
            "significant_lag_count": len(significant),
            "first_insignificant_lag": next(
                (item["lag"] for item in lags if not item["significant"]), None
            ),
            "lag_one": float(acf_values[1]) if len(acf_values) > 1 else None,
            "seasonal_candidates": candidates[:MAX_SEASONAL_CANDIDATES],
            "expected_seasonal_period": frequency.seasonal_period,
        },
        method="acf_pacf_v1",
        description=f"Autocorrelation structure of {table}.{value}.",
        confidence=0.8,
        assumptions=(
            "The confidence band is the large-sample 1.96/sqrt(n) approximation "
            "and assumes no strong trend; a trending series shows slowly "
            "decaying autocorrelation at every lag for that reason alone.",
        ),
    )


def stationarity_evidence(
    series: pd.Series,
    *,
    table: str,
    value: str,
) -> Evidence | None:
    """ADF and KPSS together, with their disagreement reported rather than hidden.

    The two tests have opposite nulls — ADF's is "there is a unit root", KPSS's
    is "the series is stationary" — so running only one answers half the
    question. Running both gives a four-way outcome that is far more useful than
    either verdict alone, including the honest "these two disagree, and here is
    what that usually means" case.
    """
    if len(series) < 20 or float(series.std(ddof=1) or 0.0) <= 0:
        return None
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:  # pragma: no cover - statsmodels is a core dependency
        return None

    try:
        adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
        return None
    try:
        import warnings as _warnings

        with _warnings.catch_warnings():
            # KPSS clamps its p-value at the edge of its lookup table and warns.
            # That is expected, and the clamped value is reported as-is.
            _warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
        return None

    adf_says_stationary = bool(adf_p < ALPHA)
    kpss_says_stationary = bool(kpss_p >= ALPHA)

    if adf_says_stationary and kpss_says_stationary:
        verdict, explanation = (
            "stationary",
            "Both tests agree the level is stable. Differencing is not needed.",
        )
    elif not adf_says_stationary and not kpss_says_stationary:
        verdict, explanation = (
            "non_stationary",
            "Both tests agree the level moves. Difference the series, or use a "
            "model that handles a trend directly.",
        )
    elif not adf_says_stationary and kpss_says_stationary:
        verdict, explanation = (
            "trend_stationary",
            "KPSS sees a stable level while ADF cannot rule out a unit root. "
            "This usually means a deterministic trend: de-trend rather than "
            "difference.",
        )
    else:
        verdict, explanation = (
            "difference_stationary",
            "ADF rejects a unit root while KPSS rejects stationarity. This "
            "usually means the series is stationary around a shifting level; "
            "differencing is the safer choice.",
        )

    return Evidence.create(
        kind="time_series_stationarity",
        scope=EvidenceScope(table=table, columns=(value,)),
        value={
            "value": value,
            "verdict": verdict,
            "explanation": explanation,
            "tests_agree": adf_says_stationary == kpss_says_stationary,
            "adf": {
                "statistic": as_float(adf_stat),
                "p_value": as_float(adf_p),
                "null_hypothesis": "a unit root is present (non-stationary)",
                "rejects_null": adf_says_stationary,
            },
            "kpss": {
                "statistic": as_float(kpss_stat),
                "p_value": as_float(kpss_p),
                "null_hypothesis": "the series is stationary",
                "rejects_null": not kpss_says_stationary,
            },
            "observation_count": int(len(series)),
        },
        method="adf_kpss_agreement_v1",
        description=f"Stationarity of {table}.{value}.",
        confidence=0.78,
        assumptions=(
            "Non-stationarity is a property of the series, not a defect. Most "
            "series with a trend are non-stationary by construction.",
            "Both tests lose power on short series, so a 'stationary' verdict on "
            "a few dozen observations is weak evidence rather than reassurance.",
            "KPSS clamps its p-value at the limits of its lookup table, so an "
            "extreme value is reported at the boundary rather than beyond it.",
        ),
    )
