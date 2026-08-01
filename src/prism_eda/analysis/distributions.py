"""Per-column distribution evidence: shape, fitted family, and chart data.

Two questions an analyst asks of a numeric column, in order:

1. *What does it look like?* Answered descriptively — bell-shaped, right-skewed
   with a long tail, bimodal, zero-inflated — from moments and a robust gap test.
   This is always available and never wrong, only coarse.
2. *Is it a known distribution?* Answered by fitting candidate families and
   ranking them by how far the fitted CDF sits from the empirical one. This one
   **abstains**: if nothing fits well, the report says so rather than naming the
   least-bad option.

On the fit statistic: it is a Kolmogorov-Smirnov *distance* computed with
parameters estimated from the same data. That makes it a descriptive measure of
fit quality, not a hypothesis test — the corresponding p-value would be
optimistic and is deliberately neither computed nor shown. Reporting "p = 0.31,
so the data is normal" is the single most common way fitted-distribution output
misleads people.

The chart data banked here is the exact shape ``reporting.charts.histogram_svg``
already consumes, so the renderer needs no live DataFrame.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from prism_eda.analysis._numeric import HIST_BINS, detect_modality
from prism_eda.catalog.models import TableCatalog
from prism_eda.evidence.models import Evidence, EvidenceScope

# Below this many values a histogram describes the sample, not the distribution.
MIN_HISTOGRAM_ROWS = 8
# Family fitting on a handful of points reliably "discovers" a family that is not
# there, so it does not run at all below this.
MIN_FIT_ROWS = 30
# Maximum KS distance still called a fit. Beyond the second value the report
# abstains rather than naming a family.
FIT_CLOSE_DISTANCE = 0.05
FIT_APPROXIMATE_DISTANCE = 0.10
# How many ranked candidates are kept for the report's detail view.
MAX_RANKED_FAMILIES = 3
# How close a Poisson has to come to the best continuous fit before integer data
# is described as counts. Continuous families compete on AIC (see
# _select_family); this tolerance only arbitrates the discrete/continuous choice,
# where AIC is not comparable.
FIT_PARSIMONY_TOLERANCE = 0.01
# Free parameters actually estimated per family, given that strictly positive
# families are fitted with the location pinned at zero.
_FREE_PARAMETERS = {
    "expon": 1,
    "poisson": 1,
    "norm": 2,
    "uniform": 2,
    "lognorm": 2,
    "gamma": 2,
    "weibull_min": 2,
    "beta": 4,
}

# Shape thresholds. Skewness and excess kurtosis both key off the normal
# distribution's values (0 and 0); a uniform sits at about -1.2 excess kurtosis,
# which is what separates "flat" from "bell".
SKEW_SYMMETRIC = 0.5
SKEW_STRONG = 1.0
KURTOSIS_FLAT = -0.9
KURTOSIS_HEAVY = 3.0
ZERO_INFLATED_FRACTION = 0.30
NEAR_CONSTANT_SHARE = 0.95

# Categorical frequency evidence keeps this many labels before the rest collapse
# into a single "other" bucket, so a 500-category column still renders.
MAX_CATEGORY_BARS = 15

_FAMILY_LABELS = {
    "norm": "Normal (Gaussian)",
    "lognorm": "Log-normal",
    "expon": "Exponential",
    "gamma": "Gamma",
    "weibull_min": "Weibull",
    "uniform": "Uniform",
    "beta": "Beta",
    "poisson": "Poisson",
}


def _finite(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return numeric
    return numeric[np.isfinite(numeric.to_numpy(dtype="float64"))]


def _shape(values: np.ndarray, series: pd.Series) -> dict[str, Any]:
    """Describe the column's shape in words an analyst can act on."""
    n = len(values)
    zero_fraction = float(np.mean(values == 0.0))
    unique, counts = np.unique(values, return_counts=True)
    distinct = int(len(unique))
    is_integer = bool(np.all(np.equal(np.mod(values, 1), 0)))
    top_share = float(counts.max() / n) if n else 0.0

    # Moments of a constant column are undefined, and asking for them anyway
    # emits a precision-loss warning for a question with an obvious answer.
    if distinct == 1:
        return {
            "label": "constant",
            "descriptors": ["every value is identical"],
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "zero_fraction": zero_fraction,
            "distinct_count": distinct,
            "is_integer": is_integer,
            "modality": {"is_multimodal": False, "clusters": []},
            "top_value_share": top_share,
        }

    skewness = float(stats.skew(values)) if n > 2 else 0.0
    excess_kurtosis = float(stats.kurtosis(values)) if n > 3 else 0.0
    modality = detect_modality(series)
    descriptors: list[str] = []

    if modality.get("is_multimodal"):
        label = "bimodal"
        descriptors.append("two separated groups, not one population")
    elif abs(skewness) < SKEW_SYMMETRIC and excess_kurtosis <= KURTOSIS_FLAT:
        label = "uniform"
        descriptors.append("values spread evenly across the range")
    elif abs(skewness) < SKEW_SYMMETRIC and excess_kurtosis < KURTOSIS_HEAVY:
        label = "bell-shaped"
        descriptors.append("symmetric, concentrated around the centre")
    elif skewness >= SKEW_STRONG:
        label = "right-skewed"
        descriptors.append("long upper tail; the mean sits above the median")
    elif skewness <= -SKEW_STRONG:
        label = "left-skewed"
        descriptors.append("long lower tail; the mean sits below the median")
    elif skewness > 0:
        label = "mildly right-skewed"
    else:
        label = "mildly left-skewed"

    if excess_kurtosis >= KURTOSIS_HEAVY:
        descriptors.append("heavy-tailed: extreme values are far more common here")
    if zero_fraction >= ZERO_INFLATED_FRACTION:
        descriptors.append(f"zero-inflated: {zero_fraction:.0%} of values are zero")
    if top_share >= NEAR_CONSTANT_SHARE:
        descriptors.append(f"near-constant: one value covers {top_share:.0%} of rows")
    if is_integer and float(values.min()) >= 0:
        descriptors.append("count-like: non-negative whole numbers")

    return {
        "label": label,
        "descriptors": descriptors,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "zero_fraction": zero_fraction,
        "distinct_count": distinct,
        "is_integer": is_integer,
        "modality": modality,
        "top_value_share": top_share,
    }


def _candidate_families(values: np.ndarray, is_integer: bool) -> list[str]:
    """Only offer families whose support can actually contain this data."""
    minimum = float(values.min())
    maximum = float(values.max())
    families = ["norm", "uniform"]
    if minimum > 0:
        families += ["lognorm", "expon", "gamma", "weibull_min"]
    if 0.0 <= minimum and maximum <= 1.0:
        families.append("beta")
    if is_integer and minimum >= 0:
        families.append("poisson")
    return families


def _fit_continuous(values: np.ndarray, family: str) -> dict[str, Any] | None:
    """Fit one continuous family, returning its KS distance and AIC."""
    distribution = getattr(stats, family)
    # Anchoring the location at zero is the standard two-parameter fit for a
    # strictly positive family and keeps the optimiser out of trouble.
    kwargs = (
        {"floc": 0} if family in {"lognorm", "expon", "gamma", "weibull_min"} else {}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parameters = distribution.fit(values, **kwargs)
            result = stats.kstest(values, distribution.cdf, args=parameters)
            log_likelihood = float(np.sum(distribution.logpdf(values, *parameters)))
        except (ValueError, RuntimeError, FloatingPointError, TypeError):
            return None
    distance = float(result.statistic)
    if not np.isfinite(distance) or not np.isfinite(log_likelihood):
        return None
    names = list(distribution.shapes.split(", ")) if distribution.shapes else []
    names += ["loc", "scale"]
    free = _FREE_PARAMETERS.get(family, 2)
    return {
        "family": family,
        "label": _FAMILY_LABELS.get(family, family),
        "ks_distance": distance,
        "aic": 2 * free - 2 * log_likelihood,
        "free_parameters": free,
        "discrete": False,
        "parameters": {
            name: float(value) for name, value in zip(names, parameters, strict=False)
        },
    }


def _fit_poisson(values: np.ndarray) -> dict[str, Any] | None:
    """Fit a Poisson, scoring it on the same KS-distance scale as the rest."""
    lam = float(values.mean())
    if lam <= 0:
        return None
    support = np.unique(values)
    empirical = np.searchsorted(np.sort(values), support, side="right") / len(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theoretical = stats.poisson.cdf(support, lam)
        log_likelihood = float(np.sum(stats.poisson.logpmf(values, lam)))
    distance = float(np.max(np.abs(empirical - theoretical)))
    if not np.isfinite(distance) or not np.isfinite(log_likelihood):
        return None
    return {
        "family": "poisson",
        "label": _FAMILY_LABELS["poisson"],
        "ks_distance": distance,
        # Reported for completeness, but never compared against a continuous
        # AIC: a probability mass and a probability density are not measured on
        # the same scale, and the comparison would be meaningless.
        "aic": 2 * 1 - 2 * log_likelihood,
        "free_parameters": 1,
        "discrete": True,
        "parameters": {"mu": lam},
    }


def _select_family(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the family to report, penalising unearned flexibility.

    Selecting on KS distance alone always names the most flexible candidate: a
    gamma contains the exponential, so on exponential data it fits at least as
    closely, and the report would say "Gamma" for textbook exponential data.
    AIC is the standard correction — it charges two units per free parameter, so
    a second parameter has to buy at least that much log-likelihood to be worth
    naming.

    Poisson is chosen separately rather than by AIC, because a probability mass
    and a probability density are not on a comparable scale. Integer data that a
    Poisson describes about as closely as the best continuous family gets the
    count model, which is the honest description of counts.
    """
    continuous = [item for item in candidates if not item["discrete"]]
    discrete = [item for item in candidates if item["discrete"]]
    best_continuous = (
        min(continuous, key=lambda item: item["aic"]) if continuous else None
    )
    best_discrete = (
        min(discrete, key=lambda item: item["ks_distance"]) if discrete else None
    )

    if best_continuous is None:
        assert best_discrete is not None
        return best_discrete
    if best_discrete is None:
        return best_continuous
    if (
        best_discrete["ks_distance"]
        <= best_continuous["ks_distance"] + FIT_PARSIMONY_TOLERANCE
    ):
        return best_discrete
    return best_continuous


def _fit(values: np.ndarray, is_integer: bool, limit: int) -> dict[str, Any]:
    """Fit candidate families and name one, abstaining when none fits well."""
    if len(values) < MIN_FIT_ROWS or len(np.unique(values)) < 3:
        return {
            "family": None,
            "reason": "too_few_values",
            "ranked": [],
            "evaluated_row_count": int(len(values)),
        }
    sample = values
    if len(sample) > limit:
        generator = np.random.default_rng(42)
        sample = np.sort(generator.choice(sample, size=limit, replace=False))

    candidates: list[dict[str, Any]] = []
    for family in _candidate_families(sample, is_integer):
        fitted = (
            _fit_poisson(sample)
            if family == "poisson"
            else _fit_continuous(sample, family)
        )
        if fitted is not None:
            candidates.append(fitted)
    if not candidates:
        return {
            "family": None,
            "reason": "no_candidate_converged",
            "ranked": [],
            "evaluated_row_count": int(len(sample)),
        }

    best = _select_family(candidates)
    ranked = [
        best,
        *sorted(
            (item for item in candidates if item is not best),
            key=lambda item: item["ks_distance"],
        ),
    ]
    distance = float(best["ks_distance"])
    if distance <= FIT_CLOSE_DISTANCE:
        quality = "close"
    elif distance <= FIT_APPROXIMATE_DISTANCE:
        quality = "approximate"
    else:
        return {
            "family": None,
            "reason": "no_family_fits_well",
            "best_distance": distance,
            "ranked": ranked[:MAX_RANKED_FAMILIES],
            "evaluated_row_count": int(len(sample)),
        }
    return {
        "family": best["family"],
        "label": best["label"],
        "parameters": best["parameters"],
        "ks_distance": distance,
        "quality": quality,
        "ranked": ranked[:MAX_RANKED_FAMILIES],
        "evaluated_row_count": int(len(sample)),
    }


def numeric_distribution_evidence(
    series: pd.Series,
    *,
    table: str,
    column: str,
    fit_rows: int,
) -> Evidence | None:
    """Histogram, box summary, shape label, and fitted family for one column."""
    finite = _finite(series)
    if len(finite) < MIN_HISTOGRAM_ROWS:
        return None
    values = finite.to_numpy(dtype="float64")
    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    box = {
        "min": float(finite.min()),
        "q1": q1,
        "median": float(finite.median()),
        "q3": q3,
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "lower_fence": q1 - 1.5 * iqr,
        "upper_fence": q3 + 1.5 * iqr,
    }
    bin_count = min(HIST_BINS, max(6, len(finite) // 2))
    counts, edges = np.histogram(values, bins=bin_count)
    shape = _shape(values, finite)
    fit = _fit(values, bool(shape["is_integer"]), fit_rows)

    return Evidence.create(
        kind="profile_distribution",
        scope=EvidenceScope(table=table, columns=(column,)),
        value={
            "column": column,
            "evaluated_row_count": int(len(finite)),
            "box": box,
            "histogram": {
                "counts": [int(count) for count in counts],
                "edges": [float(edge) for edge in edges],
            },
            # The renderer's histogram highlights flagged values; the baseline
            # profile flags none, so this is deliberately empty.
            "flagged_values": [],
            "shape": shape,
            "fit": fit,
        },
        method="histogram_moments_ks_family_fit_v1",
        description=f"Distribution of {table}.{column}.",
        confidence=0.9,
        assumptions=(
            "The fit statistic is a Kolmogorov-Smirnov distance computed with "
            "parameters estimated from the same data, so it describes fit "
            "quality and is not a hypothesis test.",
            "Shape labels are descriptive summaries of moments and gaps, not a "
            "fitted mixture model.",
        ),
    )


def category_frequency_evidence(
    series: pd.Series,
    *,
    table: str,
    column: str,
) -> Evidence | None:
    """Label frequencies for a categorical or boolean column."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    try:
        counts = non_null.value_counts()
    except (TypeError, ValueError):
        counts = non_null.astype(str).value_counts()
    total = int(len(series))
    shown = counts.head(MAX_CATEGORY_BARS)
    other_count = int(counts.iloc[MAX_CATEGORY_BARS:].sum())
    return Evidence.create(
        kind="profile_category_frequency",
        scope=EvidenceScope(table=table, columns=(column,)),
        value={
            "column": column,
            "distinct_count": int(len(counts)),
            "counts": [
                {
                    "value": "(blank)" if str(value).strip() == "" else str(value),
                    "count": int(count),
                    "rate": float(count / total) if total else 0.0,
                }
                for value, count in shown.items()
            ],
            "other_count": other_count,
            "other_category_count": max(0, int(len(counts)) - len(shown)),
            "missing_count": int(total - len(non_null)),
        },
        method="exact_value_counts_v1",
        description=f"Label frequencies for {table}.{column}.",
    )


def datetime_timeline_evidence(
    series: pd.Series,
    *,
    table: str,
    column: str,
) -> Evidence | None:
    """Row counts over time, at a granularity that suits the covered range."""
    values = series.dropna()
    if len(values) < MIN_HISTOGRAM_ROWS:
        return None
    start = values.min()
    end = values.max()
    span_days = float((end - start).total_seconds()) / 86_400 if end > start else 0.0
    if span_days <= 2:
        rule, granularity = "h", "hour"
    elif span_days <= 90:
        rule, granularity = "D", "day"
    elif span_days <= 900:
        rule, granularity = "W", "week"
    elif span_days <= 3_650:
        rule, granularity = "MS", "month"
    else:
        rule, granularity = "YS", "year"
    buckets = values.groupby(values.dt.to_period(rule[0]).dt.start_time).size()
    return Evidence.create(
        kind="profile_timeline",
        scope=EvidenceScope(table=table, columns=(column,)),
        value={
            "column": column,
            "granularity": granularity,
            "min": start.isoformat(),
            "max": end.isoformat(),
            "buckets": [
                {
                    "start": str(pd.Timestamp(str(stamp)).isoformat()),
                    "count": int(count),
                }
                for stamp, count in buckets.items()
            ],
        },
        method="period_resample_counts_v1",
        description=f"Row counts over time for {table}.{column}.",
    )


def build_distribution_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    chart_columns: int,
    fit_rows: int,
) -> tuple[list[Evidence], list[str]]:
    """Chart evidence for every column that fits under the chart cap.

    Returns the evidence and the names of the columns that were skipped, so the
    caller can say so instead of leaving a silent gap in the report.
    """
    evidence: list[Evidence] = []
    skipped: list[str] = []
    charted = 0
    for column in table.columns:
        if column.name not in frame.columns:
            continue
        if charted >= chart_columns:
            skipped.append(column.name)
            continue
        series = frame[column.name]
        item: Evidence | None = None
        if column.semantic_type == "numeric":
            item = numeric_distribution_evidence(
                series, table=table.name, column=column.name, fit_rows=fit_rows
            )
        elif column.semantic_type == "datetime":
            item = datetime_timeline_evidence(
                series, table=table.name, column=column.name
            )
        elif column.semantic_type in {"categorical", "boolean"}:
            item = category_frequency_evidence(
                series, table=table.name, column=column.name
            )
        if item is not None:
            evidence.append(item)
            charted += 1
    return evidence, skipped
