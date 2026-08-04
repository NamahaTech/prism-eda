"""What shape is the thing being predicted?

Regression readiness starts with the target, because several of the ways a
regression goes wrong are visible before a single feature is considered: a
target that is capped, one that piles up at zero, one whose tail is so long that
a squared-error fit will chase it, or one that turns out to be a class label
someone stored as an integer.

The rule this module holds to is that shape is a *property*, not a defect. A
right-skewed target is not broken, and saying so in a findings list would be
noise. Skew earns a finding only when a transformation measurably fixes it, and
that improvement is computed here rather than asserted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from prism_eda.analysis._regression import (
    as_float,
    distribution_payload,
    safe_ratio,
)
from prism_eda.evidence.models import Evidence, EvidenceScope

# Conventional skew bands. Below 0.5 a distribution reads as symmetric; past 1.0
# the mean has stopped describing the centre.
_SKEW_MODERATE = 0.5
_SKEW_STRONG = 1.0

# A transformation has to earn its recommendation. Cutting |skew| by less than
# this is not worth the interpretability a transformed target costs.
_MIN_SKEW_REDUCTION = 0.3

# A repeated value in a continuous target is a spike. Both bars must clear:
# a rate, so tiny tables do not fire, and a count, so huge ones do not fire on
# noise.
_SPIKE_MIN_RATE = 0.01
_SPIKE_MIN_COUNT = 5
_MAX_SPIKES = 5

# Round-number preference. Heaping is only interesting when the multiples show
# up far more often than an unrounded quantity would produce.
_HEAPING_BASES = (100.0, 10.0, 5.0)
_HEAPING_MIN_RATE = 0.25


def _looks_discrete(values: np.ndarray) -> bool:
    """True when repeated values are expected rather than suspicious."""
    distinct = np.unique(values).size
    if distinct <= 1:
        return True
    return distinct <= 20 or (distinct / values.size) < 0.05


def target_summary_evidence(
    frame: pd.DataFrame, table: str, target: str, values: np.ndarray
) -> Evidence:
    """Range, centre, spread, tails, and the histogram the report draws."""
    total = len(frame)
    finite = values[np.isfinite(values)]
    series = pd.Series(finite, dtype="float64")
    distinct = int(np.unique(finite).size) if finite.size else 0
    q1 = float(np.quantile(finite, 0.25)) if finite.size else 0.0
    q3 = float(np.quantile(finite, 0.75)) if finite.size else 0.0
    median = float(np.median(finite)) if finite.size else 0.0
    mean = float(finite.mean()) if finite.size else 0.0
    std = float(series.std(ddof=1)) if finite.size > 1 else 0.0

    quantiles = (
        {
            f"q{int(level * 100):02d}": float(np.quantile(finite, level))
            for level in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        }
        if finite.size
        else {}
    )

    return Evidence.create(
        kind="regression_target_summary",
        scope=EvidenceScope(table=table, columns=(target,)),
        value={
            "target": target,
            "row_count": total,
            "non_missing_count": int(finite.size),
            "missing_count": total - int(finite.size),
            "missing_rate": (total - int(finite.size)) / total if total else 0.0,
            "distinct_count": distinct,
            "distinct_rate": distinct / finite.size if finite.size else 0.0,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "range": float(finite.max() - finite.min()) if finite.size else None,
            "mean": mean,
            "median": median,
            "std": std,
            "iqr": q3 - q1,
            "skewness": as_float(series.skew()) if finite.size > 2 else 0.0,
            "kurtosis": as_float(series.kurtosis()) if finite.size > 3 else 0.0,
            "zero_count": int((finite == 0).sum()),
            "zero_rate": float((finite == 0).mean()) if finite.size else 0.0,
            "negative_count": int((finite < 0).sum()),
            "negative_rate": float((finite < 0).mean()) if finite.size else 0.0,
            "coefficient_of_variation": safe_ratio(std, abs(mean)),
            "looks_discrete": _looks_discrete(finite) if finite.size else True,
            "quantiles": quantiles,
            "distribution": distribution_payload(finite, column=target),
        },
        method="regression_target_summary_v1",
        description=f"Regression target summary for {table}.{target}.",
        confidence=1.0,
    )


def _transform_candidates(finite: np.ndarray, skewness: float) -> list[dict[str, Any]]:
    """Measure what each transformation actually does to the skew."""
    candidates: list[dict[str, Any]] = []
    minimum = float(finite.min())

    def record(
        name: str,
        applicable: bool,
        reason: str,
        transformed: np.ndarray | None,
    ) -> None:
        after: float | None = None
        if applicable and transformed is not None and transformed.size > 2:
            values = transformed[np.isfinite(transformed)]
            if values.size > 2:
                after = as_float(pd.Series(values, dtype="float64").skew())
        candidates.append(
            {
                "transform": name,
                "applicable": applicable,
                "reason": reason,
                "skewness_after": after,
                "skew_reduction": (
                    abs(skewness) - abs(after) if after is not None else None
                ),
            }
        )

    non_negative = minimum >= 0
    record(
        "log1p",
        non_negative,
        "Requires non-negative values." if not non_negative else "",
        np.log1p(finite) if non_negative else None,
    )
    record(
        "sqrt",
        non_negative,
        "Requires non-negative values." if not non_negative else "",
        np.sqrt(finite) if non_negative else None,
    )
    try:
        yeo, _ = stats.yeojohnson(finite)
    except (ValueError, RuntimeWarning):  # pragma: no cover - degenerate input
        yeo = None
    record(
        "yeo_johnson",
        yeo is not None,
        "" if yeo is not None else "Could not be estimated on this target.",
        yeo,
    )
    return candidates


def target_shape_evidence(
    table: str, target: str, summary: Evidence, values: np.ndarray
) -> Evidence | None:
    """Name the shape, and only recommend a transform that measurably helps."""
    finite = values[np.isfinite(values)]
    if finite.size < 20:
        return None
    skewness = float(summary.value["skewness"])
    magnitude = abs(skewness)
    if magnitude < _SKEW_MODERATE:
        shape = "symmetric"
    elif magnitude < _SKEW_STRONG:
        shape = "moderately_right_skewed" if skewness > 0 else "moderately_left_skewed"
    else:
        shape = "strongly_right_skewed" if skewness > 0 else "strongly_left_skewed"

    candidates = _transform_candidates(finite, skewness)
    usable = [
        item
        for item in candidates
        if item["applicable"]
        and item["skew_reduction"] is not None
        and item["skew_reduction"] >= _MIN_SKEW_REDUCTION
        and abs(item["skewness_after"]) < _SKEW_MODERATE
    ]
    best = max(usable, key=lambda item: item["skew_reduction"]) if usable else None

    median = float(summary.value["median"])
    quantiles = summary.value["quantiles"]
    upper = quantiles.get("q99", median) - median
    lower = median - quantiles.get("q01", median)

    transformed_distribution: dict[str, Any] = {}
    if best is not None:
        if best["transform"] == "log1p":
            transformed = np.log1p(finite)
        elif best["transform"] == "sqrt":
            transformed = np.sqrt(finite)
        else:
            transformed, _ = stats.yeojohnson(finite)
        transformed_distribution = distribution_payload(
            transformed, column=f"{best['transform']}({target})"
        )

    return Evidence.create(
        kind="regression_target_shape",
        scope=EvidenceScope(table=table, columns=(target,)),
        value={
            "target": target,
            "shape": shape,
            "skewness": skewness,
            "kurtosis": float(summary.value["kurtosis"]),
            "tail_ratio": safe_ratio(upper, lower),
            "candidates": candidates,
            "best_candidate": best["transform"] if best else None,
            "best_skewness_after": best["skewness_after"] if best else None,
            "distribution_raw": summary.value["distribution"],
            "distribution_transformed": transformed_distribution,
        },
        method="target_shape_and_measured_transform_v1",
        description=f"Target shape and transformation candidates for {target}.",
        confidence=0.88,
        assumptions=(
            "Skew is a property of the target, not a defect. A transformation is "
            "only recommended when it measurably reduces skew on this data.",
            "A transformed target changes what the model's errors mean; back-"
            "transformed predictions are biased unless corrected.",
        ),
    )


def target_spike_evidence(
    table: str, target: str, summary: Evidence, values: np.ndarray
) -> Evidence | None:
    """Repeated exact values in a target that is otherwise continuous.

    A continuous quantity should almost never repeat. When it does — the same
    number in dozens of rows — something has clipped it: a contract cap, a
    default written on insert, an imputed placeholder, or a floor at zero. Every
    one of those makes the affected rows unusable as regression targets, and
    none of them are visible in a mean or a histogram bucket.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 50 or summary.value["looks_discrete"]:
        return None
    counts = pd.Series(finite, dtype="float64").value_counts()
    minimum = float(finite.min())
    maximum = float(finite.max())
    spikes: list[dict[str, Any]] = []
    for value, count in counts.items():
        rate = int(count) / finite.size
        if int(count) < _SPIKE_MIN_COUNT or rate < _SPIKE_MIN_RATE:
            continue
        number = float(value)  # type: ignore[arg-type]
        if number == maximum:
            position = "at_maximum"
        elif number == minimum:
            position = "at_zero" if number == 0.0 else "at_minimum"
        elif number == 0.0:
            position = "at_zero"
        else:
            position = "interior"
        spikes.append(
            {
                "value": number,
                "count": int(count),
                "rate": rate,
                "position": position,
            }
        )
        if len(spikes) >= _MAX_SPIKES:
            break
    if not spikes:
        return None
    return Evidence.create(
        kind="regression_target_spikes",
        scope=EvidenceScope(table=table, columns=(target,)),
        value={
            "target": target,
            "spikes": spikes,
            "spiked_row_count": sum(item["count"] for item in spikes),
            "spiked_row_rate": sum(item["rate"] for item in spikes),
            "distinct_count": summary.value["distinct_count"],
        },
        method="continuous_target_value_spike_scan_v1",
        description=f"Repeated exact target values in {table}.{target}.",
        confidence=0.86,
        assumptions=(
            "A repeated value in a continuous target usually means censoring, a "
            "default, or an imputed placeholder — but a genuinely popular price "
            "point looks the same. Confirm against how the column is recorded.",
        ),
    )


def target_heaping_evidence(
    table: str, target: str, summary: Evidence, values: np.ndarray
) -> Evidence | None:
    """Round-number preference: values reported rather than measured."""
    finite = values[np.isfinite(values)]
    if finite.size < 50 or summary.value["looks_discrete"]:
        return None
    for base in _HEAPING_BASES:
        rate = float(np.mean(np.isclose(np.remainder(finite, base), 0.0)))
        if rate < _HEAPING_MIN_RATE:
            continue
        return Evidence.create(
            kind="regression_target_heaping",
            scope=EvidenceScope(table=table, columns=(target,)),
            value={
                "target": target,
                "base": base,
                "multiple_rate": rate,
                "multiple_count": int(rate * finite.size),
            },
            method="round_number_heaping_scan_v1",
            description=f"Round-number heaping in {table}.{target}.",
            confidence=0.78,
            assumptions=(
                "Heaping suggests the target was reported or negotiated rather "
                "than measured, so its precision is coarser than its decimals "
                "imply.",
            ),
        )
    return None


def target_evidence(
    frame: pd.DataFrame, table: str, target: str, values: np.ndarray
) -> list[Evidence]:
    """Every target-side diagnostic, summary first so the rest can read it."""
    summary = target_summary_evidence(frame, table, target, values)
    evidence: list[Evidence] = [summary]
    for builder in (
        target_shape_evidence,
        target_spike_evidence,
        target_heaping_evidence,
    ):
        item = builder(table, target, summary, values)
        if item is not None:
            evidence.append(item)
    return evidence
