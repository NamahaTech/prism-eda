"""Numeric helpers shared by the analysis recipes.

These live outside any one recipe because the profile and the anomaly review ask
the same shape questions of a numeric column — how it is binned, whether it is
really two populations — and the answers must agree between the two reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Distribution-shape diagnostics. A column with two separated clusters is not a
# clean distribution with a few tail outliers — it is two populations, and that
# reframing is usually the single most useful thing to tell an analyst. We detect
# it with a robust largest-gap split (on a log axis for heavily skewed positive
# columns) and only call it when both sides hold a real share of the rows.
HIST_BINS = 24
BIMODAL_MIN_FRACTION = 0.15
BIMODAL_MIN_GROUP = 4
BIMODAL_GAP_RATIO = 2.5
# The separating gap must also span a real share of the column's spread, so that
# the small ±1 gaps in ordinary discrete/uniform data (e.g. integer tenures) are
# not mistaken for a true two-population split.
BIMODAL_MIN_RANGE_FRACTION = 0.10
# Below this many values the largest-gap test is reading noise, not structure.
MODALITY_MIN_ROWS = 20


def as_float(value: Any) -> float:
    """Coerce a pandas/numpy scalar to ``float`` (keeps the type-checker happy)."""
    return float(value)


def detect_modality(series: pd.Series) -> dict[str, Any]:
    """Robust largest-gap test for a two-population split."""
    values = np.sort(series.to_numpy(dtype="float64"))
    n = len(values)
    result: dict[str, Any] = {"is_multimodal": False, "clusters": []}
    if n < MODALITY_MIN_ROWS:
        return result
    work = values
    log_space = False
    if float(values.min()) > 0:
        skew = as_float(series.skew()) if n > 2 else 0.0
        spread = values.max() / max(values.min(), 1e-9)
        if abs(skew) >= 1.0 or spread >= 50:
            work = np.log10(values)
            log_space = True
    gaps = np.diff(work)
    positive = gaps[gaps > 0]
    if gaps.size == 0 or positive.size == 0:
        return result
    median_gap = float(np.median(positive))
    work_range = float(work[-1] - work[0])
    max_idx = int(np.argmax(gaps))
    max_gap = float(gaps[max_idx])
    lower_n = max_idx + 1
    upper_n = n - lower_n
    if (
        median_gap > 0
        and max_gap >= BIMODAL_GAP_RATIO * median_gap
        and work_range > 0
        and max_gap >= BIMODAL_MIN_RANGE_FRACTION * work_range
        and lower_n >= BIMODAL_MIN_GROUP
        and upper_n >= BIMODAL_MIN_GROUP
        and lower_n / n >= BIMODAL_MIN_FRACTION
        and upper_n / n >= BIMODAL_MIN_FRACTION
    ):
        boundary_low = float(values[max_idx])
        boundary_high = float(values[max_idx + 1])
        result.update(
            {
                "is_multimodal": True,
                "log_space": log_space,
                "gap": {"low": boundary_low, "high": boundary_high},
                "clusters": [
                    {
                        "min": float(values[0]),
                        "max": boundary_low,
                        "count": lower_n,
                        "fraction": lower_n / n,
                    },
                    {
                        "min": boundary_high,
                        "max": float(values[-1]),
                        "count": upper_n,
                        "fraction": upper_n / n,
                    },
                ],
            }
        )
    return result
