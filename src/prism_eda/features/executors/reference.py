"""The unoptimised executor: one entity at a time, straight down the tree.

This is deliberately the slow path. Its job is not to be fast but to be
*obviously correct* -- it evaluates each feature for one entity's rows with no
sharing, no batching and no cleverness, which makes it the oracle every
optimised executor is checked against. When the planner and this disagree, this
is right by definition and the planner has a bug.

Because it defines correctness, the semantics chosen here are the module's
semantics, and the non-obvious ones are documented where they are made.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from prism_eda.features.ir import Node
from prism_eda.features.tracing import TracedFeature

__all__ = ["execute_reference"]


def _as_str(series: pd.Series) -> pd.Series:
    # ``.astype(str)`` rather than ``.str`` directly: real feature code
    # normalises object columns that mix types, and the alternative silently
    # yields NaN for every non-string. This does render missing values as the
    # literal "nan", which is why ``replace`` exists in the algebra.
    return series.astype(str)


def _safe_div_value(numerator: Any, denominator: Any, default: float) -> float:
    """Divide, falling back to ``default`` rather than producing inf or NaN.

    The fallback covers a zero, missing or infinite denominator and a missing
    numerator. Feature code guards division constantly; leaving the rule to
    each author is how one of them ends up returning inf in production.
    """
    try:
        den = float(denominator)
        num = float(numerator)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(den) or den == 0.0 or not math.isfinite(num):
        return default
    result = num / den
    return result if math.isfinite(result) else default


def _entropy(values: pd.Series) -> float:
    """Shannon entropy, base 2, of a value distribution. Empty is 0.0."""
    if len(values) == 0:
        return 0.0
    counts = values.value_counts(normalize=True).to_numpy(dtype="float64")
    if counts.size == 0:
        return 0.0
    return float(-np.sum(counts * np.log2(counts + 1e-12)))


def _aggregate(func: str, values: pd.Series) -> Any:
    """Reduce a masked column to one number.

    Empty-input behaviour follows pandas so that a feature written against a
    sparse entity behaves the way its author would predict from a notebook:
    counts are 0, sum is 0.0, and the rest are NaN and left for the feature's
    declared default to resolve.
    """
    if func == "count":
        return float(values.count())
    if func == "nunique":
        return float(values.nunique())
    if func == "mode_count":
        if values.count() == 0:
            return 0.0
        return float(values.value_counts().max())
    if func == "entropy":
        return _entropy(values.dropna())
    if len(values) == 0:
        return 0.0 if func == "sum" else float("nan")
    if func == "sum":
        return float(values.sum())
    if func == "mean":
        return float(values.mean())
    if func == "median":
        return float(values.median())
    if func == "std":
        # ddof=1, matching pandas: a single observation has undefined spread
        # rather than zero spread.
        return float(values.std()) if values.count() > 1 else float("nan")
    if func == "min":
        return float(values.min()) if values.count() else float("nan")
    if func == "max":
        return float(values.max()) if values.count() else float("nan")
    raise ValueError(f"Unsupported aggregate {func!r}")


def _run_length_max(condition: pd.Series) -> float:
    """Longest run of consecutive True values, in the order given."""
    best = current = 0
    for flag in condition.fillna(False).astype(bool).to_numpy():
        current = current + 1 if flag else 0
        best = max(best, current)
    return float(best)


class _GroupEvaluator:
    """Evaluates IR for the rows of exactly one entity."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        time: str | None,
        anchor: pd.Timestamp | None = None,
    ) -> None:
        self._frame = frame
        self._time = time
        self._anchor = anchor
        self._cache: dict[str, Any] = {}

    def evaluate(self, target: Node) -> Any:
        cached = self._cache.get(target.id)
        if cached is not None or target.id in self._cache:
            return cached
        value = self._dispatch(target)
        self._cache[target.id] = value
        return value

    def _anchor_stamp(self) -> Any:
        """Where every lookback ends for this entity.

        With a declared reference time that is the decision moment; without
        one it is the entity's own latest row. The difference is the whole of
        point-in-time correctness: the second is right at serving time, where
        the last row *is* the event being scored, and wrong for a backfill.
        """
        if self._anchor is not None:
            return self._anchor
        return self._frame[self._time].max()  # type: ignore[index]

    def _mask(self, window: Node) -> pd.Series:
        kind = window.param("kind")
        index = self._frame.index
        if self._time is None:
            if kind != "all":  # pragma: no cover - guarded at declaration
                raise ValueError("A time-based window needs a time column")
            return pd.Series(True, index=index)

        stamps = self._frame[self._time]
        if len(stamps) == 0:
            return pd.Series(True, index=index)
        anchor = self._anchor_stamp()

        if kind == "all":
            # "All history" still means all history up to the decision moment.
            if self._anchor is None:
                return pd.Series(True, index=index)
            return stamps <= anchor
        if kind == "days":
            start = anchor - pd.Timedelta(days=float(window.param("days")))
            return (stamps >= start) & (stamps <= anchor)
        if kind == "last_k":
            k = int(window.param("k"))
            eligible = stamps <= anchor
            mask = pd.Series(False, index=index)
            positions = index[eligible.to_numpy()]
            if len(positions):
                mask.loc[positions[-k:]] = True
            return mask
        raise ValueError(f"Unsupported window kind {kind!r}")

    def _dispatch(self, target: Node) -> Any:  # noqa: PLR0911, PLR0912
        op = target.op
        kids = target.children

        if op == "column":
            return self._frame[target.param("name")]
        if op == "literal":
            return target.param("value")
        if op == "window":
            return self._mask(target)

        if op == "agg":
            window, value, where = (self.evaluate(child) for child in kids)
            selected = self._select(value, self._combine(window, where))
            return _aggregate(str(target.param("func")), selected)

        if op == "run_length_max":
            window, condition = (self.evaluate(child) for child in kids)
            mask = pd.Series(window, index=self._frame.index).fillna(False)
            series = self._series(condition)
            return _run_length_max(series[mask.astype(bool)])

        if op == "safe_div":
            numerator, denominator = (self.evaluate(child) for child in kids)
            return _safe_div_value(
                numerator, denominator, float(target.param("default", 0.0))
            )
        if op == "guard":
            value = self.evaluate(kids[0])
            default = float(target.param("default", 0.0))
            try:
                number = float(value)
            except (TypeError, ValueError):
                return default
            return number if math.isfinite(number) else default

        if op == "opaque":
            raise ValueError("Opaque nodes are executed by the caller")

        if len(kids) == 1:
            return self._unary(op, self.evaluate(kids[0]), target)
        if len(kids) == 2:
            return self._binary(op, self.evaluate(kids[0]), self.evaluate(kids[1]))
        raise ValueError(f"Unsupported operation {op!r}")

    # -- helpers -----------------------------------------------------------
    def _series(self, value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            return value
        return pd.Series(value, index=self._frame.index)

    def _combine(self, window: Any, where: Any) -> pd.Series:
        mask = self._series(window).fillna(False).astype(bool)
        clause = self._series(where).fillna(False).astype(bool)
        return mask & clause

    def _select(self, value: Any, mask: pd.Series) -> pd.Series:
        return self._series(value)[mask]

    def _unary(self, op: str, value: Any, target: Node) -> Any:
        if op == "isin":
            return self._series(value).isin(set(target.param("values", ())))
        if op == "replace":
            mapping = dict(target.param("mapping", ()))
            return self._series(value).replace(mapping)
        if op in {"lag", "lead"}:
            periods = int(target.param("periods", 1))
            shift = periods if op == "lag" else -periods
            return self._series(value).shift(shift)
        if op == "diff":
            return self._series(value).diff()
        if op == "upper":
            return _as_str(self._series(value)).str.upper()
        if op == "lower":
            return _as_str(self._series(value)).str.lower()
        if op == "strip":
            return _as_str(self._series(value)).str.strip()
        if op == "notna":
            return self._series(value).notna()
        if op == "isna":
            return self._series(value).isna()
        if op == "invert":
            return ~self._series(value).fillna(False).astype(bool)
        if op == "log1p":
            return float(np.log1p(value)) if np.isscalar(value) else np.log1p(value)
        if op == "abs":
            return abs(value)
        if op == "neg":
            return -value
        if op == "floor_day":
            return self._series(value).dt.floor("D")
        if op == "floor_hour":
            return self._series(value).dt.floor("h")
        if op == "hour":
            return self._series(value).dt.hour
        if op == "total_seconds":
            return self._series(value).dt.total_seconds()
        raise ValueError(f"Unsupported unary operation {op!r}")

    def _binary(self, op: str, left: Any, right: Any) -> Any:  # noqa: PLR0911
        if op == "fillna":
            return self._series(left).fillna(right)
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "gt":
            return left > right
        if op == "ge":
            return left >= right
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        if op == "div":
            return left / right
        if op == "mod":
            return left % right
        if op == "and":
            return self._boolean(left) & self._boolean(right)
        if op == "or":
            return self._boolean(left) | self._boolean(right)
        raise ValueError(f"Unsupported binary operation {op!r}")

    def _boolean(self, value: Any) -> Any:
        if isinstance(value, pd.Series):
            return value.fillna(False).astype(bool)
        return bool(value)


def execute_reference(
    features: Sequence[TracedFeature],
    frame: pd.DataFrame,
    *,
    entity: str,
    time: str | None = None,
    reference_time: str | None = None,
) -> pd.DataFrame:
    """Compute every feature the slow, obviously-correct way.

    Entities appear in first-appearance order, which is deterministic for a
    given input without requiring the entity key to be sortable.
    """
    ordered = frame
    if time is not None:
        # Stable sort so rows sharing a timestamp keep their input order, which
        # keeps sequential features reproducible rather than tie-break dependent.
        ordered = frame.sort_values(time, kind="stable")

    names = [item.name for item in features]
    rows: dict[Any, dict[str, float]] = {}
    for key, group in ordered.groupby(entity, sort=False):
        anchor = group[reference_time].max() if reference_time is not None else None
        evaluator = _GroupEvaluator(group, time=time, anchor=anchor)
        values: dict[str, float] = {}
        for item in features:
            if item.kind == "opaque":
                function = item.metadata.get("fn")
                if function is None:  # pragma: no cover - hand-built plans only
                    raise ValueError(
                        f"Opaque feature {item.name!r} has no function attached"
                    )
                raw = function(group)
            else:
                raw = evaluator.evaluate(item.node)
            values[item.name] = _finalise(raw, item.default)
        rows[key] = values

    result = pd.DataFrame.from_dict(rows, orient="index", columns=names)
    result.index.name = entity
    return result


def _finalise(value: Any, default: float) -> float:
    """Apply the feature's declared default to a missing or non-finite result.

    Both executors do this, so a feature's contract value is a property of the
    definition rather than of which backend produced it.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)
