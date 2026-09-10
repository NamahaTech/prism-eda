"""Proving the planned answer is the answer.

Two distinct claims live here and they are deliberately not conflated.

**Planner soundness** is the claim that the optimised plan equals the
unoptimised one. It is checked on every plan build, because a planner that
quietly returns different numbers than the code it replaced is worse than no
planner: the failure is invisible, and it surfaces as a model that mysteriously
performs worse in production than it did in training.

**Migration fidelity** is the claim that prism's version of a feature equals
the hand-written pandas it is replacing. No amount of internal consistency
establishes that, so it is a separate, explicit check against the caller's own
function -- which is what makes porting an existing feature set something you
can do one feature at a time with evidence at each step.

Entities are sampled, never rows. A feature is defined over an entity's whole
history, so dropping rows would not sample the computation -- it would change
it, and then compare two different questions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from prism_eda.exceptions import PrismEDAError
from prism_eda.results import SamplingRecord

__all__ = [
    "DEFAULT_RTOL",
    "DEFAULT_SAMPLE_ENTITIES",
    "FeatureAgreement",
    "VerificationError",
    "VerificationReport",
    "compare_frames",
    "sample_entities",
]

# Tight enough to catch a logic error, loose enough to survive the reassociation
# that legitimately differs between a per-entity loop and a grouped reduction:
# log1p, std, entropy and division all move low-order bits when the order of
# operations changes.
DEFAULT_RTOL = 1e-9

# The reference executor costs roughly a millisecond per entity, so a few
# hundred entities keeps always-on verification well under a second while still
# covering every feature.
DEFAULT_SAMPLE_ENTITIES = 200


class VerificationError(PrismEDAError):
    """Raised when a planned result disagrees with the answer it must match."""


@dataclass(frozen=True, slots=True)
class FeatureAgreement:
    """How one feature compared between two executions."""

    feature: str
    matched: bool
    max_difference: float
    mismatched_entities: int
    checked_entities: int
    examples: tuple[dict[str, Any], ...] = ()

    def describe(self) -> str:
        if self.matched:
            return f"{self.feature}: matched across {self.checked_entities} entities"
        return (
            f"{self.feature}: differs on {self.mismatched_entities} of "
            f"{self.checked_entities} entities, largest relative difference "
            f"{self.max_difference:.3g}"
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """The outcome of one comparison, whether or not it passed."""

    agreements: tuple[FeatureAgreement, ...]
    checked_entities: int
    rtol: float
    kind: str = "planner_soundness"
    sampling: SamplingRecord | None = None
    missing_features: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every compared feature agreed."""
        return all(item.matched for item in self.agreements)

    @property
    def failures(self) -> tuple[FeatureAgreement, ...]:
        """The features that disagreed, worst first."""
        return tuple(
            sorted(
                (item for item in self.agreements if not item.matched),
                key=lambda item: -item.max_difference,
            )
        )

    def describe(self) -> str:
        """A readable account, leading with what failed."""
        headline = (
            f"{len(self.agreements)} feature(s) checked on {self.checked_entities} "
            f"entities at rtol={self.rtol:g}"
        )
        lines = [
            f"{'PASS' if self.ok else 'FAIL'}: {headline}",
        ]
        if self.missing_features:
            lines.append(
                "  not produced by the reference: " + ", ".join(self.missing_features)
            )
        for item in self.failures:
            lines.append("  " + item.describe())
            for example in item.examples:
                lines.append(
                    f"    entity {example['entity']!r}: "
                    f"expected {example['expected']:.10g}, "
                    f"got {example['actual']:.10g}"
                )
        if self.ok and not self.failures:
            lines.append("  every feature agreed")
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        """Raise :class:`VerificationError` when anything disagreed."""
        if self.ok:
            return
        raise VerificationError(self.describe())


def sample_entities(
    frame: pd.DataFrame,
    *,
    entity: str,
    limit: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, SamplingRecord | None]:
    """Take every row of up to ``limit`` entities, chosen deterministically.

    Whole entities, because a feature reads an entity's whole history: keeping
    a random subset of its rows would silently change what is being computed
    rather than reducing how much of it is checked.
    """
    keys = frame[entity].drop_duplicates()
    if limit <= 0 or len(keys) <= limit:
        return frame, None
    chosen = keys.sample(n=limit, random_state=seed)
    subset = frame[frame[entity].isin(set(chosen))]
    record = SamplingRecord(
        operation="feature_plan_verification",
        source_rows=len(frame),
        sampled_rows=len(subset),
        strategy="deterministic_entity_sample",
        seed=seed,
        reason="entity_count_exceeds_verification_budget",
        limitations=(
            f"Verification covered {limit:,} of {len(keys):,} entities. A "
            "disagreement confined to entities outside the sample would not "
            "have been seen.",
        ),
    )
    return subset, record


def _relative_difference(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Relative where there is a magnitude to be relative to, absolute at zero."""
    scale = np.maximum(np.abs(expected), np.abs(actual))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(expected - actual) / scale
    return np.where(scale > 0, relative, np.abs(expected - actual))


def compare_frames(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    rtol: float = DEFAULT_RTOL,
    kind: str = "planner_soundness",
    sampling: SamplingRecord | None = None,
    features: Sequence[str] | None = None,
    max_examples: int = 3,
) -> VerificationReport:
    """Compare two feature frames feature by feature."""
    names = list(features) if features is not None else list(actual.columns)
    shared_index = actual.index.intersection(expected.index)
    missing = tuple(name for name in names if name not in expected.columns)

    agreements: list[FeatureAgreement] = []
    for name in names:
        if name not in expected.columns or name not in actual.columns:
            continue
        left = pd.to_numeric(
            expected.loc[shared_index, name], errors="coerce"
        ).to_numpy(dtype="float64")
        right = pd.to_numeric(actual.loc[shared_index, name], errors="coerce").to_numpy(
            dtype="float64"
        )
        both_nan = np.isnan(left) & np.isnan(right)
        difference = _relative_difference(left, right)
        differs = (difference > rtol) & ~both_nan
        examples: list[dict[str, Any]] = []
        for position in np.flatnonzero(differs)[:max_examples]:
            examples.append(
                {
                    "entity": shared_index[position],
                    "expected": float(left[position]),
                    "actual": float(right[position]),
                }
            )
        largest = float(np.nanmax(difference[~both_nan])) if (~both_nan).any() else 0.0
        agreements.append(
            FeatureAgreement(
                feature=name,
                matched=not bool(differs.any()),
                max_difference=largest if math.isfinite(largest) else float("inf"),
                mismatched_entities=int(differs.sum()),
                checked_entities=len(shared_index),
                examples=tuple(examples),
            )
        )

    return VerificationReport(
        agreements=tuple(agreements),
        checked_entities=len(shared_index),
        rtol=rtol,
        kind=kind,
        sampling=sampling,
        missing_features=missing,
    )


def oracle_frame(
    function: Callable[[pd.DataFrame], Any],
    frame: pd.DataFrame,
    *,
    entity: str,
    time: str | None,
) -> pd.DataFrame:
    """Run the caller's existing per-entity function to build a comparison frame.

    Their function is the shape real feature code already has: one entity's
    rows in, a mapping of feature name to number out.
    """
    ordered = frame.sort_values(time, kind="stable") if time is not None else frame
    rows: dict[Any, Any] = {}
    for key, group in ordered.groupby(entity, sort=False):
        produced = function(group)
        if isinstance(produced, pd.Series):
            produced = produced.to_dict()
        if not isinstance(produced, dict):
            raise TypeError(
                "The function passed to verify_against must return a mapping of "
                f"feature name to value for one entity's rows; got "
                f"{type(produced).__name__}"
            )
        rows[key] = produced
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = entity
    return result
