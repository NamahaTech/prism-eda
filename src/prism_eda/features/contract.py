"""The feature contract: what is produced, in what order, and what fills a gap.

A model consumes a feature vector by position. Train on one column order and
serve another and nothing raises -- every value is a plausible float in the
wrong slot, and the model quietly scores garbage. Production feature configs
therefore pin the order explicitly, and the ones that do usually carry a
comment warning the next person not to sort it.

So order is not a rendering detail here, it is part of the contract, along with
the value that stands in when a feature cannot be computed. Both are properties
of the definition rather than of whichever backend happened to run it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prism_eda.features.ir import iter_nodes
from prism_eda.features.tracing import TracedFeature

__all__ = ["ContractViolation", "FeatureContract", "FeatureSpec", "build_contract"]


class ContractViolation(ValueError):
    """Raised when a frame does not satisfy a feature contract."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One output column's contract."""

    name: str
    default: float
    kind: str = "feature"
    reads: tuple[str, ...] = ()
    doc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "numeric",
            "default_val": self.default,
            "kind": self.kind,
            "reads": list(self.reads),
        }


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """The complete, exportable description of a feature set's output."""

    entity: str
    time: str | None
    features: tuple[FeatureSpec, ...]
    required_columns: tuple[str, ...]
    reference_time: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Output columns, in the order a consumer must expect them."""
        return tuple(item.name for item in self.features)

    @property
    def defaults(self) -> dict[str, float]:
        """Each feature's fallback value."""
        return {item.name: item.default for item in self.features}

    def missing_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        """Source columns the contract needs that ``frame`` does not have."""
        available = set(map(str, frame.columns))
        return tuple(
            column for column in self.required_columns if column not in available
        )

    def validate_source(self, frame: pd.DataFrame) -> None:
        """Raise unless ``frame`` carries every column the features read."""
        missing = self.missing_columns(frame)
        if missing:
            raise ContractViolation(
                "The frame is missing column(s) this feature set reads: "
                + ", ".join(missing)
            )

    def validate_output(self, frame: pd.DataFrame) -> None:
        """Raise unless ``frame`` is exactly this contract's columns, in order.

        Order is checked, not just membership. A permuted vector is the failure
        this method exists to catch, and it is invisible to a set comparison.
        """
        produced = [str(column) for column in frame.columns]
        expected = list(self.names)
        if produced == expected:
            return
        if sorted(produced) == sorted(expected):
            raise ContractViolation(
                "The feature columns are correct but out of contract order. "
                f"Expected {expected[:3]}... got {produced[:3]}... \u2014 a model "
                "consuming this by position would receive a permuted vector."
            )
        missing = [name for name in expected if name not in produced]
        extra = [name for name in produced if name not in expected]
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise ContractViolation(
            "The frame does not match the feature contract (" + "; ".join(details) + ")"
        )

    def to_dict(self) -> dict[str, Any]:
        """A machine-readable contract, shaped like a serving feature config."""
        return {
            "entity": self.entity,
            "time": self.time,
            "reference_time": self.reference_time,
            # Named after the flag real configs carry, and false for the same
            # reason: features are consumed in this order, and sorting them
            # would feed every model a permuted vector.
            "alphabetical_sorting": False,
            "required_columns": list(self.required_columns),
            "features": [item.to_dict() for item in self.features],
        }

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the contract and return its path."""
        target = Path(path)
        target.write_text(
            json.dumps(self.to_dict(), indent=indent, sort_keys=False),
            encoding="utf-8",
        )
        return target

    def describe(self) -> str:
        """A readable summary of the contract."""
        lines = [
            f"Feature contract: {len(self.features)} feature(s) "
            f"over entity={self.entity!r}"
            + (f", time={self.time!r}" if self.time else ""),
            "  reads: " + ", ".join(self.required_columns),
        ]
        non_zero = [item for item in self.features if item.default != 0.0]
        if non_zero:
            described = ", ".join(f"{i.name}={i.default:g}" for i in non_zero)
            lines.append(f"  non-zero defaults: {described}")
        opaque = [item.name for item in self.features if item.kind == "opaque"]
        if opaque:
            lines.append("  opaque: " + ", ".join(opaque))
        return "\n".join(lines)


def build_contract(
    features: tuple[TracedFeature, ...],
    *,
    entity: str,
    time: str | None,
    reference_time: str | None = None,
) -> FeatureContract:
    """Derive the contract from traced features.

    The read set comes from the IR rather than from the author, so it cannot
    drift from what the features actually touch.
    """
    specs: list[FeatureSpec] = []
    required: list[str] = [entity]
    if time is not None:
        required.append(time)
    if reference_time is not None:
        required.append(reference_time)

    for item in features:
        if item.kind == "opaque":
            reads = tuple(item.requires)
        else:
            reads = tuple(
                sorted(
                    {
                        str(reached.param("name"))
                        for reached in iter_nodes(item.node)
                        if reached.op == "column"
                    }
                )
            )
        for column in reads:
            if column not in required:
                required.append(column)
        specs.append(
            FeatureSpec(
                name=item.name,
                default=item.default,
                kind=item.kind,
                reads=reads,
                doc=item.doc,
            )
        )

    return FeatureContract(
        entity=entity,
        time=time,
        features=tuple(specs),
        required_columns=tuple(required),
        reference_time=reference_time,
    )
