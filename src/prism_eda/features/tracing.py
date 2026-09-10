"""Turning feature functions into IR by running them, not by reading them.

A feature function is executed once, at plan time, with symbolic proxies
standing in for its inputs. Whatever operations it performs on those proxies
are recorded. This is the same technique JAX and TensorFlow use, and it is
chosen over parsing the function's source for one reason: to share a grouper
between two features the planner must *prove* they group identically, and that
proof is available for free from a recorded operation while being both hard and
unsound to extract from arbitrary pandas source.

The cost is that data-dependent branching cannot work -- at trace time a value
has a shape but no contents. Schema-dependent branching, which is what real
feature code actually does (``if "direction" in fs.columns``), works fine.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from prism_eda.features.algebra import Expr, TracingError, Window
from prism_eda.features.ir import Level, Node, node

__all__ = [
    "FeatureDef",
    "TracedFeature",
    "trace_features",
]


@dataclass(frozen=True, slots=True)
class FeatureDef:
    """A registered but not yet traced feature or intermediate."""

    name: str
    fn: Callable[..., Any]
    kind: str  # "feature" | "derive" | "opaque"
    default: float = 0.0
    doc: str | None = None
    requires: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TracedFeature:
    """A feature reduced to IR, ready for planning."""

    name: str
    node: Node
    default: float
    kind: str = "feature"
    doc: str | None = None
    requires: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _describe(name: str, schema: Sequence[str]) -> str:
    known = ", ".join(sorted(schema))
    return (
        f"Feature parameter {name!r} matches no declared window, no @derive, and "
        f"no column in the frame.\nAvailable columns: {known or '(none)'}"
    )


class _Resolver:
    """Resolves a feature function's parameter names to traced values.

    A parameter name means, in order: a declared window, a declared
    intermediate, or a column of the frame. Resolution is memoised by name so
    two features naming the same intermediate share one subtree rather than
    tracing it twice.
    """

    def __init__(
        self,
        *,
        derives: Mapping[str, FeatureDef],
        windows: Mapping[str, Window],
        schema: Sequence[str],
    ) -> None:
        self._derives = derives
        self._windows = windows
        self._schema = tuple(schema)
        self._cache: dict[str, Any] = {}
        self._in_progress: list[str] = []

    def resolve(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        if name in self._windows:
            value: Any = self._windows[name]
        elif name in self._derives:
            if name in self._in_progress:
                cycle = " -> ".join([*self._in_progress, name])
                raise TracingError(f"Circular @derive dependency: {cycle}")
            self._in_progress.append(name)
            try:
                value = self._call(self._derives[name])
            finally:
                self._in_progress.pop()
        elif name in self._schema:
            value = Expr(node("column", name=name))
        else:
            raise TracingError(_describe(name, self._schema))
        self._cache[name] = value
        return value

    def _call(self, definition: FeatureDef) -> Any:
        signature = inspect.signature(definition.fn)
        kwargs: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                raise TracingError(
                    f"{definition.name!r} uses *args/**kwargs; feature functions "
                    "must name each input explicitly so the planner can see the "
                    "dependency"
                )
            kwargs[parameter.name] = self.resolve(parameter.name)
        result = definition.fn(**kwargs)
        if not isinstance(result, Expr | Window):
            raise TracingError(
                f"{definition.name!r} returned {type(result).__name__}, not a traced "
                "expression. A feature must return a value built from its inputs "
                "using the feature algebra; returning a plain number or a pandas "
                "object means the body computed something eagerly."
            )
        return result


def trace_features(
    *,
    features: Sequence[FeatureDef],
    derives: Mapping[str, FeatureDef],
    windows: Mapping[str, Window],
    schema: Sequence[str],
) -> tuple[TracedFeature, ...]:
    """Trace every registered feature into IR, preserving declaration order.

    Declaration order is the output column order, and that order is a contract:
    a model trained on one permutation and served another silently receives a
    scrambled vector.
    """
    resolver = _Resolver(derives=derives, windows=windows, schema=schema)
    traced: list[TracedFeature] = []
    for definition in features:
        if definition.kind == "opaque":
            missing = [c for c in definition.requires if c not in schema]
            if missing:
                raise TracingError(
                    f"Opaque feature {definition.name!r} requires column(s) "
                    f"{', '.join(missing)}, which the frame does not have"
                )
            traced.append(
                TracedFeature(
                    name=definition.name,
                    node=node(
                        "opaque",
                        name=definition.name,
                        requires=definition.requires,
                    ),
                    default=definition.default,
                    kind="opaque",
                    doc=definition.doc,
                    requires=definition.requires,
                    metadata={"fn": definition.fn},
                )
            )
            continue

        value = resolver._call(definition)
        if isinstance(value, Window):
            raise TracingError(
                f"{definition.name!r} returned a window rather than a value. "
                "Aggregate it first, for example w30d.count()."
            )
        if value.level is not Level.ENTITY:
            raise TracingError(
                f"{definition.name!r} returned a per-row value, but a feature must "
                "be one number per entity. Wrap it in an aggregate -- "
                "w.mean(...), w.count(...), w.max(...) -- to reduce it."
            )
        traced.append(
            TracedFeature(
                name=definition.name,
                node=value.node,
                default=definition.default,
                doc=definition.doc,
            )
        )
    return tuple(traced)
