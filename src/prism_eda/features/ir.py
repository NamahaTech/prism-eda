"""The feature intermediate representation.

Every feature a user writes is traced (never parsed) into a tree of :class:`Node`
values. A node is immutable and carries a **structural id**: a hash over its
operation, its parameters, and the ids of its children. Two subexpressions that
compute the same thing therefore land on the same id no matter which feature
function produced them, which is what lets the planner collapse them without
proving anything about the Python they came from.

The op vocabulary is deliberately closed. It was derived by enumerating every
operation in a real 136-feature production contract, and it is the module's
expressiveness bar: anything outside it belongs in an opaque node rather than
being approximated by something that nearly fits.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "AGGREGATE_FUNCS",
    "Level",
    "Node",
    "OpSpec",
    "children_of",
    "iter_nodes",
    "node",
    "param",
    "walk",
]


class Level(StrEnum):
    """Where a value lives.

    ``ROW`` is a vector with one element per input row. ``ENTITY`` is a single
    value per entity. The distinction is not cosmetic: it decides whether the
    planner can share a node across features and whether an executor may batch
    it into a grouped aggregation.
    """

    ROW = "row"
    ENTITY = "entity"


# Aggregate functions. ``mode_count`` (size of the largest value group) and
# ``entropy`` (Shannon, base 2, over the value distribution) are here rather
# than in a separate family because they reduce a window to one number per
# entity exactly as ``sum`` does; splitting them out bought nothing but a
# second dispatch table.
AGGREGATE_FUNCS = frozenset(
    {
        "count",
        "sum",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "nunique",
        "mode_count",
        "entropy",
    }
)

# Row-level unary operations, keyed by op name.
_UNARY_OPS = frozenset(
    {
        "upper",
        "lower",
        "strip",
        "notna",
        "isna",
        "invert",
        "log1p",
        "abs",
        "neg",
        "floor_day",
        "floor_hour",
        "hour",
        "total_seconds",
    }
)

# Row-level binary operations. Comparison and logical ops yield booleans;
# arithmetic ops yield numbers. Both promote to ENTITY when either side is.
_BINARY_OPS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "add",
        "sub",
        "mul",
        "div",
        "mod",
        "and",
        "or",
    }
)

# Window kinds. ``all`` is the entity's whole history; ``days`` is an as-of
# lookback anchored per entity; ``last_k`` is the final k rows in time order.
_WINDOW_KINDS = frozenset({"all", "days", "last_k"})


@dataclass(frozen=True, slots=True)
class OpSpec:
    """The shape of one operation: how many children, and at what level."""

    arity: int | None  # None means variadic
    level: Level | None  # None means "promote from children"


# The closed vocabulary. Adding an entry here is the only way to widen the
# algebra, which keeps "what can be planned" answerable by reading one dict.
_OPS: dict[str, OpSpec] = {
    "column": OpSpec(0, Level.ROW),
    "literal": OpSpec(0, Level.ROW),
    "isin": OpSpec(1, None),
    "replace": OpSpec(1, None),
    "fillna": OpSpec(2, None),
    # Ordered row ops. Group-boundary aware by construction: an executor must
    # never let these read across an entity edge.
    "lag": OpSpec(1, Level.ROW),
    "lead": OpSpec(1, Level.ROW),
    "diff": OpSpec(1, Level.ROW),
    "window": OpSpec(0, Level.ROW),
    # (window, value, where)
    "agg": OpSpec(3, Level.ENTITY),
    # (window, value)
    "run_length_max": OpSpec(2, Level.ENTITY),
    # (numerator, denominator)
    "safe_div": OpSpec(2, None),
    "guard": OpSpec(1, None),
    "opaque": OpSpec(None, Level.ENTITY),
}
# Unary ops promote rather than forcing ROW: log1p of a per-entity scalar is a
# per-entity scalar, and hardcoding ROW here made every derived ratio fail to
# type-check as a feature. The genuinely row-only ops (lag/lead/diff, which need
# an ordering, and window, which produces one) keep an explicit ROW above.
for _op in _UNARY_OPS:
    _OPS[_op] = OpSpec(1, None)
for _op in _BINARY_OPS:
    _OPS[_op] = OpSpec(2, None)


@dataclass(frozen=True, slots=True)
class Node:
    """One operation in a feature expression.

    ``id`` is derived, not supplied. Build nodes with :func:`node`.
    """

    id: str
    op: str
    level: Level
    params: tuple[tuple[str, Any], ...]
    children: tuple[Node, ...]

    def param(self, name: str, default: Any = None) -> Any:
        """Return a parameter value, or ``default`` when it is absent."""
        for key, value in self.params:
            if key == name:
                return value
        return default

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.params)
        return f"Node({self.op}, {rendered}, children={len(self.children)})"


def _canonical(value: Any) -> Any:
    """Reduce a parameter value to something JSON can order deterministically."""
    if isinstance(value, frozenset | set):
        return ["__set__", sorted(_canonical(item) for item in value)]
    if isinstance(value, tuple | list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def node(
    op: str,
    *children: Node,
    level: Level | None = None,
    **params: Any,
) -> Node:
    """Build a node, validating it against the closed op vocabulary.

    The id is a hash over ``(op, level, params, child ids)``. Children are part
    of the hash by id rather than by value, so equality is structural all the
    way down and costs one hash per node rather than a deep comparison.
    """
    spec = _OPS.get(op)
    if spec is None:
        known = ", ".join(sorted(_OPS))
        raise ValueError(f"Unknown feature operation {op!r}. Known ops: {known}")
    if spec.arity is not None and len(children) != spec.arity:
        raise ValueError(
            f"Operation {op!r} takes {spec.arity} operand(s), got {len(children)}"
        )
    if op == "agg":
        func = params.get("func")
        if func not in AGGREGATE_FUNCS:
            allowed = ", ".join(sorted(AGGREGATE_FUNCS))
            raise ValueError(f"Unknown aggregate {func!r}. Allowed: {allowed}")
    if op == "window":
        kind = params.get("kind")
        if kind not in _WINDOW_KINDS:
            allowed = ", ".join(sorted(_WINDOW_KINDS))
            raise ValueError(f"Unknown window kind {kind!r}. Allowed: {allowed}")

    resolved = level if level is not None else spec.level
    if resolved is None:
        # Promote: a mixed expression is entity-level, because a per-entity
        # scalar broadcast against a row vector yields a row vector only when
        # every operand is row-level.
        resolved = (
            Level.ENTITY
            if any(child.level is Level.ENTITY for child in children)
            else Level.ROW
        )

    ordered = tuple(sorted(params.items()))
    payload = {
        "op": op,
        "level": resolved.value,
        "params": [[k, _canonical(v)] for k, v in ordered],
        "children": [child.id for child in children],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return Node(
        id=f"fn_{digest}",
        op=op,
        level=resolved,
        params=ordered,
        children=children,
    )


def param(target: Node, name: str, default: Any = None) -> Any:
    """Module-level alias for :meth:`Node.param`."""
    return target.param(name, default)


def children_of(target: Node) -> tuple[Node, ...]:
    """Return a node's children."""
    return target.children


def walk(root: Node) -> list[Node]:
    """Return every distinct node in ``root``, children before parents.

    Deduplicated by structural id, so a subexpression shared by two branches is
    returned once. That ordering is what an executor materialises in.
    """
    seen: dict[str, Node] = {}
    order: list[Node] = []

    def visit(current: Node) -> None:
        if current.id in seen:
            return
        for child in current.children:
            visit(child)
        seen[current.id] = current
        order.append(current)

    visit(root)
    return order


def iter_nodes(roots: Node | Iterable[Node]) -> list[Node]:
    """Return every distinct node across several roots, children first."""
    seen: set[str] = set()
    order: list[Node] = []
    sources: Iterable[Node] = [roots] if isinstance(roots, Node) else roots
    for root in sources:
        for current in walk(root):
            if current.id not in seen:
                seen.add(current.id)
                order.append(current)
    return order
