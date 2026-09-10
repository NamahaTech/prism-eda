"""The operations a feature author writes.

These are thin proxies. Calling one does not compute anything; it records a
node in the intermediate representation. That is what makes the planner
possible: by the time a feature function has returned, we hold an exact record
of what it asked for rather than an opinion about what its source code meant.

Two levels of value exist, and the distinction is enforced rather than implied.
An :class:`Expr` at ``Level.ROW`` is a vector over the entity's rows; at
``Level.ENTITY`` it is one number for the entity. A feature must end at
``Level.ENTITY`` — one row per entity is the whole point of the output.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping
from typing import Any

from prism_eda.features.ir import Level, Node, node

__all__ = [
    "Expr",
    "Window",
    "distinct_ratio",
    "guard",
    "lag",
    "lead",
    "log1p",
    "safe_div",
    "time_delta",
    "top_share",
]


class TracingError(TypeError):
    """Raised when a feature function does something tracing cannot record."""


def _lit(value: Any) -> Node:
    return node("literal", value=value)


def _as_node(value: Any) -> Node:
    if isinstance(value, Expr):
        return value.node
    if isinstance(value, Window):
        return value.node
    return _lit(value)


def _wrap(value: Any) -> Expr:
    return value if isinstance(value, Expr) else Expr(_as_node(value))


class _StrAccessor:
    """``.str`` namespace, mirroring the pandas idiom feature authors know."""

    __slots__ = ("_owner",)

    def __init__(self, owner: Expr) -> None:
        self._owner = owner

    def upper(self) -> Expr:
        return self._owner._unary("upper")

    def lower(self) -> Expr:
        return self._owner._unary("lower")

    def strip(self) -> Expr:
        return self._owner._unary("strip")


class _DtAccessor:
    """``.dt`` namespace for the time derivations the algebra supports."""

    __slots__ = ("_owner",)

    def __init__(self, owner: Expr) -> None:
        self._owner = owner

    def floor(self, freq: str) -> Expr:
        normalized = freq.strip().lower()
        if normalized in {"d", "day"}:
            return self._owner._unary("floor_day")
        if normalized in {"h", "hour"}:
            return self._owner._unary("floor_hour")
        raise ValueError(
            f"Unsupported floor frequency {freq!r}; the algebra supports 'D' and 'h'"
        )

    @property
    def hour(self) -> Expr:
        return self._owner._unary("hour")

    def total_seconds(self) -> Expr:
        return self._owner._unary("total_seconds")


class Expr:
    """A traced value: either a per-row vector or a per-entity scalar."""

    __slots__ = ("_node",)

    def __init__(self, ir_node: Node) -> None:
        self._node = ir_node

    @property
    def node(self) -> Node:
        """The IR node this expression records."""
        return self._node

    @property
    def level(self) -> Level:
        """Whether this value is per-row or per-entity."""
        return self._node.level

    # -- accessors ---------------------------------------------------------
    @property
    def str(self) -> _StrAccessor:  # noqa: A003 - mirrors the pandas accessor
        return _StrAccessor(self)

    @property
    def dt(self) -> _DtAccessor:
        return _DtAccessor(self)

    # -- construction helpers ---------------------------------------------
    def _unary(self, op: builtins.str) -> Expr:
        return Expr(node(op, self._node))

    def _binary(self, op: builtins.str, other: Any, *, flip: bool = False) -> Expr:
        left, right = self._node, _as_node(other)
        if flip:
            left, right = right, left
        return Expr(node(op, left, right))

    # -- comparison --------------------------------------------------------
    # These return expressions rather than booleans. That is the standard
    # tracing trade and the reason __hash__ is defined explicitly below.
    def __eq__(self, other: Any) -> Expr:  # type: ignore[override]
        return self._binary("eq", other)

    def __ne__(self, other: Any) -> Expr:  # type: ignore[override]
        return self._binary("ne", other)

    def __lt__(self, other: Any) -> Expr:
        return self._binary("lt", other)

    def __le__(self, other: Any) -> Expr:
        return self._binary("le", other)

    def __gt__(self, other: Any) -> Expr:
        return self._binary("gt", other)

    def __ge__(self, other: Any) -> Expr:
        return self._binary("ge", other)

    def __hash__(self) -> int:
        return hash(self._node.id)

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other: Any) -> Expr:
        return self._binary("add", other)

    def __radd__(self, other: Any) -> Expr:
        return self._binary("add", other, flip=True)

    def __sub__(self, other: Any) -> Expr:
        return self._binary("sub", other)

    def __rsub__(self, other: Any) -> Expr:
        return self._binary("sub", other, flip=True)

    def __mul__(self, other: Any) -> Expr:
        return self._binary("mul", other)

    def __rmul__(self, other: Any) -> Expr:
        return self._binary("mul", other, flip=True)

    def __truediv__(self, other: Any) -> Expr:
        return self._binary("div", other)

    def __rtruediv__(self, other: Any) -> Expr:
        return self._binary("div", other, flip=True)

    def __mod__(self, other: Any) -> Expr:
        return self._binary("mod", other)

    def __and__(self, other: Any) -> Expr:
        return self._binary("and", other)

    def __rand__(self, other: Any) -> Expr:
        return self._binary("and", other, flip=True)

    def __or__(self, other: Any) -> Expr:
        return self._binary("or", other)

    def __ror__(self, other: Any) -> Expr:
        return self._binary("or", other, flip=True)

    def __invert__(self) -> Expr:
        return self._unary("invert")

    def __neg__(self) -> Expr:
        return self._unary("neg")

    def __abs__(self) -> Expr:
        return self._unary("abs")

    # -- predicates --------------------------------------------------------
    def isin(self, values: Iterable[Any]) -> Expr:
        return Expr(node("isin", self._node, values=tuple(sorted(set(values)))))

    def replace(self, mapping: Mapping[Any, Any]) -> Expr:
        pairs = tuple(sorted(mapping.items(), key=lambda item: repr(item[0])))
        return Expr(node("replace", self._node, mapping=pairs))

    def fillna(self, value: Any) -> Expr:
        return Expr(node("fillna", self._node, _as_node(value)))

    def notna(self) -> Expr:
        return self._unary("notna")

    def isna(self) -> Expr:
        return self._unary("isna")

    # -- the tracing edge --------------------------------------------------
    def __bool__(self) -> bool:
        raise TracingError(
            "A feature expression cannot be used in a Python 'if', 'and', 'or', "
            "'not', or any other test, because at trace time its value is not "
            "known yet -- only its shape is.\n"
            "  Branching on the SCHEMA is fine: 'if \"direction\" in fs.columns'.\n"
            "  Branching on the DATA is not:   'if amount > 100'.\n"
            "Use a masked aggregate for the second case -- w.count(where=amount > 100) "
            "-- or, if the logic genuinely cannot be expressed in the algebra, "
            "declare the feature with @fs.opaque and it will run unplanned."
        )

    def __repr__(self) -> builtins.str:  # pragma: no cover - debugging aid
        return f"Expr({self._node.op}, level={self._node.level.value})"


class Window:
    """A row subset of one entity's history, and the aggregates over it.

    Windows are declared once on the :class:`~prism_eda.features.FeatureSet`
    and referenced by name, so two features asking for "the last 30 days"
    provably ask for the same mask rather than two masks that happen to agree.
    """

    __slots__ = ("_node",)

    def __init__(self, ir_node: Node) -> None:
        self._node = ir_node

    @property
    def node(self) -> Node:
        """The IR node for this window's row mask."""
        return self._node

    def _agg(self, func: str, value: Any, where: Any) -> Expr:
        where_node = _as_node(True) if where is None else _as_node(where)
        value_node = _as_node(1) if value is None else _as_node(value)
        return Expr(node("agg", self._node, value_node, where_node, func=func))

    def count(self, value: Any = None, *, where: Any = None) -> Expr:
        """Number of rows in the window (optionally satisfying ``where``)."""
        return self._agg("count", value, where)

    def sum(self, value: Any, *, where: Any = None) -> Expr:
        return self._agg("sum", value, where)

    def mean(self, value: Any, *, where: Any = None) -> Expr:
        return self._agg("mean", value, where)

    def median(self, value: Any, *, where: Any = None) -> Expr:
        return self._agg("median", value, where)

    def std(self, value: Any, *, where: Any = None) -> Expr:
        return self._agg("std", value, where)

    def min(self, value: Any, *, where: Any = None) -> Expr:
        return self._agg("min", value, where)

    def max(self, value: Any, *, where: Any = None) -> Expr:
        return self._agg("max", value, where)

    def nunique(self, value: Any, *, where: Any = None) -> Expr:
        return self._agg("nunique", value, where)

    def mode_count(self, value: Any, *, where: Any = None) -> Expr:
        """Size of the largest group of equal values."""
        return self._agg("mode_count", value, where)

    def entropy(self, value: Any, *, where: Any = None) -> Expr:
        """Shannon entropy, base 2, of the value distribution."""
        return self._agg("entropy", value, where)

    def run_length_max(self, condition: Any, *, where: Any = None) -> Expr:
        """Longest run of consecutive rows satisfying ``condition``.

        Time order within the entity is part of the definition, so this reads
        the plan's sort order rather than establishing its own.
        """
        target = condition if where is None else Expr(_as_node(condition)) & where
        return Expr(node("run_length_max", self._node, _as_node(target)))

    def __bool__(self) -> bool:
        raise TracingError("A window cannot be used in a Python conditional.")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        kind = self._node.param("kind")
        return f"Window({self._node.param('name')!r}, kind={kind})"


# -- free functions --------------------------------------------------------


def safe_div(numerator: Any, denominator: Any, default: float = 0.0) -> Expr:
    """Divide, yielding ``default`` when the denominator is zero or missing.

    Division guarded this way is so common in feature code that leaving it to
    the author guarantees three spellings of it in one file, one of which
    returns ``inf`` in production.
    """
    return Expr(
        node("safe_div", _as_node(numerator), _as_node(denominator), default=default)
    )


def guard(value: Any, default: float = 0.0) -> Expr:
    """Replace a non-finite result (NaN, inf, -inf) with ``default``."""
    return Expr(node("guard", _as_node(value), default=default))


def log1p(value: Any) -> Expr:
    """Natural log of one plus the value."""
    return Expr(node("log1p", _as_node(value)))


def lag(value: Any, periods: int = 1) -> Expr:
    """The value ``periods`` rows earlier, within the entity.

    Never reads across an entity boundary; rows without a predecessor are
    missing rather than borrowed from the neighbouring entity.
    """
    return Expr(node("lag", _as_node(value), periods=int(periods)))


def lead(value: Any, periods: int = 1) -> Expr:
    """The value ``periods`` rows later, within the entity."""
    return Expr(node("lead", _as_node(value), periods=int(periods)))


def time_delta(value: Any) -> Expr:
    """Difference from the previous row, within the entity."""
    return Expr(node("diff", _as_node(value)))


def top_share(window: Window, value: Any, *, where: Any = None) -> Expr:
    """Share of the window held by its single most common value."""
    return safe_div(
        window.mode_count(value, where=where),
        window.count(where=where),
    )


def distinct_ratio(
    window: Window, value: Any, *, where: Any = None, default: float = 1.0
) -> Expr:
    """Distinct values as a share of rows."""
    return safe_div(
        window.nunique(value, where=where),
        window.count(where=where),
        default=default,
    )
