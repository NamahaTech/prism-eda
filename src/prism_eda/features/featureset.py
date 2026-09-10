"""The feature session: what is declared, in what order, over which entity."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any

from prism_eda.features.algebra import TracingError, Window
from prism_eda.features.ir import node
from prism_eda.features.plan import FeaturePlan
from prism_eda.features.tracing import FeatureDef, TracedFeature, trace_features
from prism_eda.features.verify import DEFAULT_RTOL, DEFAULT_SAMPLE_ENTITIES

__all__ = ["FeatureSet"]

_RESERVED = frozenset({"all"})


class FeatureSet:
    """A named entity, its windows, and the features defined over it.

    The entity is the unit a feature is *written* for -- one account, one
    customer -- while the planner is responsible for evaluating that definition
    across every entity at once. Keeping those two things separate is what lets
    the same definition serve a training backfill and a single-row request.
    """

    def __init__(
        self,
        *,
        entity: str,
        time: str | None = None,
        reference_time: str | None = None,
        target: str | None = None,
    ) -> None:
        if not entity or not entity.strip():
            raise ValueError("A FeatureSet needs an entity column name")
        if reference_time is not None and time is None:
            raise ValueError(
                "reference_time= anchors time windows, so time= is required too"
            )
        self._entity = entity
        self._time = time
        self._reference_time = reference_time
        self._target = target
        # Every window carries its anchor, so the IR says on its face whether
        # a lookback ends at the entity's own last row or at a decision moment
        # the caller supplied.
        self._anchor = "reference" if reference_time else "entity_max"
        self._windows: dict[str, Window] = {
            "all": Window(node("window", kind="all", name="all", anchor=self._anchor))
        }
        self._derives: dict[str, FeatureDef] = {}
        self._features: list[FeatureDef] = []
        self._schema: tuple[str, ...] | None = None

    # -- identity ----------------------------------------------------------
    @property
    def entity(self) -> str:
        """The column identifying an entity."""
        return self._entity

    @property
    def time(self) -> str | None:
        """The column establishing order within an entity, if any."""
        return self._time

    @property
    def reference_time(self) -> str | None:
        """The column holding each entity's decision moment, if declared.

        Without one, a lookback ends at the entity's own most recent row --
        which is correct at serving time, where that row *is* the transaction
        being scored, and quietly wrong for a training backfill assembled
        later, where it is whatever the extract happened to end at. Declaring
        it makes the two agree and lets prism check that the frame does not
        already contain rows from after the decision.
        """
        return self._reference_time

    @property
    def target(self) -> str | None:
        """The outcome column, if declared, so features can be checked for it."""
        return self._target

    @property
    def columns(self) -> tuple[str, ...]:
        """The frame's columns, available while tracing.

        Branching on this is legitimate -- the schema is known at trace time --
        and it is what real feature code does to tolerate an optional column.
        """
        if self._schema is None:
            raise TracingError(
                "fs.columns is only available while a plan is being built, "
                "because the frame's schema is not known before then. Read it "
                "inside a feature function, not at module import time."
            )
        return self._schema

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Declared output columns, in the order they will be produced."""
        return tuple(item.name for item in self._features)

    @property
    def all(self) -> Window:
        """The window covering the entity's whole history."""
        return self._windows["all"]

    # -- declaration -------------------------------------------------------
    def window(
        self,
        name: str,
        *,
        days: float | None = None,
        last_k: int | None = None,
    ) -> Window:
        """Declare a reusable window and return it.

        Declaring a window by name rather than inlining it is what makes
        sharing provable: two features naming ``w30d`` ask for one mask, where
        two inline lookbacks would only happen to agree.
        """
        self._check_name(name, "window")
        if (days is None) == (last_k is None):
            raise ValueError(f"Window {name!r} needs exactly one of days= or last_k=")
        if days is not None:
            if self._time is None:
                raise ValueError(
                    f"Window {name!r} is a time lookback, so the FeatureSet needs "
                    "time='<timestamp column>'"
                )
            if days <= 0:
                raise ValueError(f"Window {name!r} needs days > 0")
            created = Window(node("window", kind="days", days=float(days), name=name))
        else:
            assert last_k is not None
            if self._time is None:
                raise ValueError(
                    f"Window {name!r} takes the last {last_k} rows, which needs "
                    "time='<timestamp column>' to order them"
                )
            if last_k <= 0:
                raise ValueError(f"Window {name!r} needs last_k > 0")
            created = Window(node("window", kind="last_k", k=int(last_k), name=name))
        self._windows[name] = created
        return created

    def derive(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a shared intermediate: reusable, but not an output column."""
        name = fn.__name__
        self._check_name(name, "derive")
        self._derives[name] = FeatureDef(
            name=name, fn=fn, kind="derive", doc=fn.__doc__
        )
        return fn

    def feature(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        default: float = 0.0,
    ) -> Any:
        """Register an output feature. Usable bare or with ``default=``."""

        def register(target: Callable[..., Any]) -> Callable[..., Any]:
            name = target.__name__
            self._check_name(name, "feature")
            self._features.append(
                FeatureDef(
                    name=name,
                    fn=target,
                    kind="feature",
                    default=default,
                    doc=target.__doc__,
                )
            )
            return target

        return register if fn is None else register(fn)

    def opaque(
        self,
        *,
        requires: Iterable[str],
        default: float = 0.0,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a feature the planner cannot see inside.

        The escape hatch for logic the algebra does not express. It receives
        one entity's rows as a DataFrame and returns a number. It is correct
        but unplanned, so it runs per entity -- the cost report names it.
        """
        required = tuple(requires)
        if not required:
            raise ValueError(
                "An opaque feature must declare the columns it reads, so the "
                "planner can still order it and share its inputs"
            )

        def register(target: Callable[..., Any]) -> Callable[..., Any]:
            name = target.__name__
            self._check_name(name, "opaque feature")
            self._features.append(
                FeatureDef(
                    name=name,
                    fn=target,
                    kind="opaque",
                    default=default,
                    doc=target.__doc__,
                    requires=required,
                )
            )
            return target

        return register

    # -- planning ----------------------------------------------------------
    def plan(
        self,
        source: Any,
        *,
        table: str | None = None,
        verify: bool = True,
        verify_entities: int = DEFAULT_SAMPLE_ENTITIES,
        rtol: float = DEFAULT_RTOL,
        random_seed: int = 42,
    ) -> FeaturePlan:
        """Trace every feature against a source and return an executable plan.

        Tracing happens here, not at decoration time, because a feature may
        legitimately branch on the schema and the schema is not known until a
        frame is in hand.

        Verification is on by default and raises on any disagreement. A
        planner that silently returns different numbers from the code it
        replaced is worse than no planner, so the default is to find out at
        build time rather than in a model's metrics three weeks later. Pass
        ``verify=False`` to skip it once a plan is known-good and being
        rebuilt in a hot path.
        """
        frame = _resolve_frame(source, table=table)
        schema = tuple(str(column) for column in frame.columns)
        plan = FeaturePlan(
            features=self.trace(schema),
            entity=self._entity,
            time=self._time,
            schema=schema,
            reference_time=self._reference_time,
            target=self._target,
        )
        if not verify:
            return plan
        report = plan.verify(frame, sample=verify_entities, rtol=rtol, seed=random_seed)
        report.raise_for_status()
        return replace(plan, verification=report)

    def trace(self, schema: Sequence[str]) -> tuple[TracedFeature, ...]:
        """Trace every feature against a schema. Mostly useful in tests."""
        self._require_entity(schema)
        previous = self._schema
        self._schema = tuple(schema)
        try:
            return trace_features(
                features=self._features,
                derives=self._derives,
                windows=self._windows,
                schema=self._schema,
            )
        finally:
            self._schema = previous

    # -- internals ---------------------------------------------------------
    def _require_entity(self, schema: Sequence[str]) -> None:
        if self._reference_time is not None and self._reference_time not in schema:
            raise TracingError(
                f"Reference-time column {self._reference_time!r} is not in the "
                f"frame. Columns: {', '.join(sorted(schema)) or '(none)'}"
            )
        if self._entity not in schema:
            raise TracingError(
                f"Entity column {self._entity!r} is not in the frame. "
                f"Columns: {', '.join(sorted(schema)) or '(none)'}"
            )
        if self._time is not None and self._time not in schema:
            raise TracingError(
                f"Time column {self._time!r} is not in the frame. "
                f"Columns: {', '.join(sorted(schema)) or '(none)'}"
            )

    def _check_name(self, name: str, kind: str) -> None:
        if kind != "window" and name in _RESERVED:
            raise ValueError(f"{name!r} is reserved; choose another name")
        if name in self._windows:
            raise ValueError(f"A window named {name!r} is already declared")
        if name in self._derives:
            raise ValueError(f"A @derive named {name!r} is already declared")
        if any(item.name == name for item in self._features):
            raise ValueError(f"A feature named {name!r} is already declared")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FeatureSet(entity={self._entity!r}, time={self._time!r}, "
            f"features={len(self._features)}, windows={len(self._windows)})"
        )


def _resolve_frame(source: Any, *, table: str | None) -> Any:
    """Accept a DataFrame, a mapping of tables, or a Dataset session."""
    import pandas as pd

    if isinstance(source, pd.DataFrame):
        if table is not None:
            raise ValueError("table= is meaningless when passing a DataFrame")
        return source

    tables: Any
    if hasattr(source, "tables"):
        tables = source.tables
    elif isinstance(source, dict):
        tables = source
    else:
        raise TypeError(
            f"Cannot build a feature plan from {type(source).__name__}; pass a "
            "pandas DataFrame, a mapping of named frames, or a prism Dataset"
        )

    names = list(tables)
    if table is not None:
        if table not in tables:
            available = ", ".join(sorted(names)) or "(none)"
            raise KeyError(f"No table named {table!r}. Available: {available}")
        return tables[table]
    if len(names) == 1:
        return tables[names[0]]
    available = ", ".join(sorted(names))
    raise ValueError(
        f"The source has {len(names)} tables ({available}); name one with table="
    )
