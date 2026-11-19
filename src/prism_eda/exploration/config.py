"""Knobs for deep exploration, with defaults sized for production databases."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ColumnHint:
    """Names a column the caller knows the meaning of.

    ``table`` is matched the way include patterns are (bare name,
    ``schema.table``, or ``source:schema.table``); leave it out to match the
    column name in every table that has it.
    """

    column: str
    table: str | None = None
    #: For a label hint: the values that mean "positive". When omitted, the
    #: rarer of two values is taken as positive.
    positive: tuple[Any, ...] = ()

    @classmethod
    def parse(cls, value: ColumnHint | str | dict[str, Any]) -> ColumnHint:
        if isinstance(value, ColumnHint):
            return value
        if isinstance(value, dict):
            positive = value.get("positive", ())
            if not isinstance(positive, (list, tuple)):
                positive = (positive,)
            return cls(
                column=str(value["column"]),
                table=value.get("table"),
                positive=tuple(positive),
            )
        text = str(value)
        if "." in text:
            table, column = text.rsplit(".", 1)
            return cls(column=column, table=table)
        return cls(column=text)


@dataclass(frozen=True, slots=True)
class Hints:
    """Optional knowledge about the data. Everything works without it."""

    #: A column marking the rows of interest (confirmed fraud, churned…).
    label: ColumnHint | None = None
    #: The identifier of the thing being studied (account, customer…).
    entity: ColumnHint | None = None
    #: The event time column.
    timestamp: ColumnHint | None = None
    #: The monetary (or otherwise primary) measure.
    amount: ColumnHint | None = None

    @classmethod
    def parse(cls, value: Hints | dict[str, Any] | None) -> Hints:
        if value is None:
            return cls()
        if isinstance(value, Hints):
            return value
        unknown = set(value) - {"label", "entity", "timestamp", "amount"}
        if unknown:
            raise ValueError(f"Unknown hint(s): {', '.join(sorted(unknown))}")
        return cls(
            **{
                key: ColumnHint.parse(item)
                for key, item in value.items()
                if item is not None
            }
        )

    def __bool__(self) -> bool:
        return any((self.label, self.entity, self.timestamp, self.amount))


@dataclass(frozen=True, slots=True)
class ExploreConfig:
    """How much work exploration does per table, column, and relationship."""

    #: Rows pulled per table for distribution, correlation and quality checks.
    sample_rows: int = 50_000
    #: Columns per table that get full statistics; wider tables are truncated
    #: (and the report says so).
    max_columns: int = 200
    #: Columns with at most this many distinct values get exact top values;
    #: above it top values come from the row sample.
    exact_top_values_distinct: int = 5_000
    top_values: int = 10
    #: Time columns charted per table.
    timelines_per_table: int = 3

    # -- relationship discovery -------------------------------------------
    relationships: bool = True
    #: A column needs at least this many distinct values to be considered an
    #: identifier. Lower values make every status flag "related" to every other.
    min_key_distinct: int = 30
    #: Identifier-like columns considered per table (widest value range first).
    max_key_columns: int = 16
    #: Text longer than this on average is prose, not an identifier.
    max_key_length: float = 64.0
    #: Distinct values kept per identifier column for coverage estimates.
    value_sample: int = 2_048
    #: Values per column used in the cheap first screening pass.
    screen_values: int = 256
    #: Rows of the row sample used for row-weighted coverage.
    row_coverage_rows: int = 4_000
    #: A pair is reported once this many sampled values are found on the other
    #: side *and* they make up at least ``min_coverage`` of the sample.
    min_matches: int = 5
    min_coverage: float = 0.005
    #: Short all-digit values (serials, counts) overlap by coincidence. Below
    #: this length a pair needs the stricter evidence described in the guide.
    weak_value_length: int = 7

    # -- links between records of one kind --------------------------------
    network: bool = True
    #: Link column pairs analysed in full (largest tables first).
    max_links: int = 12
    #: Entities sampled for degree and fan-out distributions.
    degree_entities: int = 20_000
    #: Seed entities walked outward to measure 2- and 3-hop reach.
    hop_seeds: int = 300
    #: Distinct entities followed per hop before the walk is truncated.
    hop_frontier: int = 20_000
    #: Pairs fetched per hop.
    hop_pairs: int = 400_000
    max_hops: int = 3

    # -- label-aware analysis ---------------------------------------------
    #: Labelled rows pulled for the with/without comparison.
    label_rows: int = 20_000

    seed: int = 42

    def __post_init__(self) -> None:
        if self.sample_rows < 100:
            raise ValueError("sample_rows must be at least 100")
        if not 2 <= self.max_hops <= 4:
            raise ValueError("max_hops must be between 2 and 4")


@dataclass(frozen=True, slots=True)
class Progress:
    """One progress report from a running exploration."""

    stage: str
    message: str
    done: int = 0
    total: int = 0
    fraction: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


#: Called with every progress report. Unlike event callbacks, an exception
#: raised here is *not* swallowed: raising is how a caller cancels a run.
ProgressCallback = Callable[[Progress], None]


class CheckpointStore(Protocol):
    """Where finished pieces of work are kept so a run can be resumed.

    Keys are short ASCII strings; values are opaque bytes. A store that loses
    a value only costs time: the piece is computed again.
    """

    def load(self, key: str) -> bytes | None: ...

    def save(self, key: str, payload: bytes) -> None: ...


class MemoryCheckpoints:
    """A checkpoint store that lives as long as the object does."""

    def __init__(self) -> None:
        self._items: dict[str, bytes] = {}

    def load(self, key: str) -> bytes | None:
        return self._items.get(key)

    def save(self, key: str, payload: bytes) -> None:
        self._items[key] = payload

    def keys(self) -> Sequence[str]:
        return tuple(self._items)
