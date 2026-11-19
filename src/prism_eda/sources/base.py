"""The contract every data source implements for deep exploration.

Exploration never asks a source for "the table". It asks narrow questions —
how many rows, which values are in this column, which of these values exist
over there — so a 500-million-row ClickHouse table and a small DataFrame can be
explored by the same code. A source answers each question the cheapest way it
can: SQL pushed down to the database, or pandas on a frame already in memory.

Three rules hold for every implementation:

1. Read-only. A source never writes, creates, or alters anything.
2. Bounded. Every answer has a size cap, and every database query has a time
   limit, so one pathological table cannot take the run (or the database) down.
3. Comparable. Values are compared across columns, tables, and databases in
   one canonical text form (:func:`normalise_value`), and sampled by one hash
   (:func:`value_hash`) that every backend computes identically.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from prism_eda.exceptions import PrismEDAError

#: Column kinds a source reports. ``other`` covers arrays, JSON, binary and
#: anything else exploration profiles only by null count.
KINDS = ("integer", "float", "string", "datetime", "date", "boolean", "other")


class SourceError(PrismEDAError):
    """A source could not answer a question (connection, permission, SQL)."""


class SourceTimeout(SourceError):
    """A query exceeded its time or resource limit."""


@dataclass(frozen=True, slots=True)
class SourceLimits:
    """Per-query safety limits applied by database sources.

    The defaults are deliberately conservative: exploration runs against
    production databases that are serving other work.
    """

    #: Seconds one query may run before the database cancels it.
    query_timeout: int = 600
    #: Seconds to wait for a connection.
    connect_timeout: int = 15
    #: Threads one ClickHouse query may use (ignored by Postgres).
    max_threads: int = 4
    #: Bytes one ClickHouse query may hold in memory (ignored by Postgres).
    max_memory_bytes: int = 4 * 1024**3
    #: Transient failures (dropped connection, timeout on connect) are retried
    #: this many times before the question is reported as failed.
    retries: int = 2
    #: Largest SQL text sent in one statement; probe lists are chunked to fit.
    max_sql_bytes: int = 180_000


@dataclass(frozen=True, slots=True)
class TableInfo:
    """One table as a source lists it, before anything has been read."""

    source: str
    schema: str | None
    name: str
    kind: str = "table"
    approx_rows: int | None = None
    bytes: int | None = None
    engine: str | None = None
    comment: str | None = None

    @property
    def qualified(self) -> str:
        """``schema.table`` (or just the table name when there is no schema)."""
        return f"{self.schema}.{self.name}" if self.schema else self.name

    @property
    def key(self) -> str:
        """Stable identifier, unique across sources."""
        return f"{self.source}:{self.qualified}"


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    name: str
    native_type: str
    kind: str
    nullable: bool = True


@dataclass(frozen=True, slots=True)
class ColumnStats:
    """Full-table statistics for one column.

    ``distinct_exact`` is False when the database estimated the distinct count
    (HyperLogLog-style); reports must say so rather than present it as exact.
    """

    name: str
    rows: int
    non_null: int
    distinct: int | None = None
    distinct_exact: bool = True
    blank: int = 0
    minimum: Any = None
    maximum: Any = None
    mean: float | None = None
    std: float | None = None
    quantiles: Mapping[str, float] = field(default_factory=dict)
    zeros: int | None = None
    negatives: int | None = None
    min_length: int | None = None
    max_length: int | None = None
    mean_length: float | None = None


# --------------------------------------------------------------------------
# Canonical values and the shared hash
# --------------------------------------------------------------------------

_ASCII_LOWER = {code: code + 32 for code in range(ord("A"), ord("Z") + 1)}


def normalise_value(value: Any) -> str | None:
    """Canonical text form used to compare values across columns and sources.

    Identifiers are routinely stored differently in different places: an
    integer here, a zero-padded string there, upper case in one export and
    lower in another. Comparing raw values would report those as unrelated.
    The canonical form trims surrounding spaces, lower-cases ASCII letters,
    and removes leading zeros. A value with nothing left after that (blank, or
    made only of zeros) has no canonical form and never takes part in a match:
    zero is far more often a placeholder than an identifier.

    Every SQL dialect reproduces exactly this function, so it is kept simple
    enough to express in SQL. Lower-casing is ASCII-only because that is the
    one behaviour ClickHouse, Postgres and Python can all be made to share.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        if value.is_integer():
            value = int(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    text = str(value).strip(" ").translate(_ASCII_LOWER).lstrip("0")
    return text or None


def value_hash(normalised: str) -> int:
    """Unsigned 64-bit hash of a canonical value.

    It is the first eight bytes of MD5, big-endian: ClickHouse's ``halfMD5``,
    and reproducible in Postgres from ``md5()``. Only the low bits are ever
    used (as a power-of-two sampling mask), which keeps signed and unsigned
    interpretations in agreement.
    """
    digest = hashlib.md5(normalised.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:8], "big")


def in_hash_sample(normalised: str, rate: int) -> bool:
    """Whether a canonical value belongs to the 1-in-``rate`` hash sample."""
    return (value_hash(normalised) & (rate - 1)) == 0


def sampling_rate(distinct: int | None, target: int) -> int:
    """Smallest power of two that keeps about ``target`` distinct values.

    Powers of two nest: the 1-in-8 sample is a subset of the 1-in-4 sample, so
    samples taken at different rates can still be compared.
    """
    rate = 1
    if not distinct or target <= 0:
        return rate
    while distinct / rate > target and rate < 2**40:
        rate *= 2
    return rate


def check_rate(rate: int) -> int:
    if rate < 1 or rate & (rate - 1):
        raise ValueError("hash sampling rate must be a power of two")
    return rate


# --------------------------------------------------------------------------
# Table selection
# --------------------------------------------------------------------------


def _patterns(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def matches(table: TableInfo, patterns: Sequence[str]) -> bool:
    """Whether a table matches any pattern.

    A pattern is a case-insensitive glob tested against the bare table name,
    ``schema.table``, and ``source:schema.table`` — so ``txn_*``,
    ``public.accounts`` and ``warehouse:*`` all work.
    """
    names = {table.name.lower(), table.qualified.lower(), table.key.lower()}
    for pattern in patterns:
        lowered = pattern.lower()
        if any(fnmatch.fnmatchcase(name, lowered) for name in names):
            return True
    return False


def select_tables(
    tables: Iterable[TableInfo],
    *,
    include: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
) -> list[TableInfo]:
    """Apply include then exclude patterns; no include means everything."""
    included = _patterns(include)
    excluded = _patterns(exclude)
    chosen = []
    for table in tables:
        if included and not matches(table, included):
            continue
        if excluded and matches(table, excluded):
            continue
        chosen.append(table)
    return chosen


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_name(name: str) -> str:
    """A short ASCII label for a source, used in table keys."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return cleaned or "source"


# --------------------------------------------------------------------------
# The source contract
# --------------------------------------------------------------------------


class Source(ABC):
    """A place tables live: a database, or a set of DataFrames."""

    #: Short label that prefixes table keys (``warehouse:public.accounts``).
    name: str
    #: ``clickhouse``, ``postgres``, ``dataframe``…
    kind: str

    # -- inventory ---------------------------------------------------------

    @abstractmethod
    def tables(self) -> list[TableInfo]:
        """Every user table the source exposes (system objects excluded)."""

    @abstractmethod
    def columns(self, table: TableInfo) -> list[ColumnInfo]:
        """The table's columns, in order."""

    @abstractmethod
    def row_count(self, table: TableInfo) -> int:
        """Exact number of rows."""

    # -- column profile ----------------------------------------------------

    @abstractmethod
    def column_stats(
        self, table: TableInfo, columns: Sequence[ColumnInfo]
    ) -> dict[str, ColumnStats]:
        """Full-table statistics for the given columns."""

    @abstractmethod
    def top_values(
        self, table: TableInfo, columns: Sequence[ColumnInfo], limit: int
    ) -> dict[str, list[tuple[Any, int]]]:
        """Most frequent non-null values per column, with their row counts."""

    @abstractmethod
    def time_histogram(
        self, table: TableInfo, column: ColumnInfo, grain: str
    ) -> list[tuple[str, int]]:
        """Row counts per period (``grain`` is hour, day, week, month, year)."""

    # -- rows --------------------------------------------------------------

    @abstractmethod
    def sample(
        self,
        table: TableInfo,
        columns: Sequence[ColumnInfo],
        rows: int,
        *,
        total_rows: int,
        seed: int,
        hash_columns: Sequence[ColumnInfo] = (),
        where: tuple[ColumnInfo, Sequence[Any]] | None = None,
    ) -> pd.DataFrame:
        """About ``rows`` rows chosen deterministically.

        ``hash_columns`` are the columns a database may hash to choose rows
        reproducibly; ``where`` restricts the sample to rows whose column
        holds one of the given values.
        """

    # -- values ------------------------------------------------------------

    @abstractmethod
    def distinct_sample(
        self,
        table: TableInfo,
        requests: Mapping[str, tuple[ColumnInfo, int]],
        limit: int,
    ) -> dict[str, list[str]]:
        """Distinct raw values whose canonical form falls in the hash sample.

        ``requests`` maps a column name to ``(column, rate)``. Blank and null
        values are never returned.
        """

    @abstractmethod
    def probe(
        self,
        table: TableInfo,
        requests: Mapping[str, tuple[ColumnInfo, Sequence[str]]],
    ) -> dict[str, set[str]]:
        """Which canonical values exist in each column.

        ``requests`` maps a column name to ``(column, canonical values)``; the
        answer holds the subset of those canonical values present in the
        column. This is the primitive behind every relationship: it is exact
        for the values asked about, however large the table.
        """

    # -- links between two columns of one table ----------------------------

    @abstractmethod
    def pair_profile(
        self, table: TableInfo, left: ColumnInfo, right: ColumnInfo
    ) -> dict[str, Any]:
        """Rows with both values present, rows where they are equal, and the
        (possibly estimated) number of distinct pairs."""

    @abstractmethod
    def fanout_sample(
        self,
        table: TableInfo,
        key: ColumnInfo,
        others: Sequence[ColumnInfo],
        rate: int,
        limit: int,
    ) -> pd.DataFrame:
        """Per hash-sampled ``key`` value: its row count and how many distinct
        values of each other column it appears with.

        Columns: ``key`` (canonical), ``rows``, then one per other column.
        Sampling whole keys (not rows) keeps each key's counts exact.
        """

    @abstractmethod
    def neighbours(
        self,
        table: TableInfo,
        left: ColumnInfo,
        right: ColumnInfo,
        values: Sequence[str],
        limit: int,
    ) -> list[tuple[str, str]]:
        """Distinct canonical ``(left, right)`` pairs whose left value is one
        of ``values``; at most ``limit`` pairs."""

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release anything held open. Sources hold nothing by default."""

    def describe(self) -> dict[str, Any]:
        """Non-secret provenance for the report (never credentials)."""
        return {"name": self.name, "kind": self.kind}
