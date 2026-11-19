"""Helpers shared by the SQL sources."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from decimal import Decimal
from typing import Any, TypeVar
from uuid import UUID

import pandas as pd

from prism_eda.sources.base import ColumnInfo, SourceError, SourceTimeout

_T = TypeVar("_T")

#: Strings longer than this are cut in row samples. Statistics about length come
#: from the full table; the sample only needs enough of the text to recognise it.
SAMPLE_TEXT_LIMIT = 400

#: Above this many rows a Postgres table is profiled with planner statistics
#: and samples where an exact answer would need a sort of the whole table.
LARGE_TABLE_ROWS = 5_000_000

GRAINS = ("hour", "day", "week", "month", "year")


def chunk_values(values: Sequence[str], max_bytes: int) -> Iterator[list[str]]:
    """Split a literal list so each piece fits in one SQL statement."""
    chunk: list[str] = []
    size = 0
    for value in values:
        cost = len(value.encode("utf-8")) + 4
        if chunk and size + cost > max_bytes:
            yield chunk
            chunk, size = [], 0
        chunk.append(value)
        size += cost
    if chunk:
        yield chunk


def with_retries(
    work: Callable[[], _T],
    *,
    retries: int,
    is_transient: Callable[[BaseException], bool],
    is_timeout: Callable[[BaseException], bool],
    label: str,
) -> _T:
    """Run one query, retrying dropped connections with a short back-off.

    A timeout is never retried: the query already had its full allowance, and
    running it again would double the load on the database for the same result.
    """
    attempt = 0
    while True:
        try:
            return work()
        except SourceError:
            raise
        except Exception as error:  # noqa: BLE001 - classified below
            if is_timeout(error):
                raise SourceTimeout(f"{label}: query exceeded its limit") from error
            if attempt < retries and is_transient(error):
                attempt += 1
                time.sleep(min(2.0**attempt, 10.0))
                continue
            raise SourceError(f"{label}: {_brief(error)}") from error


def _brief(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return text[:400] if text else type(error).__name__


def tidy_frame(frame: pd.DataFrame, columns: Sequence[ColumnInfo]) -> pd.DataFrame:
    """Give a fetched sample plain pandas types, whatever the driver returned."""
    for column in columns:
        if column.name not in frame.columns:
            continue
        series = frame[column.name]
        if column.kind in {"integer", "float"}:
            numeric = pd.to_numeric(series.map(_number), errors="coerce")
            if column.kind == "integer":
                # numeric(18,0) arrives as Decimal; keep it whole, not 5298.0.
                try:
                    numeric = numeric.astype("Int64")
                except (TypeError, ValueError):
                    pass
            frame[column.name] = numeric
        elif column.kind in {"datetime", "date"}:
            converted = pd.to_datetime(series, errors="coerce", utc=True)
            frame[column.name] = converted.dt.tz_localize(None)
        elif column.kind == "boolean":
            frame[column.name] = series.astype("boolean")
        else:
            frame[column.name] = series.map(_text).astype(object)
    return frame


def _number(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def _text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
