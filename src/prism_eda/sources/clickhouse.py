"""ClickHouse as an exploration source.

Every question is one bounded, read-only query on a connection opened for that
query and closed straight after. Nothing is held open between questions, so a
long exploration cannot be broken by an idle connection being dropped.

Requires the ``clickhouse`` extra (``pip install 'prism-eda[clickhouse]'``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from prism_eda.sources._sql import (
    SAMPLE_TEXT_LIMIT,
    chunk_values,
    tidy_frame,
    to_float,
    to_int,
    with_retries,
)
from prism_eda.sources.base import (
    ColumnInfo,
    ColumnStats,
    Source,
    SourceError,
    SourceLimits,
    TableInfo,
    check_rate,
    normalise_value,
    safe_name,
)

_SYSTEM_DATABASES = ("system", "INFORMATION_SCHEMA", "information_schema")
#: Engines that hold no rows of their own or cannot be scanned safely.
_SKIPPED_ENGINES = (
    "Dictionary",
    "Kafka",
    "RabbitMQ",
    "NATS",
    "Null",
    "Set",
    "Join",
    "MaterializedView",
)
_VIEW_ENGINES = ("View", "LiveView", "WindowView")
_GRAIN_SQL = {
    "hour": "toStartOfHour({c})",
    "day": "toStartOfDay({c})",
    "week": "toStartOfWeek({c}, 1)",
    "month": "toStartOfMonth({c})",
    "year": "toStartOfYear({c})",
}
_SETTINGS_REFUSED = re.compile(r"readonly|Cannot modify", re.IGNORECASE)
_STATS_BATCH = 20


def clickhouse_kind(native_type: str) -> str:
    inner = native_type.strip()
    while True:
        match = re.match(r"^(Nullable|LowCardinality)\((.*)\)$", inner)
        if not match:
            break
        inner = match.group(2).strip()
    if re.match(r"^U?Int\d+$", inner):
        return "integer"
    if inner.startswith(("Float", "Decimal")):
        return "float"
    if inner == "Bool":
        return "boolean"
    if inner.startswith("DateTime"):
        return "datetime"
    if inner in {"Date", "Date32"}:
        return "date"
    if inner.startswith(("String", "FixedString", "Enum", "UUID", "IPv")):
        return "string"
    return "other"


def _ident(name: str) -> str:
    return "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"


def _literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


class ClickHouseSource(Source):
    """Explore the tables of a ClickHouse server.

    Pass connection details, or ``client_factory`` — a callable returning a
    ``clickhouse_connect`` client — when the application already knows how to
    connect. ``databases`` limits which databases are listed: the connection's
    own database by default, a list of names, or ``"*"`` for every database
    the user can see.
    """

    kind = "clickhouse"

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        secure: bool = False,
        databases: Sequence[str] | str | None = None,
        include_views: bool = False,
        name: str = "clickhouse",
        limits: SourceLimits | None = None,
        client_factory: Callable[[], Any] | None = None,
        connect_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = safe_name(name)
        self.limits = limits or SourceLimits()
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._database = database
        self._secure = secure
        self._databases = databases
        self._include_views = include_views
        self._factory = client_factory
        self._options = dict(connect_options or {})
        self._send_settings = True

    # -- connection --------------------------------------------------------

    def _client(self) -> Any:
        if self._factory is not None:
            return self._factory()
        try:
            import clickhouse_connect
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise SourceError(
                "ClickHouse support needs the 'clickhouse' extra: "
                "pip install 'prism-eda[clickhouse]'"
            ) from error
        options: dict[str, Any] = {
            "connect_timeout": self.limits.connect_timeout,
            "send_receive_timeout": self.limits.query_timeout + 30,
        }
        options.update(self._options)
        for key, value in (
            ("host", self._host),
            ("port", self._port),
            ("username", self._username),
            ("password", self._password),
            ("database", self._database),
        ):
            if value is not None:
                options[key] = value
        if self._secure:
            options["secure"] = True
        return clickhouse_connect.get_client(**options)

    def _settings(self) -> dict[str, Any]:
        return {
            "max_execution_time": self.limits.query_timeout,
            "max_threads": self.limits.max_threads,
            "max_memory_usage": self.limits.max_memory_bytes,
        }

    def _run(self, sql: str, label: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        def once() -> tuple[list[str], list[tuple[Any, ...]]]:
            client = self._client()
            try:
                try:
                    settings = self._settings() if self._send_settings else {}
                    result = client.query(sql, settings=settings)
                except Exception as error:  # noqa: BLE001
                    # A read-only profile may forbid changing settings. The
                    # query is still safe to run under the server's own limits.
                    if self._send_settings and _SETTINGS_REFUSED.search(str(error)):
                        self._send_settings = False
                        result = client.query(sql)
                    else:
                        raise
                return list(result.column_names), list(result.result_rows)
            finally:
                try:
                    client.close()
                except Exception:  # noqa: BLE001 - closing is best effort
                    pass

        return with_retries(
            once,
            retries=self.limits.retries,
            is_transient=_is_transient,
            is_timeout=_is_timeout,
            label=label,
        )

    # -- SQL fragments -----------------------------------------------------

    def _table(self, table: TableInfo) -> str:
        if table.schema:
            return f"{_ident(table.schema)}.{_ident(table.name)}"
        return _ident(table.name)

    def _text(self, column: ColumnInfo) -> str:
        inner = column.native_type
        if "FixedString" in inner:
            return f"toStringCutToZero({_ident(column.name)})"
        return f"toString({_ident(column.name)})"

    def _norm(self, column: ColumnInfo) -> str:
        text = self._text(column)
        return f"nullIf(trim(LEADING '0' FROM lower(trim(BOTH ' ' FROM {text}))), '')"

    @staticmethod
    def _in_sample(norm: str, rate: int) -> str:
        if rate <= 1:
            return f"{norm} IS NOT NULL"
        return f"bitAnd(halfMD5(assumeNotNull({norm})), {rate - 1}) = 0"

    # -- inventory ---------------------------------------------------------

    def tables(self) -> list[TableInfo]:
        where = ["is_temporary = 0", "name NOT LIKE '.inner%'"]
        if self._databases == "*":
            names = ", ".join(_literal(name) for name in _SYSTEM_DATABASES)
            where.append(f"database NOT IN ({names})")
        elif self._databases:
            wanted = (
                [self._databases]
                if isinstance(self._databases, str)
                else list(self._databases)
            )
            names = ", ".join(_literal(name) for name in wanted)
            where.append(f"database IN ({names})")
        else:
            where.append("database = currentDatabase()")
        skipped = list(_SKIPPED_ENGINES)
        if not self._include_views:
            skipped.extend(_VIEW_ENGINES)
        engines = ", ".join(_literal(engine) for engine in skipped)
        where.append(f"engine NOT IN ({engines})")
        sql = (
            "SELECT database, name, engine, total_rows, total_bytes, comment "
            f"FROM system.tables WHERE {' AND '.join(where)} "
            "ORDER BY database, name"
        )
        _, rows = self._run(sql, "list tables")
        return [
            TableInfo(
                source=self.name,
                schema=str(database),
                name=str(name),
                kind="view" if engine in _VIEW_ENGINES else "table",
                approx_rows=to_int(total_rows),
                bytes=to_int(total_bytes),
                engine=str(engine),
                comment=str(comment) if comment else None,
            )
            for database, name, engine, total_rows, total_bytes, comment in rows
        ]

    def columns(self, table: TableInfo) -> list[ColumnInfo]:
        sql = (
            "SELECT name, type FROM system.columns "
            f"WHERE database = {_literal(table.schema or '')} "
            f"AND table = {_literal(table.name)} ORDER BY position"
        )
        _, rows = self._run(sql, f"columns of {table.qualified}")
        return [
            ColumnInfo(
                name=str(name),
                native_type=str(native),
                kind=clickhouse_kind(str(native)),
                nullable="Nullable(" in str(native),
            )
            for name, native in rows
        ]

    def row_count(self, table: TableInfo) -> int:
        _, rows = self._run(
            f"SELECT count() FROM {self._table(table)}",
            f"row count of {table.qualified}",
        )
        return int(rows[0][0])

    # -- column profile ----------------------------------------------------

    def column_stats(
        self, table: TableInfo, columns: Sequence[ColumnInfo]
    ) -> dict[str, ColumnStats]:
        stats: dict[str, ColumnStats] = {}
        for start in range(0, len(columns), _STATS_BATCH):
            stats.update(
                self._stats_batch(table, columns[start : start + _STATS_BATCH])
            )
        return stats

    def _stats_batch(
        self, table: TableInfo, columns: Sequence[ColumnInfo]
    ) -> dict[str, ColumnStats]:
        parts = ["count() AS rows_total"]
        for index, column in enumerate(columns):
            c = _ident(column.name)
            p = f"c{index}"
            parts.append(f"count({c}) AS {p}_nn")
            if column.kind != "other":
                parts.append(f"uniq({c}) AS {p}_d")
            if column.kind in {"integer", "float"}:
                value = f"toFloat64({c})"
                ok = f"isFinite({value})" if column.kind == "float" else "1"
                parts.extend(
                    [
                        f"minIf({value}, {ok}) AS {p}_min",
                        f"maxIf({value}, {ok}) AS {p}_max",
                        f"avgIf({value}, {ok}) AS {p}_avg",
                        f"stddevPopIf({value}, {ok}) AS {p}_std",
                        "quantilesTDigestIf(0.05, 0.25, 0.5, 0.75, 0.95, 0.99)"
                        f"({value}, {ok}) AS {p}_q",
                        f"countIf({value} = 0) AS {p}_zero",
                        f"countIf({value} < 0) AS {p}_neg",
                    ]
                )
            elif column.kind in {"datetime", "date"}:
                parts.extend(
                    [f"toString(min({c})) AS {p}_min", f"toString(max({c})) AS {p}_max"]
                )
            elif column.kind == "string":
                text = self._text(column)
                parts.extend(
                    [
                        f"countIf(trim(BOTH ' ' FROM {text}) = '') AS {p}_blank",
                        f"min(lengthUTF8({text})) AS {p}_lmin",
                        f"max(lengthUTF8({text})) AS {p}_lmax",
                        f"avg(lengthUTF8({text})) AS {p}_lavg",
                    ]
                )
        sql = f"SELECT {', '.join(parts)} FROM {self._table(table)}"
        names, rows = self._run(sql, f"statistics of {table.qualified}")
        row = dict(zip(names, rows[0], strict=True))
        total = int(row["rows_total"])
        answer: dict[str, ColumnStats] = {}
        for index, column in enumerate(columns):
            p = f"c{index}"
            non_null = int(row[f"{p}_nn"])
            distinct = to_int(row.get(f"{p}_d"))
            extra: dict[str, Any] = {}
            if column.kind in {"integer", "float"} and non_null:
                quantiles = row.get(f"{p}_q") or []
                labels = ("p5", "q1", "median", "q3", "p95", "p99")
                extra = {
                    "minimum": to_float(row[f"{p}_min"]),
                    "maximum": to_float(row[f"{p}_max"]),
                    "mean": to_float(row[f"{p}_avg"]),
                    "std": to_float(row[f"{p}_std"]),
                    "quantiles": {
                        label: number
                        for label, raw in zip(labels, quantiles, strict=False)
                        if (number := to_float(raw)) is not None
                    },
                    "zeros": to_int(row[f"{p}_zero"]),
                    "negatives": to_int(row[f"{p}_neg"]),
                }
            elif column.kind in {"datetime", "date"} and non_null:
                extra = {
                    "minimum": _timestamp(row[f"{p}_min"]),
                    "maximum": _timestamp(row[f"{p}_max"]),
                }
            elif column.kind == "string" and non_null:
                extra = {
                    "blank": int(row[f"{p}_blank"]),
                    "min_length": to_int(row[f"{p}_lmin"]),
                    "max_length": to_int(row[f"{p}_lmax"]),
                    "mean_length": to_float(row[f"{p}_lavg"]),
                }
            answer[column.name] = ColumnStats(
                name=column.name,
                rows=total,
                non_null=non_null,
                distinct=distinct,
                # ``uniq`` keeps up to 65,536 hashes before it starts sampling,
                # so smaller answers are exact.
                distinct_exact=distinct is not None and distinct < 60_000,
                **extra,
            )
        return answer

    def top_values(
        self, table: TableInfo, columns: Sequence[ColumnInfo], limit: int
    ) -> dict[str, list[tuple[Any, int]]]:
        answer: dict[str, list[tuple[Any, int]]] = {}
        for column in columns:
            c = _ident(column.name)
            value = self._text(column) if column.kind == "string" else c
            sql = (
                f"SELECT {value} AS v, count() AS n FROM {self._table(table)} "
                f"WHERE {c} IS NOT NULL GROUP BY v ORDER BY n DESC, v "
                f"LIMIT {int(limit)}"
            )
            _, rows = self._run(sql, f"top values of {table.qualified}.{column.name}")
            answer[column.name] = [(value, int(count)) for value, count in rows]
        return answer

    def time_histogram(
        self, table: TableInfo, column: ColumnInfo, grain: str
    ) -> list[tuple[str, int]]:
        if column.kind == "date" and grain == "hour":
            grain = "day"
        bucket = _GRAIN_SQL[grain].format(c=_ident(column.name))
        sql = (
            f"SELECT toString({bucket}) AS b, count() AS n "
            f"FROM {self._table(table)} WHERE {_ident(column.name)} IS NOT NULL "
            "GROUP BY b ORDER BY b LIMIT 5000"
        )
        _, rows = self._run(sql, f"timeline of {table.qualified}.{column.name}")
        answer = []
        for bucket_text, count in rows:
            stamp = _timestamp(bucket_text)
            if stamp is not None:
                answer.append((stamp.isoformat(), int(count)))
        return answer

    # -- rows --------------------------------------------------------------

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
        selected = []
        for column in columns:
            c = _ident(column.name)
            if column.kind == "string":
                selected.append(
                    f"leftUTF8({self._text(column)}, {SAMPLE_TEXT_LIMIT}) AS {c}"
                )
            else:
                selected.append(c)
        conditions = []
        if where is not None:
            column, values = where
            wanted = sorted(
                {norm for value in values if (norm := normalise_value(value))}
            )
            if not wanted:
                return pd.DataFrame(columns=[column.name for column in columns])
            literals = ", ".join(_literal(value) for value in wanted)
            conditions.append(f"{self._norm(column)} IN ({literals})")
        if total_rows > rows and hash_columns:
            modulus = 1_000_000
            threshold = max(1, round(rows / total_rows * modulus))
            hashed = ", ".join(_ident(column.name) for column in hash_columns)
            conditions.append(
                f"cityHash64({hashed}, {int(seed)}) % {modulus} < {threshold}"
            )
        sql = f"SELECT {', '.join(selected)} FROM {self._table(table)}"
        if conditions:
            sql += f" WHERE {' AND '.join(conditions)}"
        sql += f" LIMIT {int(rows * 1.5) + 100}"
        names, fetched = self._run(sql, f"sample of {table.qualified}")
        frame = pd.DataFrame(fetched, columns=names)
        return tidy_frame(frame, columns)

    # -- values ------------------------------------------------------------

    def distinct_sample(
        self,
        table: TableInfo,
        requests: Mapping[str, tuple[ColumnInfo, int]],
        limit: int,
    ) -> dict[str, list[str]]:
        if not requests:
            return {}
        keys = list(requests)
        parts = []
        for index, key in enumerate(keys):
            column, rate = requests[key]
            check_rate(rate)
            norm = self._norm(column)
            condition = f"{norm} IS NOT NULL AND {self._in_sample(norm, rate)}"
            parts.append(
                f"groupUniqArrayIf({int(limit)})({self._text(column)}, "
                f"{condition}) AS s{index}"
            )
        sql = f"SELECT {', '.join(parts)} FROM {self._table(table)}"
        _, rows = self._run(sql, f"value sample of {table.qualified}")
        return {
            key: sorted(str(value) for value in rows[0][index] if value is not None)
            for index, key in enumerate(keys)
        }

    def probe(
        self,
        table: TableInfo,
        requests: Mapping[str, tuple[ColumnInfo, Sequence[str]]],
    ) -> dict[str, set[str]]:
        answer: dict[str, set[str]] = {key: set() for key in requests}
        for key, (column, values) in requests.items():
            norm = self._norm(column)
            wanted = sorted(set(values))
            for chunk in chunk_values(wanted, self.limits.max_sql_bytes):
                literals = ", ".join(_literal(value) for value in chunk)
                sql = (
                    f"SELECT DISTINCT {norm} AS v FROM {self._table(table)} "
                    f"WHERE {norm} IN ({literals})"
                )
                _, rows = self._run(sql, f"probe of {table.qualified}.{column.name}")
                answer[key].update(str(row[0]) for row in rows if row[0] is not None)
        return answer

    # -- links -------------------------------------------------------------

    def pair_profile(
        self, table: TableInfo, left: ColumnInfo, right: ColumnInfo
    ) -> dict[str, Any]:
        nl, nr = self._norm(left), self._norm(right)
        both = f"{nl} IS NOT NULL AND {nr} IS NOT NULL"
        sql = (
            f"SELECT countIf({both}) AS n, countIf({both} AND {nl} = {nr}) AS e, "
            f"uniqIf(({nl}, {nr}), {both}) AS p FROM {self._table(table)}"
        )
        _, rows = self._run(
            sql, f"pair profile of {table.qualified}.{left.name}/{right.name}"
        )
        pairs = int(rows[0][2])
        return {
            "rows": int(rows[0][0]),
            "equal_rows": int(rows[0][1]),
            "distinct_pairs": pairs,
            "distinct_pairs_exact": pairs < 60_000,
        }

    def fanout_sample(
        self,
        table: TableInfo,
        key: ColumnInfo,
        others: Sequence[ColumnInfo],
        rate: int,
        limit: int,
    ) -> pd.DataFrame:
        check_rate(rate)
        norm = self._norm(key)
        parts = [f"{norm} AS k", "count() AS n"]
        for index, other in enumerate(others):
            parts.append(f"uniqExact({self._norm(other)}) AS o{index}")
        sql = (
            f"SELECT {', '.join(parts)} FROM {self._table(table)} "
            f"WHERE {norm} IS NOT NULL AND {self._in_sample(norm, rate)} "
            f"GROUP BY k ORDER BY k LIMIT {int(limit)}"
        )
        _, rows = self._run(sql, f"fan-out of {table.qualified}.{key.name}")
        frame = pd.DataFrame(
            rows, columns=["key", "rows", *[other.name for other in others]]
        )
        for name in frame.columns[1:]:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0)
            frame[name] = frame[name].astype("int64")
        return frame

    def neighbours(
        self,
        table: TableInfo,
        left: ColumnInfo,
        right: ColumnInfo,
        values: Sequence[str],
        limit: int,
    ) -> list[tuple[str, str]]:
        nl, nr = self._norm(left), self._norm(right)
        pairs: list[tuple[str, str]] = []
        for chunk in chunk_values(sorted(set(values)), self.limits.max_sql_bytes):
            remaining = limit - len(pairs)
            if remaining <= 0:
                break
            literals = ", ".join(_literal(value) for value in chunk)
            sql = (
                f"SELECT DISTINCT {nl} AS l, {nr} AS r FROM {self._table(table)} "
                f"WHERE {nl} IN ({literals}) AND {nr} IS NOT NULL "
                f"LIMIT {int(remaining)}"
            )
            _, rows = self._run(sql, f"links of {table.qualified}.{left.name}")
            pairs.extend((str(a), str(b)) for a, b in rows)
        return pairs

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "host": self._host,
            "database": self._database,
        }


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if stamp is pd.NaT:
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp


def _is_timeout(error: BaseException) -> bool:
    text = str(error)
    return (
        "TIMEOUT_EXCEEDED" in text
        or "Code: 159" in text
        or "MEMORY_LIMIT_EXCEEDED" in text
        or "Code: 241" in text
        or "timed out" in text.lower()
    )


def _is_transient(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "connection reset",
            "connection refused",
            "connection aborted",
            "broken pipe",
            "remote end closed",
            "network is unreachable",
            "temporarily unavailable",
            "max retries exceeded",
        )
    )
