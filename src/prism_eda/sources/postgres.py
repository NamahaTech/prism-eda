"""Postgres as an exploration source.

Every question runs in its own read-only transaction with a statement timeout,
on a connection opened for that question and closed straight after.

Postgres is a row store, so a question that would need one full scan per column
is phrased to need one scan per table instead, and anything that would sort a
very large table (exact distinct counts, exact quantiles) is answered from the
planner's statistics or left to the row sample, and reported as an estimate.

Requires the ``postgres`` extra (``pip install 'prism-eda[postgres]'``), or an
existing ``psycopg2`` installation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from prism_eda.sources._sql import (
    LARGE_TABLE_ROWS,
    SAMPLE_TEXT_LIMIT,
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

_SYSTEM_SCHEMAS = ("pg_catalog", "information_schema")
_INTEGER_TYPES = {"smallint", "integer", "bigint", "oid"}
_FLOAT_TYPES = {"real", "double precision", "money"}
_STRING_TYPES = {
    "text",
    "character varying",
    "character",
    "uuid",
    "citext",
    "name",
    "inet",
    "cidr",
    "macaddr",
    '"char"',
}


def postgres_kind(type_name: str, type_class: str, scale: int | None) -> str:
    base = type_name.split("(")[0].strip()
    if base in _INTEGER_TYPES:
        return "integer"
    if base in {"numeric", "decimal"}:
        # numeric(20,0) is how many core-banking schemas store identifiers.
        return "integer" if scale == 0 else "float"
    if base in _FLOAT_TYPES:
        return "float"
    if base == "boolean":
        return "boolean"
    if base.startswith("timestamp"):
        return "datetime"
    if base == "date":
        return "date"
    if base in _STRING_TYPES or type_class == "e":
        return "string"
    return "other"


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class PostgresSource(Source):
    """Explore the tables of a Postgres database.

    Pass connection details (or a ``dsn``), or ``connect`` — a callable
    returning a DB-API connection — when the application already knows how to
    connect. ``schemas`` limits which schemas are listed; every non-system
    schema by default.
    """

    kind = "postgres"

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        dsn: str | None = None,
        schemas: Sequence[str] | str | None = None,
        include_views: bool = False,
        name: str = "postgres",
        limits: SourceLimits | None = None,
        connect: Callable[[], Any] | None = None,
        connect_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = safe_name(name)
        self.limits = limits or SourceLimits()
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._database = database
        self._dsn = dsn
        self._schemas = [schemas] if isinstance(schemas, str) else schemas
        self._include_views = include_views
        self._connect = connect
        self._options = dict(connect_options or {})

    # -- connection --------------------------------------------------------

    def _open(self) -> Any:
        if self._connect is not None:
            connection = self._connect()
            if connection is None:
                raise SourceError("The Postgres connection factory returned nothing")
            return connection
        options: dict[str, Any] = {"connect_timeout": self.limits.connect_timeout}
        options.update(self._options)
        for key, value in (
            ("host", self._host),
            ("port", self._port),
            ("user", self._username),
            ("password", self._password),
            ("dbname", self._database),
        ):
            if value is not None:
                options[key] = value
        try:
            import psycopg

            if self._dsn:
                return psycopg.connect(self._dsn, **options)
            return psycopg.connect(**options)
        except ImportError:
            pass
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise SourceError(
                "Postgres support needs the 'postgres' extra: "
                "pip install 'prism-eda[postgres]'"
            ) from error
        if self._dsn:
            return psycopg2.connect(self._dsn, **options)
        return psycopg2.connect(**options)

    def _run(
        self, sql: str, params: Sequence[Any] | None, label: str
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        def once() -> tuple[list[str], list[tuple[Any, ...]]]:
            connection = self._open()
            try:
                try:
                    connection.autocommit = False
                except Exception:  # noqa: BLE001 - already in a transaction
                    pass
                cursor = connection.cursor()
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(
                    "SET LOCAL statement_timeout = "
                    f"{int(self.limits.query_timeout) * 1000}"
                )
                if params is None:
                    cursor.execute(sql)
                else:
                    cursor.execute(sql, list(params))
                names = [column[0] for column in cursor.description or ()]
                rows = [tuple(_plain(row)) for row in cursor.fetchall()]
                cursor.close()
                connection.rollback()
                return names, rows
            finally:
                try:
                    connection.close()
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

    @staticmethod
    def _text(column: ColumnInfo) -> str:
        return f"({_ident(column.name)})::text"

    def _norm(self, column: ColumnInfo) -> str:
        text = self._text(column)
        return f"nullif(ltrim(lower(btrim({text}, ' ') COLLATE \"C\"), '0'), '')"

    @staticmethod
    def _in_sample(norm: str, rate: int) -> str:
        if rate <= 1:
            return f"{norm} IS NOT NULL"
        return (
            f"(('x' || substr(md5({norm}), 1, 16))::bit(64)::bigint & {rate - 1}) = 0"
        )

    # -- inventory ---------------------------------------------------------

    def tables(self) -> list[TableInfo]:
        kinds = ["r", "p"]
        if self._include_views:
            kinds.extend(["v", "m"])
        sql = (
            "SELECT n.nspname, c.relname, c.relkind::text, c.reltuples::bigint, "
            "CASE WHEN c.relkind IN ('r', 'm') THEN pg_table_size(c.oid) END, "
            "obj_description(c.oid, 'pg_class') "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relkind = ANY(%s) AND NOT c.relispartition "
            "AND n.nspname <> ALL(%s) AND n.nspname NOT LIKE %s "
            "AND has_table_privilege(c.oid, 'SELECT') "
        )
        params: list[Any] = [kinds, list(_SYSTEM_SCHEMAS), "pg\\_%"]
        if self._schemas:
            sql += "AND n.nspname = ANY(%s) "
            params.append(list(self._schemas))
        sql += "ORDER BY n.nspname, c.relname"
        _, rows = self._run(sql, params, "list tables")
        labels = {"r": "table", "p": "table", "v": "view", "m": "view"}
        return [
            TableInfo(
                source=self.name,
                schema=str(schema),
                name=str(name),
                kind=labels.get(str(kind), "table"),
                # reltuples is -1 for a table that has never been analysed.
                approx_rows=(
                    to_int(estimate)
                    if estimate is not None and int(estimate) >= 0
                    else None
                ),
                bytes=to_int(size),
                engine="partitioned" if kind == "p" else None,
                comment=str(comment) if comment else None,
            )
            for schema, name, kind, estimate, size, comment in rows
        ]

    def columns(self, table: TableInfo) -> list[ColumnInfo]:
        sql = (
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod), t.typtype::text, "
            "CASE WHEN a.atttypid = 1700 AND a.atttypmod >= 0 "
            "THEN (a.atttypmod - 4) & 65535 END, NOT a.attnotnull "
            "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_type t ON t.oid = a.atttypid "
            "WHERE n.nspname = %s AND c.relname = %s AND a.attnum > 0 "
            "AND NOT a.attisdropped ORDER BY a.attnum"
        )
        _, rows = self._run(
            sql, [table.schema, table.name], f"columns of {table.qualified}"
        )
        return [
            ColumnInfo(
                name=str(name),
                native_type=str(native),
                kind=postgres_kind(str(native), str(type_class), to_int(scale)),
                nullable=bool(nullable),
            )
            for name, native, type_class, scale, nullable in rows
        ]

    def row_count(self, table: TableInfo) -> int:
        _, rows = self._run(
            f"SELECT count(*) FROM {self._table(table)}",
            None,
            f"row count of {table.qualified}",
        )
        return int(rows[0][0])

    # -- column profile ----------------------------------------------------

    def _planner_distinct(self, table: TableInfo, rows: int) -> dict[str, int]:
        sql = (
            "SELECT attname, n_distinct FROM pg_stats "
            "WHERE schemaname = %s AND tablename = %s"
        )
        try:
            _, found = self._run(
                sql, [table.schema, table.name], f"planner stats of {table.qualified}"
            )
        except SourceError:
            return {}
        answer = {}
        for name, value in found:
            if value is None:
                continue
            number = float(value)
            # A negative n_distinct is minus the fraction of rows that are
            # distinct; a positive one is the count itself.
            answer[str(name)] = (
                int(round(-number * rows)) if number < 0 else int(number)
            )
        return answer

    def column_stats(
        self, table: TableInfo, columns: Sequence[ColumnInfo]
    ) -> dict[str, ColumnStats]:
        large = (table.approx_rows or 0) > LARGE_TABLE_ROWS
        parts = ["count(*) AS rows_total"]
        for index, column in enumerate(columns):
            c = _ident(column.name)
            p = f"c{index}"
            parts.append(f"count({c}) AS {p}_nn")
            if not large and column.kind != "other":
                parts.append(f"count(DISTINCT {c}) AS {p}_d")
            if column.kind in {"integer", "float"}:
                value = f"({c})::float8"
                ok = (
                    f"FILTER (WHERE {value} NOT IN ('NaN', 'Infinity', '-Infinity'))"
                    if column.kind == "float"
                    else ""
                )
                parts.extend(
                    [
                        f"min({value}) {ok} AS {p}_min",
                        f"max({value}) {ok} AS {p}_max",
                        f"avg({value}) {ok} AS {p}_avg",
                        f"stddev_pop({value}) {ok} AS {p}_std",
                        f"count(*) FILTER (WHERE {value} = 0) AS {p}_zero",
                        f"count(*) FILTER (WHERE {value} < 0) AS {p}_neg",
                    ]
                )
                if not large:
                    parts.append(
                        "percentile_cont(ARRAY[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) "
                        f"WITHIN GROUP (ORDER BY {value}) {ok} AS {p}_q"
                    )
            elif column.kind in {"datetime", "date"}:
                parts.extend(
                    [f"min({c})::text AS {p}_min", f"max({c})::text AS {p}_max"]
                )
            elif column.kind == "string":
                text = self._text(column)
                parts.extend(
                    [
                        f"count(*) FILTER (WHERE btrim({text}, ' ') = '') AS {p}_blank",
                        f"min(char_length({text})) AS {p}_lmin",
                        f"max(char_length({text})) AS {p}_lmax",
                        f"avg(char_length({text})) AS {p}_lavg",
                    ]
                )
        sql = f"SELECT {', '.join(parts)} FROM {self._table(table)}"
        names, rows = self._run(sql, None, f"statistics of {table.qualified}")
        row = dict(zip(names, rows[0], strict=True))
        total = int(row["rows_total"])
        estimates = self._planner_distinct(table, total) if large else {}
        answer: dict[str, ColumnStats] = {}
        for index, column in enumerate(columns):
            p = f"c{index}"
            non_null = int(row[f"{p}_nn"])
            if large:
                distinct = estimates.get(column.name)
                if distinct is not None:
                    distinct = max(0, min(distinct, non_null))
            else:
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
                distinct_exact=not large,
                **extra,
            )
        return answer

    def top_values(
        self, table: TableInfo, columns: Sequence[ColumnInfo], limit: int
    ) -> dict[str, list[tuple[Any, int]]]:
        if not columns:
            return {}
        # One scan for every column: each grouping set counts one column.
        texts = [self._text(column) for column in columns]
        sets = ", ".join(f"({text})" for text in texts)
        tag = " + ".join(
            f"(CASE WHEN GROUPING({text}) = 0 THEN {index} ELSE 0 END)"
            for index, text in enumerate(texts)
        )
        value = "COALESCE(" + ", ".join(texts) + ")"
        sql = (
            f"SELECT tag, v, n FROM (SELECT tag, v, n, row_number() OVER "
            f"(PARTITION BY tag ORDER BY n DESC, v) AS position FROM ("
            f"SELECT {tag} AS tag, {value} AS v, count(*) AS n "
            f"FROM {self._table(table)} GROUP BY GROUPING SETS ({sets})"
            f") counted WHERE v IS NOT NULL) ranked WHERE position <= {int(limit)} "
            "ORDER BY tag, n DESC, v"
        )
        _, rows = self._run(sql, None, f"top values of {table.qualified}")
        answer: dict[str, list[tuple[Any, int]]] = {
            column.name: [] for column in columns
        }
        for tag_index, found, count in rows:
            answer[columns[int(tag_index)].name].append((found, int(count)))
        return answer

    def time_histogram(
        self, table: TableInfo, column: ColumnInfo, grain: str
    ) -> list[tuple[str, int]]:
        if column.kind == "date" and grain == "hour":
            grain = "day"
        c = _ident(column.name)
        sql = (
            f"SELECT date_trunc('{grain}', {c}::timestamp)::text AS b, count(*) AS n "
            f"FROM {self._table(table)} WHERE {c} IS NOT NULL "
            "GROUP BY b ORDER BY b LIMIT 5000"
        )
        _, rows = self._run(sql, None, f"timeline of {table.qualified}.{column.name}")
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
                    f"left({self._text(column)}, {SAMPLE_TEXT_LIMIT}) AS {c}"
                )
            elif column.kind in {"datetime", "date"}:
                selected.append(f"({c})::timestamp AS {c}")
            else:
                selected.append(c)
        source = self._table(table)
        conditions: list[str] = []
        params: list[Any] = []
        if where is not None:
            column, values = where
            wanted = sorted(
                {norm for value in values if (norm := normalise_value(value))}
            )
            if not wanted:
                return pd.DataFrame(columns=[column.name for column in columns])
            conditions.append(f"{self._norm(column)} = ANY(%s)")
            params.append(wanted)
        if total_rows > rows:
            percent = min(100.0, max(0.0001, rows / total_rows * 100.0))
            if where is None and table.kind == "table":
                # Block sampling on big tables avoids reading every page;
                # row sampling on smaller ones avoids clustered samples.
                method = "SYSTEM" if total_rows > LARGE_TABLE_ROWS else "BERNOULLI"
                source += (
                    f" TABLESAMPLE {method} ({percent:.6f}) REPEATABLE ({int(seed)})"
                )
            elif hash_columns:
                hashed = " || '|' || ".join(
                    f"coalesce({self._text(column)}, '')" for column in hash_columns
                )
                threshold = max(1, round(rows / total_rows * 1_000_000))
                conditions.append(
                    f"mod(hashtext({hashed} || '{int(seed)}')::bigint & 2147483647, "
                    f"1000000) < {threshold}"
                )
        sql = f"SELECT {', '.join(selected)} FROM {source}"
        if conditions:
            sql += f" WHERE {' AND '.join(conditions)}"
        sql += f" LIMIT {int(rows * 1.5) + 100}"
        names, fetched = self._run(sql, params or None, f"sample of {table.qualified}")
        frame = pd.DataFrame(fetched, columns=names)
        return tidy_frame(frame, columns)

    # -- values ------------------------------------------------------------

    def distinct_sample(
        self,
        table: TableInfo,
        requests: Mapping[str, tuple[ColumnInfo, int]],
        limit: int,
    ) -> dict[str, list[str]]:
        answer: dict[str, list[str]] = {}
        for key, (column, rate) in requests.items():
            check_rate(rate)
            norm = self._norm(column)
            sql = (
                f"SELECT DISTINCT {self._text(column)} FROM {self._table(table)} "
                f"WHERE {norm} IS NOT NULL AND {self._in_sample(norm, rate)} "
                f"LIMIT {int(limit)}"
            )
            _, rows = self._run(
                sql, None, f"value sample of {table.qualified}.{column.name}"
            )
            answer[key] = sorted(str(row[0]) for row in rows if row[0] is not None)
        return answer

    def probe(
        self,
        table: TableInfo,
        requests: Mapping[str, tuple[ColumnInfo, Sequence[str]]],
    ) -> dict[str, set[str]]:
        if not requests:
            return {}
        keys = list(requests)
        pool = sorted({value for key in keys for value in requests[key][1]})
        if not pool:
            return {key: set() for key in keys}
        # One scan for every column: each row is unpivoted into (column, value)
        # pairs and hash-joined to the probe list.
        pairs = ", ".join(
            f"({index}, {self._norm(requests[key][0])})"
            for index, key in enumerate(keys)
        )
        sql = (
            f"SELECT DISTINCT x.tag, x.v FROM {self._table(table)} "
            f"CROSS JOIN LATERAL (VALUES {pairs}) AS x(tag, v) "
            "JOIN (SELECT DISTINCT unnest(%s::text[]) AS v) wanted ON wanted.v = x.v"
        )
        _, rows = self._run(sql, [pool], f"probe of {table.qualified}")
        answer: dict[str, set[str]] = {key: set() for key in keys}
        for tag, value in rows:
            answer[keys[int(tag)]].add(str(value))
        for key in keys:
            answer[key].intersection_update(requests[key][1])
        return answer

    # -- links -------------------------------------------------------------

    def pair_profile(
        self, table: TableInfo, left: ColumnInfo, right: ColumnInfo
    ) -> dict[str, Any]:
        nl, nr = self._norm(left), self._norm(right)
        both = f"{nl} IS NOT NULL AND {nr} IS NOT NULL"
        large = (table.approx_rows or 0) > LARGE_TABLE_ROWS
        distinct = (
            "NULL::bigint"
            if large
            else f"count(DISTINCT ({nl}, {nr})) FILTER (WHERE {both})"
        )
        sql = (
            f"SELECT count(*) FILTER (WHERE {both}), "
            f"count(*) FILTER (WHERE {both} AND {nl} = {nr}), {distinct} "
            f"FROM {self._table(table)}"
        )
        _, rows = self._run(
            sql, None, f"pair profile of {table.qualified}.{left.name}/{right.name}"
        )
        return {
            "rows": int(rows[0][0]),
            "equal_rows": int(rows[0][1]),
            "distinct_pairs": to_int(rows[0][2]),
            "distinct_pairs_exact": not large,
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
        parts = [f"{norm} AS k", "count(*) AS n"]
        for index, other in enumerate(others):
            parts.append(f"count(DISTINCT {self._norm(other)}) AS o{index}")
        sql = (
            f"SELECT {', '.join(parts)} FROM {self._table(table)} "
            f"WHERE {norm} IS NOT NULL AND {self._in_sample(norm, rate)} "
            f"GROUP BY 1 ORDER BY 1 LIMIT {int(limit)}"
        )
        _, rows = self._run(sql, None, f"fan-out of {table.qualified}.{key.name}")
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
        wanted = sorted(set(values))
        if not wanted:
            return []
        nl, nr = self._norm(left), self._norm(right)
        sql = (
            f"SELECT DISTINCT e.l, e.r FROM (SELECT {nl} AS l, {nr} AS r "
            f"FROM {self._table(table)}) e "
            "JOIN (SELECT DISTINCT unnest(%s::text[]) AS v) wanted ON wanted.v = e.l "
            f"WHERE e.r IS NOT NULL LIMIT {int(limit)}"
        )
        _, rows = self._run(sql, [wanted], f"links of {table.qualified}.{left.name}")
        return [(str(a), str(b)) for a, b in rows]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "host": self._host,
            "database": self._database,
        }


def _plain(row: Any) -> Sequence[Any]:
    """A plain tuple from whatever row type the driver's cursor returns."""
    if isinstance(row, Mapping):
        return tuple(row.values())
    return tuple(row)


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
    text = str(error).lower()
    code = getattr(error, "sqlstate", None) or getattr(error, "pgcode", None)
    return code == "57014" or "statement timeout" in text


def _is_transient(error: BaseException) -> bool:
    text = str(error).lower()
    code = str(getattr(error, "sqlstate", None) or getattr(error, "pgcode", None) or "")
    return code.startswith("08") or any(
        marker in text
        for marker in (
            "connection reset",
            "connection refused",
            "server closed the connection",
            "could not connect",
            "connection timed out",
            "terminating connection",
            "broken pipe",
        )
    )
