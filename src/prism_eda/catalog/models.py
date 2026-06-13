"""Serializable catalog models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism_eda._serialization import to_jsonable


@dataclass(frozen=True, slots=True)
class SourceInfo:
    kind: str
    location: str | None = None


@dataclass(frozen=True, slots=True)
class ColumnCatalog:
    name: str
    physical_type: str
    semantic_type: str
    roles: tuple[str, ...]
    row_count: int
    non_null_count: int
    missing_count: int
    missing_rate: float
    unique_count: int | None
    unique_rate: float | None
    statistics: dict[str, Any] = field(default_factory=dict)
    top_values: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class TableCatalog:
    name: str
    row_count: int
    column_count: int
    memory_bytes: int
    duplicate_row_count: int | None
    fingerprint: str
    fingerprint_method: str
    source: SourceInfo
    columns: tuple[ColumnCatalog, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class DatasetCatalog:
    fingerprint: str
    fingerprint_method: str
    table_count: int
    row_count: int
    column_count: int
    tables: tuple[TableCatalog, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def table(self, name: str) -> TableCatalog:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(name)
