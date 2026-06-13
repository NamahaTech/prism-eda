"""Input normalization for DataFrames, files, and table collections."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd

from prism_eda.catalog.models import SourceInfo
from prism_eda.exceptions import DataLoadError, UnsupportedSourceError

PathLike: TypeAlias = str | Path
TableSource: TypeAlias = pd.DataFrame | PathLike
DataSource: TypeAlias = (
    pd.DataFrame | PathLike | Sequence[PathLike] | Mapping[str, TableSource]
)

SUPPORTED_SUFFIXES = {".csv", ".parquet"}


def _matches(path: Path, patterns: Sequence[str] | None) -> bool:
    if not patterns:
        return False
    return any(
        fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern)
        for pattern in patterns
    )


def _discover(
    directory: Path,
    *,
    recursive: bool,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and (not include or _matches(path, include))
        and not _matches(path, exclude)
    ]
    return sorted(paths)


def _format_options(
    suffix: str, read_options: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not read_options:
        return {}
    format_name = "csv" if suffix == ".csv" else "parquet"
    nested = read_options.get(format_name)
    if isinstance(nested, Mapping):
        return dict(nested)
    if any(key in read_options for key in ("csv", "parquet")):
        return {}
    return dict(read_options)


def _read(path: Path, read_options: Mapping[str, Any] | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    options = _format_options(suffix, read_options)
    try:
        if suffix == ".csv":
            return pd.read_csv(path, **options)
        if suffix == ".parquet":
            return pd.read_parquet(path, **options)
    except Exception as error:
        raise DataLoadError(f"Could not load {path}: {error}") from error
    raise UnsupportedSourceError(
        f"Unsupported file format {suffix!r}; expected CSV or Parquet."
    )


def _table_name(path: Path, names: Mapping[str, str] | None) -> str:
    if names:
        for key in (str(path), path.name, path.stem):
            if key in names:
                return names[key]
    return path.stem


def _validate_frame(name: str, frame: pd.DataFrame) -> None:
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DataLoadError(f"Table {name!r} has duplicate columns: {duplicates}")


def load_tables(
    source: DataSource,
    *,
    recursive: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    names: Mapping[str, str] | None = None,
    read_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, SourceInfo]]:
    """Normalize supported sources into named pandas tables and source metadata."""
    if isinstance(source, pd.DataFrame):
        tables = {"table": source}
        sources = {"table": SourceInfo(kind="dataframe")}
    elif isinstance(source, Mapping):
        tables = {}
        sources = {}
        for name, table_source in source.items():
            if isinstance(table_source, pd.DataFrame):
                frame = table_source
                source_info = SourceInfo(kind="dataframe")
            else:
                path = Path(table_source).expanduser()
                frame = _read(path, read_options)
                source_info = SourceInfo(
                    kind=path.suffix.lower().lstrip("."), location=str(path)
                )
            tables[str(name)] = frame
            sources[str(name)] = source_info
    else:
        if isinstance(source, (str, Path)):
            root = Path(source).expanduser()
            if root.is_dir():
                paths = _discover(
                    root,
                    recursive=recursive,
                    include=include,
                    exclude=exclude,
                )
            else:
                paths = [root]
        elif isinstance(source, Sequence):
            paths = [Path(path).expanduser() for path in source]
        else:
            raise UnsupportedSourceError(f"Unsupported source type: {type(source)!r}")

        if not paths:
            raise DataLoadError("No supported CSV or Parquet files were found.")
        tables = {}
        sources = {}
        for path in paths:
            if not path.exists():
                raise DataLoadError(f"Source does not exist: {path}")
            name = _table_name(path, names)
            if name in tables:
                raise DataLoadError(
                    f"Multiple sources resolve to table name {name!r}; provide names=."
                )
            tables[name] = _read(path, read_options)
            sources[name] = SourceInfo(
                kind=path.suffix.lower().lstrip("."), location=str(path)
            )

    if not tables:
        raise DataLoadError("A dataset must contain at least one table.")
    for name, frame in tables.items():
        _validate_frame(name, frame)
    return tables, sources
