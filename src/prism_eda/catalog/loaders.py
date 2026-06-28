"""Input normalization for DataFrames, files, and table collections."""

from __future__ import annotations

import fnmatch
import warnings
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

CSV_SUFFIXES = {".csv"}
PARQUET_SUFFIXES = {".parquet"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
SUPPORTED_SUFFIXES = CSV_SUFFIXES | PARQUET_SUFFIXES | EXCEL_SUFFIXES


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


_FORMAT_KEYS = ("csv", "parquet", "excel")


def _format_name(suffix: str) -> str:
    if suffix in CSV_SUFFIXES:
        return "csv"
    if suffix in PARQUET_SUFFIXES:
        return "parquet"
    return "excel"


def _format_options(
    suffix: str, read_options: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not read_options:
        return {}
    nested = read_options.get(_format_name(suffix))
    if isinstance(nested, Mapping):
        return dict(nested)
    if any(key in read_options for key in _FORMAT_KEYS):
        return {}
    return dict(read_options)


def _read_excel(path: Path, options: dict[str, Any]) -> pd.DataFrame:
    try:
        with warnings.catch_warnings():
            # openpyxl warns "Workbook contains no default style" for many files
            # exported by other tools; it is harmless and just noise to users.
            warnings.filterwarnings(
                "ignore",
                message="Workbook contains no default style",
                category=UserWarning,
            )
            frame = pd.read_excel(path, **options)
    except ImportError as error:
        # pandas raises this when the Excel engine (e.g. openpyxl for .xlsx,
        # xlrd for legacy .xls) is not installed. Keep Excel an opt-in extra.
        raise DataLoadError(
            f"Reading Excel files requires an Excel engine. Install it with "
            f"\"pip install 'prism-eda[excel]'\", or the engine pandas names "
            f"below. Original error: {error}"
        ) from error
    if isinstance(frame, dict):
        sheets = ", ".join(map(str, frame))
        raise DataLoadError(
            f"{path} resolved to multiple sheets ({sheets}). Read one at a time, "
            f"e.g. read_options={{'excel': {{'sheet_name': '<sheet>'}}}}."
        )
    return frame


def _read(path: Path, read_options: Mapping[str, Any] | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    options = _format_options(suffix, read_options)
    try:
        if suffix in CSV_SUFFIXES:
            return pd.read_csv(path, **options)
        if suffix in PARQUET_SUFFIXES:
            return pd.read_parquet(path, **options)
        if suffix in EXCEL_SUFFIXES:
            return _read_excel(path, options)
    except DataLoadError:
        raise
    except Exception as error:
        raise DataLoadError(f"Could not load {path}: {error}") from error
    raise UnsupportedSourceError(
        f"Unsupported file format {suffix!r}; expected CSV, Parquet, or Excel."
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
