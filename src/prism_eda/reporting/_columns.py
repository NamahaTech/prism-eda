"""How a column is summarised in one table cell, shared by HTML and PDF."""

from __future__ import annotations

from typing import Any

from prism_eda.catalog.models import ColumnCatalog


def number(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:
            return "–"
        if abs(value) >= 1_000 or value.is_integer():
            return f"{value:,.0f}"
        return f"{value:,.4g}"
    return str(value)


def percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    share = float(value)
    if 0 < share < 0.001:
        return "<0.1%"
    if 0.999 < share < 1:
        return ">99.9%"
    return f"{share:.{digits}%}"


def clip(value: Any, limit: int = 60) -> str:
    text = " ".join(str(value).split()) if value is not None else ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def size(value: Any) -> str:
    if not value:
        return "–"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            digits = 0 if unit in {"B", "KB"} else 1
            return f"{amount:,.{digits}f} {unit}"
        amount /= 1024
    return "–"


def _moment(value: Any) -> str:
    text = str(value) if value else "–"
    return text[:16].replace("T", " ")


def column_detail(column: ColumnCatalog, is_key: bool) -> str:
    """Range for a measure, period for a date, top values for a category,
    and example values for an identifier."""
    stats = column.statistics
    if stats.get("profiled") is False:
        return "not profiled"
    kind = stats.get("kind") or column.semantic_type
    if column.non_null_count == 0:
        return "empty"
    numeric = kind in {"integer", "float", "numeric"}
    if numeric and stats.get("min") is not None and not is_key:
        text = f"{number(stats['min'])} – {number(stats['max'])}"
        if stats.get("median") is not None:
            approx = "≈ " if stats.get("quantiles_basis") == "sample" else ""
            text += f" · median {approx}{number(stats['median'])}"
        return text
    if kind in {"datetime", "date"} and stats.get("min") is not None:
        return f"{_moment(stats['min'])} → {_moment(stats['max'])}"
    if is_key or (column.unique_rate or 0) >= 0.5:
        # An identifier's "most frequent value" says nothing; show what a
        # value looks like instead.
        examples = [
            clip(entry["value"], 20)
            for entry in column.top_values[:2]
            if str(entry["value"]).strip()
        ]
        text = ("e.g. " + ", ".join(examples)) if examples else ""
        if stats.get("min_length") is not None:
            low, high = stats["min_length"], stats["max_length"]
            length = number(low) if low == high else f"{number(low)}–{number(high)}"
            text += f"{' · ' if text else ''}{length} characters"
        return text
    if column.top_values:
        approx = "≈ " if stats.get("top_values_basis") == "sample" else ""
        total = column.row_count or 1
        shown = []
        for entry in column.top_values[:3]:
            label = entry["value"] if str(entry["value"]).strip() else "(blank)"
            share = percent(entry["count"] / total, 0)
            shown.append(f"{clip(label, 18)} ({approx}{share})")
        return ", ".join(shown)
    if stats.get("min_length") is not None:
        return f"length {number(stats['min_length'])}–{number(stats['max_length'])}"
    if stats.get("min") is not None:
        return f"{number(stats['min'])} – {number(stats['max'])}"
    return ""


def column_flags(column: ColumnCatalog, is_shared: bool) -> list[str]:
    flags = []
    if "unique_key_candidate" in column.roles:
        flags.append("unique")
    if is_shared:
        flags.append("shared key")
    if column.unique_count == 1 and column.non_null_count:
        flags.append("constant")
    blank = column.statistics.get("blank_count")
    if blank and column.non_null_count and blank / column.non_null_count >= 0.01:
        flags.append(f"{percent(blank / column.non_null_count, 0)} blank")
    return flags


def shared_columns(evidence: Any) -> set[tuple[str, str]]:
    """Columns that share identifiers with another column."""
    shared: set[tuple[str, str]] = set()
    for item in evidence:
        if item.kind == "explore_relationship" and item.value["tier"] != "possible":
            for side in ("child", "parent"):
                shared.add((item.value[side]["table"], item.value[side]["column"]))
    return shared
