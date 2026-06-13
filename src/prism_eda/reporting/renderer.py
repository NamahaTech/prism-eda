"""Jinja-based HTML renderer."""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from prism_eda.results import AnalysisResult


def _format_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1_000:
            return f"{value:,.2f}"
        return f"{value:.4g}"
    return str(value)


def _environment() -> Environment:
    environment = Environment(
        loader=PackageLoader("prism_eda", "reporting/templates"),
        autoescape=select_autoescape(("html", "xml")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["format_value"] = _format_value
    return environment


def render_html(result: AnalysisResult) -> str:
    """Render a complete report as a standalone HTML document."""
    template = _environment().get_template("report.html")
    return template.render(result=result)
