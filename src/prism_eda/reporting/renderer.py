"""Jinja-based HTML renderer."""

from __future__ import annotations

from importlib import resources
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from prism_eda.reporting.charts import (
    format_cell,
    histogram_svg,
    peer_group_svg,
    scatter_svg,
    why_bars_svg,
)
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
    environment.filters["format_cell"] = format_cell
    environment.filters["histogram_svg"] = histogram_svg
    environment.filters["scatter_svg"] = scatter_svg
    environment.filters["why_bars_svg"] = why_bars_svg
    environment.filters["peer_group_svg"] = peer_group_svg
    return environment


def _load_cytoscape_js() -> str | None:
    """Return the vendored Cytoscape.js source, or None when unavailable."""
    try:
        asset = resources.files("prism_eda.reporting").joinpath(
            "assets/cytoscape.min.js"
        )
        text = asset.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return None
    # A literal "</script" inside the inlined payload would terminate the
    # surrounding script tag early and break the document.
    return text.replace("</script", "<\\/script")


def render_html(result: AnalysisResult) -> str:
    """Render a complete report as a standalone HTML document."""
    template = _environment().get_template("report.html")
    cytoscape_js = _load_cytoscape_js() if result.goal == "schema_discovery" else None
    return template.render(result=result, cytoscape_js=cytoscape_js)
