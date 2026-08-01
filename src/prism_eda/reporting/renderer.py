"""Jinja-based HTML renderer."""

from __future__ import annotations

from base64 import b64encode
from importlib import resources
from pathlib import PurePath
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from prism_eda.evidence.models import split_findings
from prism_eda.reporting.charts import (
    association_heatmap_svg,
    category_bars_svg,
    co_missing_heatmap_svg,
    format_cell,
    histogram_svg,
    image_dimension_svg,
    label_bars_svg,
    missing_bars_svg,
    peer_group_svg,
    scatter_svg,
    timeline_svg,
    why_bars_svg,
)
from prism_eda.reporting.sections import report_sections
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
    environment.filters["image_dimension_svg"] = image_dimension_svg
    environment.filters["label_bars_svg"] = label_bars_svg
    environment.filters["category_bars_svg"] = category_bars_svg
    environment.filters["missing_bars_svg"] = missing_bars_svg
    environment.filters["association_heatmap_svg"] = association_heatmap_svg
    environment.filters["co_missing_heatmap_svg"] = co_missing_heatmap_svg
    environment.filters["timeline_svg"] = timeline_svg
    return environment


def _data_uri(asset: str, media_type: str) -> str | None:
    """Base64-encode a packaged asset for inlining, or None when unavailable.

    Reports must stay single self-contained files with no network requests, so
    images travel inside the document. A missing asset degrades the masthead
    rather than failing the render.
    """
    try:
        payload = resources.files("prism_eda.reporting").joinpath(asset).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return None
    return f"data:{media_type};base64,{b64encode(payload).decode('ascii')}"


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
    issues, observations = split_findings(result.findings)
    return template.render(
        result=result,
        cytoscape_js=cytoscape_js,
        sections=report_sections(result),
        issues=issues,
        observations=observations,
        dataset_title=dataset_title(result),
        source_location=_source_location(result),
        logo_uri=_data_uri("assets/logo.png", "image/png"),
        favicon_uri=_data_uri("assets/favicon.png", "image/png"),
        column_charts=_by_column(result, "profile_distribution"),
        column_frequencies=_by_column(result, "profile_category_frequency"),
        column_timelines=_by_column(result, "profile_timeline"),
    )


def _by_column(result: AnalysisResult, kind: str) -> dict[tuple[str, str], Any]:
    """Index chart evidence by (table, column) so a card can look its own up.

    Cards render from the catalog, which every recipe produces. Chart evidence
    only exists for the baseline profile, so a card enriches itself when the
    evidence is there and renders plainly when it is not — that is what lets the
    same card markup serve all five reports.
    """
    return {
        (str(item.scope.table), item.scope.columns[0]): item.value
        for item in result.evidence
        if item.kind == kind and item.scope.columns
    }


def _source_location(result: AnalysisResult) -> str | None:
    """Where the data came from, for the masthead — provenance, not decoration."""
    locations = {
        str(table.source.location)
        for table in result.catalog.tables
        if table.source.location
    }
    if not locations:
        return None
    if len(locations) == 1:
        return next(iter(locations))
    return f"{len(locations)} sources"


def dataset_title(result: AnalysisResult) -> str:
    """Name the data the report is about, preferring the source over a label.

    A report headed with the file it profiled tells the reader which of their
    five extracts they are looking at. A generic product line does not.
    """
    tables = result.catalog.tables
    if not tables:
        return "Data profile"
    if len(tables) > 1:
        return f"{len(tables)} tables"
    table = tables[0]
    location = table.source.location
    if location:
        stem = PurePath(str(location)).name
        if stem:
            return stem
    return table.name
