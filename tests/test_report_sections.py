"""The navigation and the page must never disagree about what exists.

`reporting/sections.py` decides which sections a report has; the template renders
them. If those two drift, the report grows a nav link to nowhere or a section
nobody can reach. Every recipe is checked here, because the template is shared.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

import prism_eda as pe
from examples.sample_data import load_sample, sample_images

NAV_LINK = re.compile(r'<a href="#(section-[\w-]+)">([^<]*)</a>')
SECTION_ID = re.compile(r'<section[^>]*id="(section-[\w-]+)"')
SECTION_INDEX = re.compile(r'<div class="sec-idx"[^>]*>(.*?)</div>')


@pytest.fixture(scope="module")
def dataset() -> pe.Dataset:
    tables = {
        name: frame
        for name, frame in load_sample().items()
        if isinstance(frame, pd.DataFrame)
    }
    return pe.load(tables)


def _reports(dataset: pe.Dataset, tmp_path_factory) -> dict[str, str]:
    folder = sample_images(tmp_path_factory.mktemp("images"))
    return {
        "profile": dataset.profile().render_html(),
        "schema_discovery": dataset.discover_schema().render_html(),
        "anomaly_detection": dataset.anomaly_detection(table="orders").render_html(),
        "classification": dataset.classification(
            "churned", table="customers"
        ).render_html(),
        "image_profile": pe.profile_images(folder).render_html(),
    }


@pytest.fixture(scope="module")
def reports(dataset, tmp_path_factory) -> dict[str, str]:
    return _reports(dataset, tmp_path_factory)


@pytest.mark.parametrize(
    "goal",
    [
        "profile",
        "schema_discovery",
        "anomaly_detection",
        "classification",
        "image_profile",
    ],
)
def test_every_nav_link_reaches_a_section_and_back(reports, goal) -> None:
    html = reports[goal]
    nav = [anchor for anchor, _ in NAV_LINK.findall(html)]
    rendered = SECTION_ID.findall(html)

    assert nav, f"{goal} report has no navigation"
    assert [anchor for anchor in nav if anchor not in rendered] == []
    assert [anchor for anchor in rendered if anchor not in nav] == []
    assert len(nav) == len(set(nav)), "duplicate anchors in the navigation"


@pytest.mark.parametrize(
    "goal",
    [
        "profile",
        "schema_discovery",
        "anomaly_detection",
        "classification",
        "image_profile",
    ],
)
def test_every_section_header_is_numbered(reports, goal) -> None:
    """A blank number means the section list did not know about the section."""
    numbers = [
        value.strip()
        for value in SECTION_INDEX.findall(reports[goal])
        # Sub-headings inside a section deliberately hide their index slot.
        if "visibility:hidden" not in value
    ]

    assert numbers
    assert all(value.isdigit() for value in numbers), numbers


def test_profile_sections_are_ordered_for_reading(reports) -> None:
    labels = [label for _, label in NAV_LINK.findall(reports["profile"])]

    assert labels[:2] == ["Issues", "Alerts"]
    for earlier, later in (("Alerts", "Correlations"), ("Correlations", "Sample rows")):
        assert labels.index(earlier) < labels.index(later)


def test_reports_carry_no_product_copy(reports) -> None:
    for goal, html in reports.items():
        assert "Decision-first" not in html, goal
        assert "backed by structured evidence" not in html, goal


def test_column_cards_replace_the_table_in_every_report(reports) -> None:
    for goal, html in reports.items():
        assert 'class="col-card"' in html, goal
        assert "Role candidates</th>" not in html, goal


def test_profile_renders_the_analysis_sections(reports) -> None:
    html = reports["profile"]

    assert 'id="section-correlations"' in html
    assert 'id="section-interactions"' in html
    assert 'id="section-sample"' in html
    assert "Cram" in html  # the statistic is named, not just the number
    assert "not a hypothesis test" in html


def test_a_clean_single_column_table_still_renders(tmp_path) -> None:
    """The smallest possible profile must not trip the section machinery."""
    result = pe.profile(pd.DataFrame({"value": [1, 2, 3]}))

    html = result.to_html(tmp_path / "tiny.html").read_text(encoding="utf-8")

    nav = [anchor for anchor, _ in NAV_LINK.findall(html)]
    rendered = SECTION_ID.findall(html)
    assert sorted(nav) == sorted(rendered)
