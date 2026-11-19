"""Branding is one switch, and it has to reach every report.

A report that still says the library's name in one footer is a report an
integrator cannot hand to their own customer. So the check here is deliberately
blunt: with branding off, the product name must not appear anywhere in the
rendered document, for every recipe, including the ones with their own
templates.
"""

from __future__ import annotations

from base64 import b64encode
from importlib import resources

import pandas as pd
import pytest

import prism_eda as pe
from examples.bank_data import bank_tables
from examples.sample_data import (
    customer_segments,
    daily_orders,
    load_sample,
    sample_images,
    subscriptions,
)
from prism_eda.branding import Branding, resolve_branding
from tests.test_features import _featureset, _txns

#: A 1x1 transparent PNG, standing in for an integrator's own logo.
PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff"
    b"?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82"
)

GOALS = (
    "profile",
    "schema_discovery",
    "anomaly_detection",
    "classification",
    "regression",
    "time_series",
    "clustering",
    "image_profile",
    "feature_plan",
    "comparison",
    "exploration",
)


def _packaged(asset: str) -> str:
    payload = resources.files("prism_eda.reporting").joinpath(asset).read_bytes()
    return b64encode(payload).decode("ascii")


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> dict[str, object]:
    """One result per report the library can render."""
    tables = {
        name: frame
        for name, frame in load_sample().items()
        if isinstance(frame, pd.DataFrame)
    }
    dataset = pe.load(tables)
    folder = sample_images(tmp_path_factory.mktemp("images"))
    frame = _txns()
    return {
        "profile": dataset.profile(),
        "schema_discovery": dataset.discover_schema(),
        "anomaly_detection": dataset.anomaly_detection(table="orders"),
        "classification": dataset.classification("churned", table="customers"),
        "regression": pe.load({"subscriptions": subscriptions()}).regression(
            "monthly_revenue"
        ),
        "time_series": pe.load({"daily_orders": daily_orders()}).time_series(
            "orders", entity_id="store", horizon=28
        ),
        "clustering": pe.load({"members": customer_segments()}).clustering(),
        "image_profile": pe.profile_images(folder),
        "feature_plan": _featureset().plan(frame).run(frame).report,
        "comparison": pe.compare_datasets(
            {"orders": tables["orders"].head(200)},
            {"orders": tables["orders"].tail(200)},
        ),
        "exploration": pe.explore(bank_tables(customers=300, transactions=6000)),
    }


def test_every_report_is_covered(results) -> None:
    assert set(results) == set(GOALS)


@pytest.mark.parametrize("goal", GOALS)
def test_default_report_carries_the_library_identity(results, goal) -> None:
    html = results[goal].render_html()

    assert "Prism" in html
    assert "<title>" in html and "— Prism EDA</title>" in html
    assert _packaged("assets/logo.png")[:200] in html
    assert _packaged("assets/favicon.png")[:200] in html
    assert 'class="spectrum-rule"' in html
    assert "Generated locally by Prism EDA." in html


@pytest.mark.parametrize("goal", GOALS)
def test_unbranded_report_never_names_the_product(results, goal) -> None:
    html = results[goal].render_html(branding=False)

    assert "prism" not in html.lower()
    assert _packaged("assets/logo.png")[:200] not in html
    assert _packaged("assets/favicon.png")[:200] not in html
    assert 'rel="icon"' not in html
    assert 'class="brand-mark"' not in html
    assert 'class="spectrum-rule"' not in html
    assert "linear-gradient(90deg,#6d5bd0" not in html
    assert "Generated" not in html
    # Still a complete, titled report.
    assert html.startswith("<!doctype html>")
    assert "<title>" in html and "<h1>" in html


@pytest.mark.parametrize("goal", GOALS)
def test_unbranded_report_keeps_its_substance(results, goal) -> None:
    """Switching branding off removes identity, never content."""
    branded = results[goal].render_html()
    plain = results[goal].render_html(branding=False)

    assert branded.count("<section") == plain.count("<section")
    assert branded.count("<svg") == plain.count("<svg")
    assert branded.count("<table") == plain.count("<table")


@pytest.mark.parametrize("goal", GOALS)
def test_custom_branding_replaces_the_identity(results, goal) -> None:
    brand = Branding(
        name="Acme",
        name_accent="Risk",
        tagline="Data review",
        logo=PIXEL_PNG,
        favicon=PIXEL_PNG,
        footer="Prepared by Acme Risk.",
    )
    html = results[goal].render_html(branding=brand)

    assert "prism" not in html.lower()
    assert "— Acme Risk</title>" in html
    assert '<div class="brand-name">Acme<span class="brand-accent">Risk</span>' in html
    assert "Data review" in html
    assert "Prepared by Acme Risk." in html
    pixel = b64encode(PIXEL_PNG).decode("ascii")
    assert f'<img class="brand-mark" src="data:image/png;base64,{pixel}"' in html
    assert f'<link rel="icon" href="data:image/png;base64,{pixel}"' in html


@pytest.mark.parametrize("goal", GOALS)
def test_json_result_never_names_the_product(results, goal) -> None:
    """Findings and evidence travel into other people's reports too."""
    import json

    assert "prism" not in json.dumps(results[goal].to_dict()).lower()


def test_to_html_passes_branding_through(results, tmp_path) -> None:
    for goal in GOALS:
        path = results[goal].to_html(tmp_path / f"{goal}.html", branding=False)
        assert "prism" not in path.read_text(encoding="utf-8").lower(), goal


def test_logo_can_be_a_file_path(results, tmp_path) -> None:
    logo = tmp_path / "logo.png"
    logo.write_bytes(PIXEL_PNG)
    brand = Branding(name="Acme", logo=logo)

    assert brand.logo_bytes() == PIXEL_PNG
    assert brand.logo_data_uri().startswith("data:image/png;base64,")
    assert brand.logo_data_uri() in results["profile"].render_html(branding=brand)


def test_a_missing_custom_logo_is_an_error_not_a_blank(tmp_path) -> None:
    brand = Branding(name="Acme", logo=tmp_path / "nope.png")
    with pytest.raises(FileNotFoundError):
        brand.logo_bytes()


def test_image_formats_are_recognised() -> None:
    assert (
        Branding(logo=b"\xff\xd8\xff\xe0rest")
        .logo_data_uri()
        .startswith("data:image/jpeg;base64,")
    )
    assert (
        Branding(logo=b"<svg xmlns='x'></svg>")
        .logo_data_uri()
        .startswith("data:image/svg+xml;base64,")
    )


def test_branding_helpers() -> None:
    assert Branding.none() == Branding()
    assert Branding.none().title_suffix == ""
    assert Branding.none().display_name is None
    assert Branding.none().logo_bytes() is None
    assert not Branding.none().has_masthead
    assert Branding.prism().title_suffix == " — Prism EDA"
    assert Branding.prism().logo_bytes().startswith(b"\x89PNG")
    assert Branding(name="Acme").title_suffix == " — Acme"


def test_resolve_branding() -> None:
    custom = Branding(name="Acme")
    assert resolve_branding(None) == pe.get_default_branding()
    assert resolve_branding(True) == Branding.prism()
    assert resolve_branding(False) == Branding.none()
    assert resolve_branding(custom) is custom
    with pytest.raises(TypeError):
        resolve_branding("acme")  # type: ignore[arg-type]


def test_default_branding_can_be_switched_off_for_the_process(results) -> None:
    assert pe.get_default_branding() == Branding.prism()
    previous = pe.set_default_branding(False)
    try:
        assert previous == Branding.prism()
        assert pe.get_default_branding() == Branding.none()
        for goal in GOALS:
            assert "prism" not in results[goal].render_html().lower(), goal
        # An explicit request still wins over the process default.
        assert "Prism" in results["profile"].render_html(branding=True)

        custom = Branding(name="Acme")
        pe.set_default_branding(custom)
        assert "— Acme</title>" in results["profile"].render_html()
    finally:
        pe.set_default_branding(previous)
    assert pe.get_default_branding() == Branding.prism()
    assert "Prism" in results["profile"].render_html()


def test_set_default_branding_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        pe.set_default_branding("acme")  # type: ignore[arg-type]
    assert pe.get_default_branding() == Branding.prism()


def test_feature_plan_report_embeds_the_packaged_logo(results) -> None:
    """It asked for the assets by the wrong path and silently had no logo."""
    html = results["feature_plan"].render_html()

    assert '<img class="brand-mark" src="data:image/png;base64,' in html
    assert '<link rel="icon" href="data:image/png;base64,' in html


def test_comparison_report_renders(results) -> None:
    """The template it loads did not exist, so this used to raise."""
    result = results["comparison"]
    html = result.render_html()

    assert html.startswith("<!doctype html>")
    assert "Dataset comparison" in html
    charts = [e for e in result.evidence if e.kind == "column_comparison_chart"]
    assert charts
    assert html.count('class="col-card"') == len(charts)
    assert html.count("<svg") == len(charts)
    for item in charts:
        assert item.scope.columns[0] in html
    assert "https://" not in html
    assert "http://" not in html.replace("http://www.w3.org", "")


def test_comparison_report_names_both_sides_and_shows_warnings() -> None:
    result = pe.compare_datasets(
        {"last_year": pd.DataFrame({"amount": [1.0, 2.0, 3.0], "kind": list("aab")})},
        {"this_year": pd.DataFrame({"amount": [2.0, 3.0, 9.0], "kind": list("abb")})},
    )
    html = result.render_html()

    assert "last_year vs this_year" in html
    assert "table_name_mismatch" in html
    assert 'id="section-warnings"' in html


def test_comparison_report_with_nothing_to_compare_still_renders() -> None:
    result = pe.compare_datasets(
        {"a": pd.DataFrame({"x": [1]}), "b": pd.DataFrame({"x": [1]})},
        {"c": pd.DataFrame({"x": [1]}), "d": pd.DataFrame({"x": [1]})},
    )
    html = result.render_html()

    assert "no_common_tables" in html
    assert 'class="col-card"' not in html
