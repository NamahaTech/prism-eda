"""PDF export.

Fonts are embedded as subsets, so text in the finished file cannot be found
with a byte search. Body text is therefore checked on the flowables
(``build_story``), and the finished file is checked for what *is* stored
plainly: the header, the document information, and the embedded images.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("matplotlib")

from reportlab.platypus import Image, KeepTogether, Paragraph, Table  # noqa: E402

import prism_eda as pe  # noqa: E402
from examples.bank_data import bank_tables  # noqa: E402
from examples.sample_data import load_sample  # noqa: E402
from prism_eda.branding import Branding  # noqa: E402
from prism_eda.reporting.pdf import build_story  # noqa: E402

HINTS = {"label": "accounts.is_flagged", "amount": "transactions.amount"}


def _png() -> bytes:
    import io

    from PIL import Image as PillowImage

    buffer = io.BytesIO()
    PillowImage.new("RGB", (40, 20), "#2a78d6").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def exploration():
    # Large enough for the labelled-rows section (it needs 30 labelled rows).
    return pe.explore(bank_tables(), hints=HINTS)


@pytest.fixture(scope="module")
def profile():
    return pe.profile({"orders": load_sample()["orders"]})


def _walk(flowables: Any) -> Any:
    for flowable in flowables:
        if isinstance(flowable, (list, tuple)):
            yield from _walk(flowable)
            continue
        yield flowable
        if isinstance(flowable, Table):
            yield from _walk(flowable._cellvalues)
        elif isinstance(flowable, KeepTogether):
            yield from _walk(flowable._content)


def _text(story: Any) -> str:
    return "\n".join(
        item.getPlainText() for item in _walk(story) if isinstance(item, Paragraph)
    )


def _images(story: Any) -> int:
    return sum(1 for item in _walk(story) if isinstance(item, Image))


def test_exploration_pdf_renders(exploration) -> None:
    pdf = exploration.render_pdf()
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 50_000


def test_exploration_pdf_has_every_section(exploration) -> None:
    story, title, _brand, _stamp = build_story(exploration, branding=False)
    text = _text(story)
    assert title == "Data exploration report"
    for heading in (
        "Contents",
        "Summary",
        "Data inventory",
        "How the data connects",
        "Links between records",
        "transactions: sender → receiver",
        "Labelled rows",
        "Findings",
        "Table by table",
        "Method and limits",
    ):
        assert heading in text, heading
    # The numbers a reader came for are on the page, not only in a chart.
    assert "devices.acc" in text and "accounts.acct_no" in text
    assert "Gap in transactions.ts" in text
    assert _images(story) > 10


def test_unbranded_pdf_never_names_the_library(exploration, profile) -> None:
    for result in (exploration, profile):
        story, _title, brand, _stamp = build_story(result, branding=False)
        assert "prism" not in _text(story).lower()
        assert brand.logo_bytes() is None and brand.footer is None
        pdf = result.render_pdf(branding=False)
        # Document information is stored as plain strings.
        assert b"Prism" not in pdf and b"prism" not in pdf


def test_default_pdf_carries_the_library_identity(exploration) -> None:
    story, _title, brand, _stamp = build_story(exploration)
    assert "Prism EDA" in _text(story)
    assert brand.footer == "Generated locally by Prism EDA."
    assert b"Prism EDA" in exploration.render_pdf()
    # One more image than the unbranded story: the logo on the cover.
    plain, *_ = build_story(exploration, branding=False)
    assert _images(story) == _images(plain) + 1


def test_custom_branding_and_titles(exploration) -> None:
    brand = Branding(
        name="Acme", name_accent="Risk", logo=_png(), footer="Prepared by Acme."
    )
    story, title, _brand, _stamp = build_story(
        exploration, branding=brand, title="Core banking review", subtitle="Phase one"
    )
    text = _text(story)
    assert title == "Core banking review"
    assert "Acme Risk" in text and "Core banking review" in text
    assert "Phase one" in text
    assert "prism" not in text.lower()
    pdf = exploration.render_pdf(branding=brand, title="Core banking review")
    assert b"Acme Risk" in pdf and b"Core banking review" in pdf
    assert b"Prism" not in pdf


def test_detailed_tables_zero_keeps_statistics_drops_charts(exploration) -> None:
    full, *_ = build_story(exploration, branding=False)
    lean, *_ = build_story(exploration, branding=False, detailed_tables=0)
    assert _images(lean) < _images(full)
    assert "Range or most frequent values" in _text(lean)
    assert "column statistics only" in _text(lean)
    assert exploration.render_pdf(detailed_tables=0).startswith(b"%PDF-")


def test_pdf_for_other_recipes(profile) -> None:
    assert profile.render_pdf().startswith(b"%PDF-")
    story, title, *_ = build_story(profile, branding=False)
    assert title == "Data profile"
    assert "orders" in _text(story)
    dataset = pe.load({"orders": load_sample()["orders"]})
    anomalies = dataset.anomaly_detection(table="orders")
    assert anomalies.render_pdf(branding=False).startswith(b"%PDF-")
    story, title, *_ = build_story(anomalies, branding=False)
    assert title == "Anomaly detection report"
    assert "Findings" in _text(story)


def test_pdf_with_no_tables(exploration) -> None:
    empty = pe.explore(bank_tables(customers=300, transactions=6_000), include="none")
    assert empty.catalog.table_count == 0
    assert empty.render_pdf(branding=False).startswith(b"%PDF-")


def test_unreadable_logo_does_not_break_the_report(exploration) -> None:
    brand = Branding(name="Acme", logo=b"\x89PNG\r\n\x1a\n not really a png")
    story, *_ = build_story(exploration, branding=brand)
    plain, *_ = build_story(exploration, branding=False)
    assert _images(story) == _images(plain)
    assert exploration.render_pdf(branding=brand).startswith(b"%PDF-")


def test_to_pdf_writes_a_file(exploration, tmp_path) -> None:
    path = exploration.to_pdf(tmp_path / "report.pdf", branding=False)
    assert path.read_bytes().startswith(b"%PDF-")


def test_pdf_is_stable_for_the_same_result(exploration) -> None:
    first, *_ = build_story(exploration, branding=False)
    second, *_ = build_story(exploration, branding=False)
    assert _text(first) == _text(second)
