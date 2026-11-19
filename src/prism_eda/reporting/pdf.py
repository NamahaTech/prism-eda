"""PDF report renderer (the ``pdf`` extra: reportlab + matplotlib).

The PDF is built for reading away from a screen: a cover, a contents page with
page numbers, a one-page summary, then sections that go from the whole dataset
down to single tables. Charts are static; every figure a chart shows is also in
a table or in the JSON export.

The full layout is written for exploration and baseline-profile results. Other
recipes render their summary, findings and per-table column statistics.
"""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape

from prism_eda.branding import Branding, BrandingLike, resolve_branding
from prism_eda.catalog.models import ColumnCatalog, TableCatalog
from prism_eda.evidence.models import Evidence, Finding, split_findings
from prism_eda.exceptions import PrismEDAError
from prism_eda.reporting._columns import column_detail, column_flags, shared_columns
from prism_eda.results import AnalysisResult


class PdfSupportMissing(PrismEDAError):
    """The optional PDF dependencies are not installed."""


try:  # pragma: no cover - exercised only without the extra
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate,
        Flowable,
        Frame,
        Image,
        KeepTogether,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents

    from prism_eda.reporting import pdf_charts as charts
except ImportError as error:  # pragma: no cover
    raise PdfSupportMissing(
        "PDF export needs the 'pdf' extra: pip install 'prism-eda[pdf]'"
    ) from error

PAGE = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE[0] - 2 * MARGIN
#: Tables beyond this many (largest first) get their column statistics but no
#: charts, which keeps a 300-table report to a size that can be emailed.
DEFAULT_DETAILED_TABLES = 80

INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
INK_3 = colors.HexColor("#8a8983")
RULE = colors.HexColor("#dcdbd6")
BAND = colors.HexColor("#f6f5f2")
ACCENT = colors.HexColor("#2a78d6")
SEVERITY = {
    "critical": colors.HexColor("#b3261e"),
    "high": colors.HexColor("#d03b3b"),
    "medium": colors.HexColor("#c77700"),
    "low": colors.HexColor("#52514e"),
    "info": colors.HexColor("#2a78d6"),
}

_FONT = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"


def _register_fonts() -> None:
    """Use matplotlib's bundled DejaVu Sans so text and charts share a face
    and non-Latin-1 characters in data values still render."""
    global _FONT, _FONT_BOLD
    if _FONT != "Helvetica":
        return
    try:
        import matplotlib

        base = f"{matplotlib.get_data_path()}/fonts/ttf"
        pdfmetrics.registerFont(TTFont("ReportSans", f"{base}/DejaVuSans.ttf"))
        pdfmetrics.registerFont(
            TTFont("ReportSans-Bold", f"{base}/DejaVuSans-Bold.ttf")
        )
        pdfmetrics.registerFontFamily(
            "ReportSans",
            normal="ReportSans",
            bold="ReportSans-Bold",
            italic="ReportSans",
            boldItalic="ReportSans-Bold",
        )
        _FONT, _FONT_BOLD = "ReportSans", "ReportSans-Bold"
    except Exception:  # noqa: BLE001 - fall back to the built-in face
        pass


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "base", fontName=_FONT, fontSize=8.6, leading=12.2, textColor=INK
    )
    return {
        "body": base,
        "small": ParagraphStyle(
            "small", parent=base, fontSize=7.4, leading=10, textColor=INK_2
        ),
        "note": ParagraphStyle(
            "note", parent=base, fontSize=7.4, leading=10, textColor=INK_3
        ),
        "cell": ParagraphStyle("cell", parent=base, fontSize=7.2, leading=9.2),
        "cell_right": ParagraphStyle(
            "cell_right", parent=base, fontSize=7.2, leading=9.2, alignment=TA_RIGHT
        ),
        "head": ParagraphStyle(
            "head",
            parent=base,
            fontSize=6.8,
            leading=8.6,
            fontName=_FONT_BOLD,
            textColor=INK_2,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base,
            fontName=_FONT_BOLD,
            fontSize=17,
            leading=21,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base,
            fontName=_FONT_BOLD,
            fontSize=11.5,
            leading=15,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base,
            fontName=_FONT_BOLD,
            fontSize=9,
            leading=12,
            spaceBefore=7,
            spaceAfter=2,
            textColor=INK_2,
        ),
        "lead": ParagraphStyle(
            "lead", parent=base, fontSize=9.6, leading=14, textColor=INK_2, spaceAfter=6
        ),
        "cover_title": ParagraphStyle(
            "cover_title", parent=base, fontName=_FONT_BOLD, fontSize=30, leading=35
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base, fontSize=12, leading=17, textColor=INK_2
        ),
        "cover_brand": ParagraphStyle(
            "cover_brand", parent=base, fontName=_FONT_BOLD, fontSize=12, leading=15
        ),
        "kpi_value": ParagraphStyle(
            "kpi_value", parent=base, fontName=_FONT_BOLD, fontSize=15, leading=18
        ),
        "kpi_label": ParagraphStyle(
            "kpi_label", parent=base, fontSize=7, leading=9, textColor=INK_2
        ),
        "toc1": ParagraphStyle(
            "toc1", parent=base, fontName=_FONT_BOLD, fontSize=9.4, leading=16
        ),
        "toc2": ParagraphStyle(
            "toc2",
            parent=base,
            fontSize=8.2,
            leading=12.5,
            leftIndent=12,
            textColor=INK_2,
        ),
        "finding_title": ParagraphStyle(
            "finding_title", parent=base, fontName=_FONT_BOLD, fontSize=8.8, leading=12
        ),
    }


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def _n(value: Any) -> str:
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


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    share = float(value)
    if 0 < share < 0.001:
        return "<0.1%"
    if 0.999 < share < 1:
        return ">99.9%"
    return f"{share:.{digits}%}"


def _bytes(value: Any) -> str:
    if not value:
        return "–"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return (
                f"{size:,.0f} {unit}" if unit in {"B", "KB"} else f"{size:,.1f} {unit}"
            )
        size /= 1024
    return "–"


def _clip(value: Any, limit: int = 60) -> str:
    text = " ".join(str(value).split()) if value is not None else ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(value: Any) -> str:
    """One value of an example row, short enough for a narrow column."""
    if value is None:
        return "–"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)}"
    text = str(value)
    # ISO timestamps: drop the "T" and the seconds so they fit on one line.
    if len(text) >= 19 and text[4] == "-" and text[10] == "T":
        text = text[:16].replace("T", " ")
        if text.endswith(" 00:00"):
            text = text[:10]
    return _clip(text, 20)


def _date(value: Any) -> str:
    return str(value)[:10] if value else "–"


def _date_time(value: Any) -> str:
    text = str(value) if value else "–"
    return text[:16].replace("T", " ")


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------


class _Document(BaseDocTemplate):
    """A document that records headings for the contents page and outline."""

    def __init__(
        self, buffer: io.BytesIO, title: str, brand: Branding, stamp: str
    ) -> None:
        super().__init__(
            buffer,
            pagesize=PAGE,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN + 6 * mm,
            bottomMargin=MARGIN + 4 * mm,
            title=title,
            author=brand.display_name or "",
            creator=brand.display_name or "",
            subject=title,
        )
        self._title = title
        self._brand = brand
        self._stamp = stamp
        self._heading_count = 0
        frame = Frame(
            MARGIN,
            MARGIN + 4 * mm,
            CONTENT_WIDTH,
            PAGE[1] - 2 * MARGIN - 10 * mm,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="body",
        )
        cover = Frame(
            MARGIN,
            MARGIN,
            CONTENT_WIDTH,
            PAGE[1] - 2 * MARGIN,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="cover",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[cover]),
                PageTemplate(id="body", frames=[frame], onPage=self._decorate),
            ]
        )

    def _decorate(self, canvas: Any, _document: Any) -> None:
        canvas.saveState()
        canvas.setFont(_FONT, 7)
        canvas.setFillColor(INK_3)
        top = PAGE[1] - MARGIN + 1 * mm
        canvas.drawString(MARGIN, top, _clip(self._title, 90))
        canvas.drawRightString(PAGE[0] - MARGIN, top, self._stamp)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, top - 2.2 * mm, PAGE[0] - MARGIN, top - 2.2 * mm)
        bottom = MARGIN - 3 * mm
        if self._brand.footer:
            canvas.drawString(MARGIN, bottom, _clip(self._brand.footer, 110))
        canvas.drawRightString(
            PAGE[0] - MARGIN, bottom, f"Page {canvas.getPageNumber()}"
        )
        canvas.restoreState()

    def handle_documentBegin(self) -> None:
        # The contents page is resolved over several passes; bookmark names
        # must be the same on each or it never settles.
        self._heading_count = 0
        super().handle_documentBegin()

    def afterFlowable(self, flowable: Flowable) -> None:
        level = getattr(flowable, "_toc_level", None)
        if level is None:
            return
        self._heading_count += 1
        key = f"h{self._heading_count}"
        text = flowable._toc_text  # type: ignore[attr-defined]
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, text, self.page, key))


class _Rule(Flowable):
    def __init__(self, width: float, color: Any = RULE, thickness: float = 0.6) -> None:
        super().__init__()
        self.width, self.height = width, 3
        self._color, self._thickness = color, thickness

    def draw(self) -> None:
        self.canv.setStrokeColor(self._color)
        self.canv.setLineWidth(self._thickness)
        self.canv.line(0, 1.5, self.width, 1.5)


class _Builder:
    """Accumulates the report's flowables."""

    def __init__(
        self, result: AnalysisResult, brand: Branding, detailed_tables: int
    ) -> None:
        self.result = result
        self.brand = brand
        self.detailed_tables = detailed_tables
        self.s = _styles()
        self.story: list[Any] = []
        self.by_kind: dict[str, list[Evidence]] = defaultdict(list)
        self.by_table: dict[str, dict[str, list[Evidence]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for item in result.evidence:
            self.by_kind[item.kind].append(item)
            if item.scope.table:
                self.by_table[item.scope.table][item.kind].append(item)
        self.evidence_by_id = {item.id: item for item in result.evidence}
        self.shared = shared_columns(result.evidence)

    # -- primitives --------------------------------------------------------

    def p(self, text: str, style: str = "body") -> Paragraph:
        return Paragraph(text, self.s[style])

    def text(self, value: Any, style: str = "body") -> Paragraph:
        return Paragraph(escape(str(value)), self.s[style])

    def heading(self, text: str, level: int = 0) -> Paragraph:
        paragraph = Paragraph(escape(text), self.s["h1" if level == 0 else "h2"])
        paragraph._toc_level = level  # type: ignore[attr-defined]
        paragraph._toc_text = text  # type: ignore[attr-defined]
        return paragraph

    def section(self, title: str, lead: str | None = None) -> None:
        self.story.append(PageBreak())
        self.story.append(self.heading(title, 0))
        self.story.append(_Rule(CONTENT_WIDTH, INK, 1.1))
        self.story.append(Spacer(1, 5))
        if lead:
            self.story.append(self.p(lead, "lead"))

    def image(self, png: bytes | None, max_width: float = CONTENT_WIDTH) -> Any | None:
        if not png:
            return None
        reader = ImageReader(io.BytesIO(png))
        width, height = reader.getSize()
        scale = min(max_width / width, (72 / charts.DPI))
        flowable = Image(io.BytesIO(png), width=width * scale, height=height * scale)
        flowable.hAlign = "LEFT"
        return flowable

    def add_chart(
        self, png: bytes | None, caption: str | None = None, title: str | None = None
    ) -> None:
        flowable = self.image(png)
        if flowable is None:
            return
        block: list[Any] = []
        if title:
            block.append(self.text(title, "h3"))
        block.append(flowable)
        if caption:
            block.append(self.p(caption, "note"))
        block.append(Spacer(1, 6))
        self.story.append(KeepTogether(block))

    def table(
        self,
        header: Sequence[str],
        rows: Sequence[Sequence[Any]],
        widths: Sequence[float],
        *,
        right: Sequence[int] = (),
        repeat: bool = True,
    ) -> Table:
        total = sum(widths)
        col_widths = [CONTENT_WIDTH * width / total for width in widths]
        data: list[list[Any]] = [[self.text(cell, "head") for cell in header]]
        for row in rows:
            cells = []
            for index, cell in enumerate(row):
                if isinstance(cell, Flowable):
                    cells.append(cell)
                else:
                    style = "cell_right" if index in right else "cell"
                    cells.append(
                        Paragraph(
                            escape(str(cell)) if cell is not None else "–",
                            self.s[style],
                        )
                    )
            data.append(cells)
        table = Table(data, colWidths=col_widths, repeatRows=1 if repeat else 0)
        commands: list[tuple[Any, ...]] = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK_2),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ]
        for index in range(2, len(data), 2):
            commands.append(("BACKGROUND", (0, index), (-1, index), BAND))
        table.setStyle(TableStyle(commands))
        table.hAlign = "LEFT"
        return table

    def kpis(self, items: Sequence[tuple[str, str]], per_row: int = 4) -> None:
        cells = [
            [self.text(value, "kpi_value"), self.text(label, "kpi_label")]
            for label, value in items
        ]
        for start in range(0, len(cells), per_row):
            row = cells[start : start + per_row]
            row += [""] * (per_row - len(row))  # type: ignore[list-item]
            table = Table([row], colWidths=[CONTENT_WIDTH / per_row] * per_row)
            table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ("BACKGROUND", (0, 0), (-1, -1), BAND),
                        ("LINEBEFORE", (1, 0), (-1, -1), 2, colors.white),
                    ]
                )
            )
            self.story.append(table)
            self.story.append(Spacer(1, 3))
        self.story.append(Spacer(1, 5))

    def finding(self, finding: Finding) -> None:
        color = SEVERITY.get(finding.severity, INK_2)
        label = finding.severity.upper()
        body = [
            Paragraph(
                f'<font color="{color.hexval()}" size="6.6"><b>{label}</b></font>'
                f"&nbsp;&nbsp;{escape(finding.title)}",
                self.s["finding_title"],
            ),
            self.text(finding.summary),
        ]
        if finding.recommendation:
            body.append(
                self.p(f"<b>What to do:</b> {escape(finding.recommendation)}", "small")
            )
        table = Table([[body]], colWidths=[CONTENT_WIDTH])
        table.setStyle(
            TableStyle(
                [
                    ("LINEBEFORE", (0, 0), (0, -1), 2.2, color),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        self.story.append(KeepTogether([table, Spacer(1, 6)]))

    # -- sections ----------------------------------------------------------

    def cover(self, title: str, subtitle: str | None, stamp: str) -> None:
        story = self.story
        story.append(Spacer(1, 38 * mm))
        logo = self.brand.logo_bytes()
        if logo:
            try:
                # Decode the whole image now: a logo that only fails when the
                # page is drawn would take the report down with it.
                from PIL import Image as PillowImage

                PillowImage.open(io.BytesIO(logo)).load()
                reader = ImageReader(io.BytesIO(logo))
                width, height = reader.getSize()
                target = 16 * mm
                image = Image(
                    io.BytesIO(logo), width=width * target / height, height=target
                )
                image.hAlign = "LEFT"
                story.append(image)
                story.append(Spacer(1, 4 * mm))
            except Exception:  # noqa: BLE001 - an unreadable logo is not fatal
                pass
        if self.brand.display_name:
            story.append(self.text(self.brand.display_name, "cover_brand"))
            if self.brand.tagline:
                story.append(self.text(self.brand.tagline, "small"))
            story.append(Spacer(1, 10 * mm))
        story.append(self.text(title, "cover_title"))
        story.append(Spacer(1, 4 * mm))
        if subtitle:
            story.append(self.text(subtitle, "cover_sub"))
        story.append(Spacer(1, 10 * mm))
        story.append(_Rule(CONTENT_WIDTH, INK, 1.2))
        story.append(Spacer(1, 6 * mm))
        self.kpis(self._headline_numbers(), per_row=3)
        story.append(Spacer(1, 6 * mm))
        story.append(self.text(f"Generated {stamp}", "small"))
        story.append(NextPageTemplate("body"))
        story.append(PageBreak())

    def _headline_numbers(self) -> list[tuple[str, str]]:
        result = self.result
        catalog = result.catalog
        meta = result.metadata
        items = [
            ("Tables", _n(meta.get("table_count", catalog.table_count))),
            ("Rows", charts.compact(catalog.row_count)),
            ("Columns", _n(catalog.column_count)),
        ]
        if result.goal == "exploration":
            items += [
                ("Relationships found", _n(meta.get("relationship_count", 0))),
                ("Record-to-record links", _n(meta.get("link_count", 0))),
                ("Data-quality issues", _n(meta.get("issue_count", 0))),
            ]
        else:
            issues, observations = split_findings(result.findings)
            items += [
                ("Issues", _n(len(issues))),
                ("Observations", _n(len(observations))),
            ]
        return items

    def contents(self) -> None:
        self.story.append(self.text("Contents", "h1"))
        self.story.append(_Rule(CONTENT_WIDTH, INK, 1.1))
        self.story.append(Spacer(1, 6))
        toc = TableOfContents()
        toc.levelStyles = [self.s["toc1"], self.s["toc2"]]
        toc.dotsMinLevel = 0
        self.story.append(toc)

    def summary(self) -> None:
        result = self.result
        self.section("Summary", result.summary)
        self.kpis(self._headline_numbers(), per_row=3)
        issues, observations = split_findings(result.findings)
        top = [f for f in issues if f.severity in {"critical", "high", "medium"}][:8]
        if top:
            self.story.append(self.text("What needs attention first", "h2"))
            for finding in top:
                self.finding(finding)
        notable = observations[:6]
        if notable:
            self.story.append(self.text("What the data can support", "h2"))
            for finding in notable:
                self.finding(finding)
        self.story.append(self.text("How to read the numbers", "h2"))
        self.story.append(
            self.p(
                "Row counts, missing values, distinct counts, ranges, most frequent "
                "values and rows over time are computed over <b>every row</b> of each "
                "table. Distribution charts, correlations, format checks and example "
                "rows come from a <b>deterministic row sample</b> of each table and are "
                "labelled as such. Relationship coverage is measured by looking up a "
                "sample of a column's distinct values in the other column; each lookup "
                "is exact, and the interval shown reflects only the choice of which "
                "values were looked up. A distinct count marked ≈ is the database's "
                "estimate."
            )
        )

    def inventory(self) -> None:
        items = self.by_kind.get("explore_inventory")
        if not items:
            return
        value = items[0].value
        tables = value["tables"]
        self.section(
            "Data inventory",
            f"{len(tables)} table(s) across {len(value['sources'])} source(s), "
            f"{_n(value['total_rows'])} rows in total.",
        )
        ranges = {}
        for item in self.by_kind.get("explore_time_coverage", []):
            for row in item.value["tables"]:
                ranges[row["table"]] = row
        ordered = sorted(tables, key=lambda row: -row["rows"])
        rows = []
        for row in ordered:
            span = ranges.get(row["table"])
            status = {
                "profiled": "",
                "partial": "Partly profiled",
                "empty": "Empty",
                "failed": "Not readable",
            }[row["status"]]
            rows.append(
                [
                    row["table"],
                    row["source"],
                    _n(row["rows"]) + ("" if row["rows_exact"] else " ≈"),
                    _n(row["columns"]),
                    _bytes(row["bytes"]),
                    f"{_date(span['min'])} → {_date(span['max'])}" if span else "–",
                    status,
                ]
            )
        self.story.append(
            self.table(
                [
                    "Table",
                    "Source",
                    "Rows",
                    "Columns",
                    "Size",
                    "Period covered",
                    "Note",
                ],
                rows,
                [28, 12, 13, 11, 11, 24, 13],
                right=(2, 3, 4),
            )
        )
        self.story.append(Spacer(1, 8))
        populated = [row for row in ordered if row["rows"] > 0][:30]
        if len(populated) >= 2:
            largest, smallest = populated[0]["rows"], populated[-1]["rows"]
            self.add_chart(
                charts.horizontal_bars(
                    [row["table"] for row in populated],
                    [row["rows"] for row in populated],
                    log=largest > 200 * max(smallest, 1),
                ),
                title="Rows per table",
                caption="Logarithmic scale."
                if largest > 200 * max(smallest, 1)
                else None,
            )
        for item in self.by_kind.get("explore_time_coverage", []):
            self.add_chart(
                charts.coverage_chart(item.value["tables"]),
                title="Period covered by each table",
                caption="Each bar runs from the earliest to the latest value of the table's busiest date column.",
            )

    def relationships(self) -> None:
        items = self.by_kind.get("explore_relationship", [])
        domains = self.by_kind.get("explore_domain", [])
        if self.result.goal != "exploration":
            return
        compared = self.result.metadata.get("identifier_columns", 0)
        self.section(
            "How the data connects",
            f"{compared} identifier-like column(s) were compared with one another by "
            "their values, not their names. Columns that hold the same identifiers are "
            "listed below, whether they sit in one table, in different tables, or in "
            "different databases.",
        )
        if not items:
            self.story.append(
                self.text("No two columns were found to share identifiers.")
            )
            return
        for artifact in self.result.artifacts:
            if artifact.kind == "relationship_graph":
                self.add_chart(
                    charts.relationship_graph(
                        artifact.data["nodes"], artifact.data["edges"]
                    ),
                    title=artifact.title,
                )
        confirmed = [item.value for item in items if item.value["tier"] != "possible"]
        primary = [value for value in confirmed if value.get("primary", True)]
        secondary = [value for value in confirmed if not value.get("primary", True)]
        possible = [item.value for item in items if item.value["tier"] == "possible"]

        def rows_for(values: Sequence[dict[str, Any]]) -> list[list[Any]]:
            rows = []
            for value in values:
                low, high = value["coverage_interval"]
                kind = {
                    "many_to_one": "Reference",
                    "one_to_one": "Same key",
                    "shared_values": "Shared values",
                }[value["kind"]]
                if value["same_table"]:
                    kind = "Link within table"
                notes = []
                if value["cross_source"]:
                    notes.append("across databases")
                if value["tier"] == "likely":
                    notes.append("short numeric values; supported by names")
                if value.get("format_note"):
                    notes.append(
                        "format differs"
                        if "cast" not in value["format_note"]
                        else "type differs"
                    )
                rows.append(
                    [
                        f"{value['child']['table']}.{value['child']['column']}",
                        f"{value['parent']['table']}.{value['parent']['column']}",
                        kind,
                        f"{_pct(value['coverage'])}  ({_pct(low, 0)}–{_pct(high, 0)})",
                        _pct(value["row_coverage"]),
                        _pct(value["reverse_coverage"]),
                        "; ".join(notes),
                    ]
                )
            return rows

        header = [
            "Column",
            "Also found in",
            "Type",
            "Values covered (95% interval)",
            "Rows covered",
            "Reverse",
            "Note",
        ]
        widths = [24, 24, 12, 16, 9, 10, 14]
        self.story.append(self.text("Relationships", "h2"))
        self.story.append(
            self.p(
                "<b>Values covered</b> is the share of the first column's distinct values "
                "that exist in the second; <b>rows covered</b> weights that by how often "
                "each value occurs; <b>reverse</b> is the share of the second column's "
                "values that appear in the first.",
                "small",
            )
        )
        self.story.append(Spacer(1, 3))
        self.story.append(
            self.table(header, rows_for(primary), widths, right=(3, 4, 5))
        )
        if secondary:
            self.story.append(
                self.text("Further pairs within the same identifier groups", "h3")
            )
            self.story.append(
                self.table(header, rows_for(secondary[:60]), widths, right=(3, 4, 5))
            )
            if len(secondary) > 60:
                self.story.append(
                    self.text(
                        f"{len(secondary) - 60} more pairs are in the JSON export.",
                        "note",
                    )
                )
        if domains:
            self.story.append(self.text("Identifier groups", "h2"))
            self.story.append(
                self.p(
                    "Columns connected by shared values, grouped. The reference is the "
                    "column in which every value appears once.",
                    "small",
                )
            )
            self.story.append(Spacer(1, 3))
            rows = []
            for item in domains:
                value = item.value
                reference = value["reference"]
                rows.append(
                    [
                        value["id"],
                        f"{reference['table']}.{reference['column']}"
                        if reference
                        else "None (no column is unique)",
                        ", ".join(
                            f"{m['table']}.{m['column']}" for m in value["members"]
                        ),
                        ", ".join(
                            _clip(example, 18)
                            for example in (value["members"][0]["examples"][:2])
                        ),
                    ]
                )
            self.story.append(
                self.table(
                    [
                        "Group",
                        "Reference column",
                        "Columns in the group",
                        "Example values",
                    ],
                    rows,
                    [7, 25, 48, 20],
                )
            )
        if possible:
            self.story.append(self.text("Possible relationships (weak evidence)", "h2"))
            self.story.append(
                self.p(
                    "These columns hold short numeric values that are fully contained in "
                    "a unique column. Small integers overlap by coincidence, so values "
                    "alone cannot confirm these.",
                    "small",
                )
            )
            self.story.append(Spacer(1, 3))
            self.story.append(
                self.table(header, rows_for(possible[:25]), widths, right=(3, 4, 5))
            )
            if len(possible) > 25:
                self.story.append(
                    self.text(
                        f"{len(possible) - 25} more are in the JSON export.", "note"
                    )
                )

    def links(self) -> None:
        items = self.by_kind.get("explore_link", [])
        if not items:
            return
        self.section(
            "Links between records",
            "Where two columns of one table hold the same kind of identifier, every row "
            "connects two records and the table forms a network. This section measures "
            "how much of that network the data actually holds.",
        )
        for item in items:
            value = item.value
            name = f"{value['table']}: {value['origin']} → {value['destination']}"
            self.story.append(self.heading(name, 1))
            known = value.get("destination_known") or {}
            reach = value.get("reach") or {}
            hops = {hop["hop"]: hop for hop in reach.get("hops", [])}
            self.kpis(
                [
                    (
                        "Rows with both ends",
                        charts.compact(value.get("linked_rows", value["rows"])),
                    ),
                    (
                        f"Distinct {value['origin']}",
                        charts.compact(value["distinct_origins"]),
                    ),
                    (
                        f"Distinct {value['destination']}",
                        charts.compact(value["distinct_destinations"]),
                    ),
                    (
                        "Distinct pairs",
                        charts.compact(value["distinct_pairs"])
                        if value.get("distinct_pairs")
                        else "–",
                    ),
                    (
                        f"{value['destination']} values we hold a record for",
                        _pct(known.get("coverage")),
                    ),
                    (
                        f"{value['destination']} that also act as {value['origin']}",
                        _pct(value.get("destinations_also_origins")),
                    ),
                    (
                        "Reach a 2-step connection",
                        _pct(hops[2]["seeds_reaching"], 0) if 2 in hops else "–",
                    ),
                    (
                        "Reach a 3-step connection",
                        _pct(hops[3]["seeds_reaching"], 0) if 3 in hops else "–",
                    ),
                ]
            )
            if known.get("coverage") is not None:
                reference = value.get("reference") or "the reference list"
                self.add_chart(
                    charts.share_bar(
                        [
                            ("In " + reference, known["coverage"]),
                            ("Not in the data", 1 - known["coverage"]),
                        ]
                    ),
                    title=f"Distinct {value['destination']} values",
                )
                if known.get("row_coverage") is not None:
                    self.add_chart(
                        charts.share_bar(
                            [
                                ("In " + reference, known["row_coverage"]),
                                ("Not in the data", 1 - known["row_coverage"]),
                            ]
                        ),
                        title=f"Rows, by whether their {value['destination']} is known",
                        caption="A few well-known counterparties can carry most of the rows, so the two bars often differ.",
                    )
            out_degree, in_degree = (
                value.get("out_degree") or {},
                value.get("in_degree") or {},
            )
            self.add_chart(
                charts.bucket_bars(
                    [
                        (
                            f"Distinct {value['destination']} per {value['origin']}",
                            out_degree.get("histogram", []),
                        ),
                        (
                            f"Distinct {value['origin']} per {value['destination']}",
                            in_degree.get("histogram", []),
                        ),
                    ]
                ),
                title="How many counterparties a record has",
                caption="Share of sampled records in each band. Measured on sampled records with all of their rows.",
            )
            rows = []
            for label, stats in (
                (f"{value['destination']} per {value['origin']}", out_degree),
                (f"{value['origin']} per {value['destination']}", in_degree),
            ):
                if stats.get("count"):
                    rows.append(
                        [
                            label,
                            _n(stats["count"]),
                            _n(stats["median"]),
                            f"{stats['mean']:.1f}",
                            _n(stats["p90"]),
                            _n(stats["p99"]),
                            _n(stats["max"]),
                            _pct(stats["share_one"]),
                        ]
                    )
            if rows:
                self.story.append(
                    self.table(
                        [
                            "Counterparties",
                            "Records sampled",
                            "Median",
                            "Mean",
                            "90th pct",
                            "99th pct",
                            "Max",
                            "With exactly one",
                        ],
                        rows,
                        [28, 12, 9, 9, 9, 9, 10, 14],
                        right=(1, 2, 3, 4, 5, 6, 7),
                    )
                )
                self.story.append(Spacer(1, 6))
            if hops:
                self.add_chart(
                    charts.reach_chart(reach["hops"]),
                    title=f"Following links outward from {reach['seeds']} sampled {value['origin']} values",
                    caption=(
                        "Later steps are lower bounds: the walk reached its size limit."
                        if reach.get("truncated_hops")
                        else None
                    ),
                )
                facts = []
                if reach.get("continuation_rate") is not None:
                    facts.append(
                        f"{_pct(reach['continuation_rate'])} of first-step counterparties have onward links of their own."
                    )
                if reach.get("reciprocity") is not None:
                    facts.append(
                        f"{_pct(reach['reciprocity'])} of first-step links are returned in the opposite direction."
                    )
                if value.get("self_links"):
                    facts.append(
                        f"{_n(value['self_links'])} rows link a record to itself."
                    )
                if value.get("rows_per_pair"):
                    facts.append(
                        f"Each distinct pair appears in {value['rows_per_pair']:.1f} rows on average."
                    )
                activity = value.get("origin_activity") or {}
                if activity.get("top_1pct_share") is not None:
                    facts.append(
                        f"The busiest 1% of {value['origin']} values account for {_pct(activity['top_1pct_share'])} of rows."
                    )
                if facts:
                    self.story.append(self.text(" ".join(facts)))
            components = value.get("components")
            if components:
                self.story.append(self.text("Connected groups", "h3"))
                if components["method"] == "exact":
                    self.story.append(
                        self.text(
                            f"Ignoring direction, the {_n(components['entities'])} records form "
                            f"{_n(components['components'])} connected group(s). The largest holds "
                            f"{_pct(components['largest_share'])} of all records; "
                            f"{_n(components['isolated_pairs'])} group(s) are isolated pairs."
                        )
                    )
                else:
                    text = (
                        f"Walking in both directions from {components['seeds']} sampled records: "
                        f"{_pct(components['open_share'], 0)} belong to a group that was still growing "
                        f"when the walk stopped"
                    )
                    if components.get("open_at_least"):
                        text += f" (at least {_n(components['open_at_least'])} records)"
                    text += f"; {_pct(components['closed_share'], 0)} sit in a small closed group"
                    if components.get("closed_median"):
                        text += f" (median size {_n(components['closed_median'])})"
                    self.story.append(self.text(text + "."))
            self.story.append(Spacer(1, 8))

        fanouts = self.by_kind.get("explore_fanout", [])
        rows = []
        for item in fanouts:
            for pair in item.value["pairs"]:
                rows.append(
                    [
                        item.value["table"],
                        pair["key"],
                        pair["other"],
                        _n(pair["entities"]),
                        _pct(pair["share_multiple"]),
                        _n(pair["median"]),
                        _n(pair["p99"]),
                        _n(pair["max"]),
                    ]
                )
        if rows:
            self.story.append(self.heading("How identifiers go together", 1))
            self.story.append(
                self.p(
                    "For each value of the first column: how many distinct values of the "
                    "second it appears with. A column expected to be one-to-one that shows "
                    "a large maximum is shared more widely than assumed.",
                    "small",
                )
            )
            self.story.append(Spacer(1, 3))
            self.story.append(
                self.table(
                    [
                        "Table",
                        "For each",
                        "Distinct",
                        "Sampled",
                        "More than one",
                        "Median",
                        "99th pct",
                        "Max",
                    ],
                    rows[:80],
                    [22, 18, 18, 10, 12, 8, 8, 8],
                    right=(3, 4, 5, 6, 7),
                )
            )

    def labelled(self) -> None:
        items = self.by_kind.get("explore_label", [])
        if not items:
            return
        value = items[0].value
        self.section(
            "Labelled rows",
            f"Rows of {value['table']} where {value['column']} is "
            f"{', '.join(value['positive_values'])}, compared with all other rows.",
        )
        self.kpis(
            [
                ("Labelled rows", _n(value.get("labelled_rows"))),
                ("Share of the table", _pct(value.get("labelled_rate"), 2)),
                ("Labelled rows compared", _n(value["labelled_sampled"])),
                ("Other rows compared", _n(value["others_sampled"])),
            ]
        )
        separators = value.get("separators") or []
        if separators:
            self.add_chart(
                charts.horizontal_bars(
                    [entry["column"] for entry in separators],
                    [entry["strength"] for entry in separators],
                    value_format=lambda v: f"{v:.2f}",
                ),
                title="Columns that differ most between labelled rows and the rest",
                caption="0 means identical distributions, 1 means completely separate. A value near 1 usually means the column was filled in because of the label.",
            )
            rows = []
            for entry in separators:
                if entry["type"] == "numeric":
                    detail = f"median {_n(entry['labelled_median'])} labelled vs {_n(entry['other_median'])} others"
                else:
                    detail = f"'{_clip(entry['value'], 24)}': {_pct(entry['labelled_share'], 0)} labelled vs {_pct(entry['other_share'], 0)} others"
                rows.append(
                    [entry["column"], entry["type"], f"{entry['strength']:.2f}", detail]
                )
            self.story.append(
                self.table(
                    ["Column", "Type", "Separation", "Difference"],
                    rows,
                    [24, 12, 12, 52],
                    right=(2,),
                )
            )
        else:
            self.story.append(
                self.text(
                    "No column differs clearly between labelled rows and the rest."
                )
            )
        for link in value.get("links") or []:
            self.story.append(
                self.text(
                    f"Labelled records in {link['table']} ({link['origin']} → {link['destination']})",
                    "h2",
                )
            )
            degree = link.get("labelled_out_degree") or {}
            self.kpis(
                [
                    (
                        f"Appear as {link['origin']}",
                        _pct(link["labelled_as_origin"], 0),
                    ),
                    (
                        f"Appear as {link['destination']}",
                        _pct(link["labelled_as_destination"], 0),
                    ),
                    ("Median counterparties, labelled", _n(degree.get("median"))),
                    (
                        "Median counterparties, everyone",
                        _n(link.get("baseline_out_degree_median")),
                    ),
                ]
            )
            if link.get("partners_also_labelled") is not None:
                note = (
                    ""
                    if link["labelled_set_complete"]
                    else " (a lower bound: not every labelled record was loaded)"
                )
                self.story.append(
                    self.text(
                        f"{_pct(link['partners_also_labelled'])} of their counterparties are themselves labelled{note}."
                    )
                )

    def findings(self) -> None:
        issues, observations = split_findings(self.result.findings)
        if not issues and not observations:
            return
        self.section(
            "Findings",
            f"{len(issues)} data-quality issue(s) and {len(observations)} observation(s), most severe first.",
        )
        if issues:
            self.story.append(self.heading("Data-quality issues", 1))
            for finding in issues:
                self.finding(finding)
        if observations:
            self.story.append(self.heading("Observations", 1))
            for finding in observations:
                self.finding(finding)

    # -- per table ---------------------------------------------------------

    def _is_key(self, table: str, column: ColumnCatalog) -> bool:
        """Unique, or shares its values with another column: an identifier."""
        return (
            "unique_key_candidate" in column.roles
            or (table, column.name) in self.shared
        )

    def _column_detail(self, table: str, column: ColumnCatalog) -> str:
        return column_detail(column, self._is_key(table, column))

    def _column_flags(self, table: str, column: ColumnCatalog) -> str:
        return ", ".join(column_flags(column, (table, column.name) in self.shared))

    def table_section(self, table: TableCatalog, detailed: bool) -> None:
        story = self.story
        evidence = self.by_table.get(table.name, {})
        story.append(self.heading(table.name, 1))
        summaries = evidence.get("explore_table_summary")
        summary = summaries[0] if summaries else None
        facts = [f"{_n(table.row_count)} rows", f"{_n(table.column_count)} columns"]
        if summary is not None:
            value = summary.value
            if not value.get("rows_exact", True):
                facts[0] += " (estimated)"
            if value.get("bytes"):
                facts.append(_bytes(value["bytes"]))
            origin = value.get("source") or ""
            if value.get("schema"):
                origin += f" · {value['schema']}.{value['name']}"
            if value.get("engine"):
                origin += f" · {value['engine']}"
            if origin:
                facts.append(origin)
        elif table.source.location:
            facts.append(str(table.source.location))
        story.append(self.text("  ·  ".join(facts), "small"))
        story.append(Spacer(1, 4))
        if table.row_count == 0:
            story.append(self.text("The table has no rows.", "note"))

        rows = []
        for column in table.columns:
            distinct = "–"
            if column.unique_count is not None:
                approx = "" if column.statistics.get("distinct_exact", True) else "≈ "
                distinct = approx + charts.compact(column.unique_count)
            rows.append(
                [
                    column.name,
                    _clip(column.physical_type, 22),
                    _pct(1 - column.missing_rate) if column.row_count else "–",
                    distinct,
                    self._column_detail(table.name, column),
                    self._column_flags(table.name, column),
                ]
            )
        story.append(
            self.table(
                [
                    "Column",
                    "Type",
                    "Filled",
                    "Distinct",
                    "Range or most frequent values",
                    "Flags",
                ],
                rows,
                [22, 15, 8, 9, 34, 12],
                right=(2, 3),
            )
        )
        story.append(Spacer(1, 6))
        if not detailed or table.row_count == 0:
            return

        for item in evidence.get("explore_timeline", []) + evidence.get(
            "profile_timeline", []
        ):
            value = item.value
            self.add_chart(
                charts.timeline(
                    value["buckets"],
                    gaps=value.get("gaps", ()),
                    granularity=value.get("granularity"),
                ),
                title=f"Rows per {value['granularity']} · {value['column']}",
                caption="Counted over every row."
                if item.kind == "explore_timeline"
                else None,
            )
        for item in evidence.get("explore_time_pattern", []):
            value = item.value
            self.add_chart(
                charts.weekday_hour_heatmap(value["grid"], value["weekdays"]),
                title=f"When rows happen · {value['column']}",
                caption=(
                    f"From a sample of {_n(value['rows'])} rows. {_pct(value['weekend_share'], 0)} fall on a weekend, "
                    f"{_pct(value['night_share'], 0)} between 22:00 and 06:00."
                ),
            )
        for item in evidence.get("explore_amount", []):
            value = item.value
            histogram = value["log10_histogram"]
            repeats = ", ".join(
                f"{_n(entry['value'])} ({_pct(entry['share'])})"
                for entry in value["top_amounts"][:4]
            )
            self.add_chart(
                charts.log_histogram(histogram["counts"], histogram["edges"]),
                title=f"Size of {value['column']}",
                caption=(
                    f"From a sample of {_n(value['rows'])} rows. {_pct(value['whole_number_share'], 0)} are whole numbers, "
                    f"{_pct(value['multiple_of_1000'], 0)} multiples of 1,000. Most repeated: {repeats}."
                ),
            )
        sample_note = None
        keys = {
            column.name for column in table.columns if self._is_key(table.name, column)
        }
        binary = {
            column.name for column in table.columns if (column.unique_count or 0) <= 2
        }
        # A histogram of account numbers, or of a 0/1 flag, is not a picture
        # of anything.
        distributions = [
            item
            for item in evidence.get("profile_distribution", [])
            if item.value["column"] not in keys | binary
        ]
        if distributions:
            basis = distributions[0].metadata
            if basis.get("basis") == "sample":
                sample_note = f"From a sample of {_n(basis.get('sample_rows'))} of {_n(basis.get('table_rows'))} rows."
            panels = [
                {
                    "name": item.value["column"],
                    "counts": item.value["histogram"]["counts"],
                    "edges": item.value["histogram"]["edges"],
                    "median": item.value["box"]["median"],
                    "log": item.value.get("histogram_log_scale", False),
                }
                for item in distributions[:16]
            ]
            clipped = any(
                item.value.get("histogram_clipped") for item in distributions[:16]
            )
            caption = (sample_note or "") + " The line marks the median."
            if clipped:
                caption += " Long-tailed columns are drawn from the 1st to the 99th percentile, with the rest counted into the end bars."
            self.add_chart(
                charts.histogram_grid(panels), title="Numeric columns", caption=caption
            )
        categories = evidence.get("profile_category_frequency", [])
        if categories:
            panels = [
                {
                    "name": item.value["column"],
                    "counts": item.value["counts"],
                    "other_category_count": item.value["other_category_count"],
                }
                for item in categories[:12]
            ]
            self.add_chart(
                charts.category_grid(panels),
                title="Categorical columns",
                caption=sample_note,
            )
        for item in evidence.get("profile_association_matrix", []):
            value = item.value
            pairs = [
                pair
                for pair in value["pairs"]
                if pair["left"] not in keys and pair["right"] not in keys
            ]
            # With nothing even moderately related, the chart is an empty grid.
            if not any(pair["strength"] >= 0.3 for pair in pairs):
                continue
            columns = [name for name in value["columns"] if name not in keys]
            png = charts.association_heatmap(columns, pairs)
            strongest = [
                pair
                for pair in pairs
                if pair["strength"] >= 0.5 and not pair.get("identical_columns")
            ][:5]
            caption = "0 is unrelated, 1 is fully determined."
            if strongest:
                caption += (
                    " Strongest: "
                    + "; ".join(
                        f"{pair['left']} ~ {pair['right']} ({pair['strength']:.2f})"
                        for pair in strongest
                    )
                    + "."
                )
            self.add_chart(
                png, title="How strongly columns move together", caption=caption
            )
        for item in evidence.get("explore_entity_activity", []):
            value = item.value
            stats = value["rows_per_entity"]
            self.add_chart(
                charts.bucket_bars(
                    [(f"Rows per {value['column']}", stats["histogram"])],
                    width=charts.FULL_WIDTH * 0.6,
                ),
                title=f"Activity per {value['column']}",
                caption=(
                    f"Median {_n(stats['median'])} rows, maximum {_n(stats['max'])}. The busiest 1% of values "
                    f"account for {_pct(value.get('top_1pct_share'))} of rows."
                ),
            )
        for item in evidence.get("profile_sample_rows", []):
            value = item.value
            names = value["columns"][:8]
            rows = [
                [_cell(cell) for cell in row["values"][: len(names)]]
                for row in value["head"][:6]
            ]
            if rows:
                # Give each column room in proportion to what it holds.
                widths = [
                    max(5, len(name), *(len(row[index]) for row in rows))
                    for index, name in enumerate(names)
                ]
                block = [
                    self.text("Example rows", "h3"),
                    self.table(names, rows, widths),
                ]
                hidden = (
                    len(value["columns"])
                    - len(names)
                    + value.get("hidden_column_count", 0)
                )
                if hidden > 0:
                    block.append(
                        self.text(f"{hidden} more column(s) not shown.", "note")
                    )
                story.append(KeepTogether(block))
        story.append(Spacer(1, 10))

    def tables(self) -> None:
        catalog = self.result.catalog
        if not catalog.tables:
            return
        ordered = sorted(catalog.tables, key=lambda table: -table.row_count)
        detailed = {table.name for table in ordered[: self.detailed_tables]}
        lead = "One section per table, largest first."
        if len(ordered) > len(detailed):
            lead += f" The {len(detailed)} largest tables include charts; the rest list column statistics only."
        self.section("Table by table", lead)
        for table in ordered:
            self.table_section(table, table.name in detailed)

    def appendix(self) -> None:
        result = self.result
        self.section("Method and limits")
        if result.warnings or result.failures:
            self.story.append(self.heading("Steps that could not be completed", 1))
            rows = [
                [warning.table or "–", warning.code, _clip(warning.message, 220)]
                for warning in result.warnings
            ]
            rows += [
                [failure.table or "–", failure.stage, _clip(failure.message, 220)]
                for failure in result.failures
            ]
            self.story.append(
                self.table(["Table", "Step", "What happened"], rows[:200], [20, 18, 62])
            )
        if result.sampling:
            self.story.append(self.heading("Sampling", 1))
            rows = [
                [
                    _clip(record.operation, 44),
                    _n(record.source_rows),
                    _n(record.sampled_rows),
                    record.strategy.replace("_", " "),
                ]
                for record in result.sampling
            ]
            self.story.append(
                self.table(
                    ["Where", "Rows available", "Rows used", "How"],
                    rows[:300],
                    [38, 16, 14, 32],
                    right=(1, 2),
                )
            )
        config = result.metadata.get("config")
        if config:
            self.story.append(self.heading("Settings", 1))
            rows = [[key.replace("_", " "), _n(value)] for key, value in config.items()]
            half = (len(rows) + 1) // 2
            paired = [
                left + (rows[half + index] if half + index < len(rows) else ["", ""])
                for index, left in enumerate(rows[:half])
            ]
            self.story.append(
                self.table(
                    ["Setting", "Value", "Setting", "Value"], paired, [30, 20, 30, 20]
                )
            )
        if result.assumptions:
            self.story.append(self.heading("Assumptions", 1))
            for assumption in result.assumptions:
                self.story.append(self.text(f"• {assumption}"))
        duration = result.metadata.get("duration_seconds")
        if duration is not None:
            self.story.append(Spacer(1, 8))
            self.story.append(
                self.text(f"The analysis took {_n(float(duration))} seconds.", "note")
            )


def _default_title(result: AnalysisResult) -> str:
    return {
        "exploration": "Data exploration report",
        "profile": "Data profile",
    }.get(result.goal, f"{result.goal.replace('_', ' ').capitalize()} report")


def _default_subtitle(result: AnalysisResult) -> str | None:
    sources = result.metadata.get("sources")
    if sources:
        parts = []
        for source in sources:
            label = str(source.get("kind", "")).capitalize()
            if source.get("database"):
                label += f" · {source['database']}"
            parts.append(label)
        return "  |  ".join(parts)
    tables = result.catalog.tables
    return tables[0].name if len(tables) == 1 else f"{len(tables)} tables"


def build_story(
    result: AnalysisResult,
    *,
    branding: BrandingLike = None,
    title: str | None = None,
    subtitle: str | None = None,
    detailed_tables: int = DEFAULT_DETAILED_TABLES,
) -> tuple[list[Any], str, Branding, str]:
    """The report as a list of flowables, before it is laid out on pages.

    Returns ``(story, title, brand, stamp)``. Split from :func:`render_pdf` so
    the content can be inspected without parsing a PDF: fonts are embedded as
    subsets, which makes text in the finished file unreadable to a plain search.
    """
    _register_fonts()
    brand = resolve_branding(branding)
    title = title or _default_title(result)
    subtitle = subtitle if subtitle is not None else _default_subtitle(result)
    started = result.metadata.get("started_at")
    try:
        moment = (
            datetime.fromisoformat(started) if started else datetime.now(timezone.utc)
        )
    except (TypeError, ValueError):
        moment = datetime.now(timezone.utc)
    stamp = moment.strftime("%d %b %Y, %H:%M UTC")

    builder = _Builder(result, brand, detailed_tables)
    builder.cover(title, subtitle, stamp)
    builder.contents()
    builder.summary()
    builder.inventory()
    builder.relationships()
    builder.links()
    builder.labelled()
    builder.findings()
    builder.tables()
    builder.appendix()
    return builder.story, title, brand, stamp


def render_pdf(
    result: AnalysisResult,
    *,
    branding: BrandingLike = None,
    title: str | None = None,
    subtitle: str | None = None,
    detailed_tables: int = DEFAULT_DETAILED_TABLES,
) -> bytes:
    """Render a result as a PDF document and return its bytes."""
    story, title, brand, stamp = build_story(
        result,
        branding=branding,
        title=title,
        subtitle=subtitle,
        detailed_tables=detailed_tables,
    )
    buffer = io.BytesIO()
    document = _Document(buffer, title, brand, stamp)
    document.multiBuild(story)
    return buffer.getvalue()
