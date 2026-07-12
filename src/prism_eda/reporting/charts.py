"""Dependency-free inline-SVG charts for the HTML report.

The analysis layer emits raw chart data (histogram bins, box five-number
summaries, scatter points, per-row contributions) as structured evidence. These
builders turn that data into small, self-contained SVG fragments — no
JavaScript, no external libraries, no network — so the report stays a single
portable file and still *shows* the analyst the data behind every claim.
"""

from __future__ import annotations

import math
from typing import Any

from markupsafe import Markup, escape


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if number.is_integer():
        return f"{int(number)}"
    return f"{number:.4g}"


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def histogram_svg(distribution: dict[str, Any]) -> Markup:
    """A histogram with a box-plot strip beneath, flagged values highlighted."""
    histogram = distribution.get("histogram", {})
    counts = histogram.get("counts", [])
    edges = histogram.get("edges", [])
    box = distribution.get("box", {})
    flagged = distribution.get("flagged_values", []) or []
    if not counts or len(edges) != len(counts) + 1:
        return Markup("")

    width, height = 680.0, 184.0
    pad_l, pad_r, pad_t = 14.0, 14.0, 12.0
    hist_h = 104.0
    strip_y = pad_t + hist_h + 22.0
    x_lo, x_hi = float(edges[0]), float(edges[-1])
    max_count = max(counts) or 1
    flagged_set = {float(value) for value in flagged}

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Distribution of {escape(distribution.get("column", ""))}">',
        f'<line class="chart-axis" x1="{pad_l}" y1="{pad_t + hist_h}" '
        f'x2="{width - pad_r}" y2="{pad_t + hist_h}"></line>',
    ]

    bar_gap = 1.5
    for index, count in enumerate(counts):
        left = _scale(float(edges[index]), x_lo, x_hi, pad_l, width - pad_r)
        right = _scale(float(edges[index + 1]), x_lo, x_hi, pad_l, width - pad_r)
        bar_w = max(right - left - bar_gap, 0.6)
        bar_h = (count / max_count) * hist_h if count else 0.0
        top = pad_t + hist_h - bar_h
        bin_lo, bin_hi = float(edges[index]), float(edges[index + 1])
        is_flagged = any(bin_lo <= value <= bin_hi for value in flagged_set)
        css = "chart-bar chart-bar-flagged" if is_flagged else "chart-bar"
        title = f"{_format_number(bin_lo)} to {_format_number(bin_hi)}: {count} row(s)"
        parts.append(
            f"<g><title>{escape(title)}</title>"
            f'<rect class="{css}" x="{left:.1f}" y="{top:.1f}" '
            f'width="{bar_w:.1f}" height="{bar_h:.1f}" rx="1.5"></rect></g>'
        )

    # Box-plot strip: whiskers to the fences, box across the IQR, median rule.
    if box:

        def bx(value: float) -> float:
            return _scale(float(value), x_lo, x_hi, pad_l, width - pad_r)

        whisker_lo = max(float(box["min"]), float(box["lower_fence"]))
        whisker_hi = min(float(box["max"]), float(box["upper_fence"]))
        q1x, q3x = bx(box["q1"]), bx(box["q3"])
        parts.append(
            f'<line class="chart-whisker" x1="{bx(whisker_lo):.1f}" y1="{strip_y}" '
            f'x2="{bx(whisker_hi):.1f}" y2="{strip_y}"></line>'
            f'<rect class="chart-box" x="{q1x:.1f}" y="{strip_y - 9}" '
            f'width="{max(q3x - q1x, 1):.1f}" height="18" rx="3"></rect>'
            f'<line class="chart-median" x1="{bx(box["median"]):.1f}" '
            f'y1="{strip_y - 10}" x2="{bx(box["median"]):.1f}" '
            f'y2="{strip_y + 10}"></line>'
        )
        # Flagged values as ticks on the strip.
        for value in sorted(flagged_set):
            parts.append(
                f'<line class="chart-flag-tick" x1="{bx(value):.1f}" '
                f'y1="{strip_y - 13}" x2="{bx(value):.1f}" y2="{strip_y + 13}">'
                f"<title>flagged: {escape(_format_number(value))}</title></line>"
            )
        parts.append(
            f'<text class="chart-label" x="{pad_l}" y="{height - 4}" '
            f'text-anchor="start">{escape(_format_number(box["min"]))}</text>'
            f'<text class="chart-label" x="{bx(box["median"]):.1f}" '
            f'y="{height - 4}" text-anchor="middle">median '
            f"{escape(_format_number(box['median']))}</text>"
            f'<text class="chart-label" x="{width - pad_r}" y="{height - 4}" '
            f'text-anchor="end">{escape(_format_number(box["max"]))}</text>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def scatter_svg(scatter: dict[str, Any]) -> Markup:
    """A scatter of the most relevant numeric pair, flagged rows highlighted."""
    points = scatter.get("points", [])
    if not points:
        return Markup("")
    width, height = 680.0, 300.0
    pad_l, pad_r, pad_t, pad_b = 48.0, 16.0, 16.0, 36.0
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_pad = (x_hi - x_lo) * 0.05 or 1.0
    y_pad = (y_hi - y_lo) * 0.05 or 1.0
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad

    def px(value: float) -> float:
        return _scale(value, x_lo, x_hi, pad_l, width - pad_r)

    def py(value: float) -> float:
        return _scale(value, y_lo, y_hi, height - pad_b, pad_t)

    x_col = escape(scatter.get("x_column", "x"))
    y_col = escape(scatter.get("y_column", "y"))
    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="{x_col} versus {y_col}">',
        f'<line class="chart-axis" x1="{pad_l}" y1="{height - pad_b}" '
        f'x2="{width - pad_r}" y2="{height - pad_b}"></line>',
        f'<line class="chart-axis" x1="{pad_l}" y1="{pad_t}" '
        f'x2="{pad_l}" y2="{height - pad_b}"></line>',
    ]
    normal = [point for point in points if not point.get("flagged")]
    flagged = [point for point in points if point.get("flagged")]
    for point in normal:
        parts.append(
            f'<circle class="chart-dot" cx="{px(float(point["x"])):.1f}" '
            f'cy="{py(float(point["y"])):.1f}" r="3.4"></circle>'
        )
    for point in flagged:
        title = (
            f"{x_col}={_format_number(point['x'])}, "
            f"{y_col}={_format_number(point['y'])}"
        )
        parts.append(
            f'<circle class="chart-dot chart-dot-flagged" '
            f'cx="{px(float(point["x"])):.1f}" cy="{py(float(point["y"])):.1f}" '
            f'r="5"><title>{escape(title)}</title></circle>'
        )
    parts.append(
        f'<text class="chart-label" x="{(pad_l + width - pad_r) / 2:.0f}" '
        f'y="{height - 8}" text-anchor="middle">{x_col}</text>'
        f'<text class="chart-label" x="14" y="{(pad_t + height - pad_b) / 2:.0f}" '
        f'text-anchor="middle" transform="rotate(-90 14 '
        f'{(pad_t + height - pad_b) / 2:.0f})">{y_col}</text>'
        f'<text class="chart-label" x="{pad_l}" y="{height - pad_b + 14}" '
        f'text-anchor="start">{escape(_format_number(x_lo))}</text>'
        f'<text class="chart-label" x="{width - pad_r}" y="{height - pad_b + 14}" '
        f'text-anchor="end">{escape(_format_number(x_hi))}</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def _truncate(text: str, limit: int = 13) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def why_bars_svg(contributors: list[dict[str, Any]]) -> Markup:
    """Fixed-lane σ bars: [label] [rail + bar] [σ value in its own gutter].

    Each row is one column's robust deviation. The bar is capped inside the rail
    so the σ figure in the right gutter never overlaps it. Used both for the
    univariate spikes and for the full per-column profile behind a multivariate
    flag — where seeing *every* column's bar is exactly what tells the analyst
    whether the row is broadly unusual or driven by one column.
    """
    if not contributors:
        return Markup("")
    width = 300.0
    row_h = 22.0
    bar_x = 96.0
    track_w = 128.0
    ref = 6.0  # bars saturate at 6σ so ordinary and extreme rows stay comparable
    min_len = 4.0
    height = len(contributors) * row_h + 4.0
    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" '
        f'height="{height:.0f}" role="img" class="whybars" aria-label="why flagged">'
    ]
    for index, contributor in enumerate(contributors):
        z = float(contributor.get("robust_z", 0.0))
        y = index * row_h
        text_y = y + row_h / 2 + 3.5
        rail_y = y + row_h / 2 - 4
        magnitude = min(1.0, abs(z) / ref)
        length = max(min_len, magnitude * track_w)
        css = "whybars-bar-high" if z >= 0 else "whybars-bar-low"
        sign = "+" if z >= 0 else "−"
        column = str(contributor.get("column", ""))
        label = escape(_truncate(column))
        title = (
            f"{column} {_format_number(contributor.get('value'))} vs typical "
            f"{_format_number(contributor.get('baseline'))} ({z:+.1f}σ)"
        )
        parts.append(
            f"<g><title>{escape(title)}</title>"
            f'<rect class="whybars-rail" x="{bar_x}" y="{rail_y:.1f}" '
            f'width="{track_w}" height="8" rx="4"></rect>'
            f'<rect class="{css}" x="{bar_x}" y="{rail_y:.1f}" '
            f'width="{length:.1f}" height="8" rx="4"></rect>'
            f'<text class="whybars-label" x="0" y="{text_y:.1f}">{label}</text>'
            f'<text class="whybars-z" x="{width:.0f}" y="{text_y:.1f}" '
            f'text-anchor="end">{sign}{abs(z):.1f}σ</text></g>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def peer_group_svg(conditional: dict[str, Any]) -> Markup:
    """A peer-band strip: where this row's value sits vs its in-context peers.

    A conditional (contextual) outlier is only meaningful against its peer group
    — the rows sharing its condition-column bin. This draws that group's typical
    band (middle 50%, whiskers to the range) and marks the flagged row's value
    outside it, so the *contextual* reason is shown, not just asserted.
    """
    keys = ("peer_q1", "peer_q3", "peer_min", "peer_max", "peer_median", "value")
    if any(conditional.get(key) is None for key in keys):
        return Markup("")
    q1 = float(conditional["peer_q1"])
    q3 = float(conditional["peer_q3"])
    p_min = float(conditional["peer_min"])
    p_max = float(conditional["peer_max"])
    median = float(conditional["peer_median"])
    value = float(conditional["value"])

    width, height = 300.0, 66.0
    pad_l, pad_r = 8.0, 8.0
    axis_y = 30.0
    lo = min(p_min, value)
    hi = max(p_max, value)
    span = (hi - lo) or 1.0
    lo -= span * 0.08
    hi += span * 0.08

    def sx(val: float) -> float:
        return _scale(val, lo, hi, pad_l, width - pad_r)

    def anchored(x: float) -> tuple[float, str]:
        """Keep a text label inside the strip instead of clipping at the edge."""
        if x > width - pad_r - 60:
            return width - pad_r, "end"
        if x < pad_l + 60:
            return pad_l, "start"
        return x, "middle"

    row_outside = value < q1 or value > q3
    value_css = "peer-value" if row_outside else "peer-value peer-value-inside"
    band_x, band_anchor = anchored(sx(median))
    value_x, value_anchor = anchored(sx(value))
    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" '
        f'height="{height:.0f}" role="img" class="peer" '
        f'aria-label="value versus peer group">',
        # whiskers across the peer range, band across the middle 50%
        f'<line class="peer-whisker" x1="{sx(p_min):.1f}" y1="{axis_y}" '
        f'x2="{sx(p_max):.1f}" y2="{axis_y}"></line>',
        f'<rect class="peer-band" x="{sx(q1):.1f}" y="{axis_y - 9}" '
        f'width="{max(sx(q3) - sx(q1), 1):.1f}" height="18" rx="3"></rect>',
        f'<line class="peer-median" x1="{sx(median):.1f}" y1="{axis_y - 11}" '
        f'x2="{sx(median):.1f}" y2="{axis_y + 11}"></line>',
        f'<line class="peer-cap" x1="{sx(p_min):.1f}" y1="{axis_y - 5}" '
        f'x2="{sx(p_min):.1f}" y2="{axis_y + 5}"></line>',
        f'<line class="peer-cap" x1="{sx(p_max):.1f}" y1="{axis_y - 5}" '
        f'x2="{sx(p_max):.1f}" y2="{axis_y + 5}"></line>',
        # the flagged row's own value
        f'<circle class="{value_css}" cx="{sx(value):.1f}" cy="{axis_y}" r="5">'
        f"<title>this row: {escape(_format_number(value))}</title></circle>",
        f'<text class="peer-label" x="{band_x:.1f}" y="{axis_y - 15:.0f}" '
        f'text-anchor="{band_anchor}">peers {escape(_format_number(q1))}'
        f"–{escape(_format_number(q3))}</text>",
        f'<text class="peer-value-label" x="{value_x:.1f}" '
        f'y="{axis_y + 22:.0f}" text-anchor="{value_anchor}">this row '
        f"{escape(_format_number(value))}</text>",
        "</svg>",
    ]
    return Markup("".join(parts))


def format_cell(value: Any) -> str:
    """Human-friendly cell formatting for the flagged-rows table."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return _format_number(value)
    return str(value)
