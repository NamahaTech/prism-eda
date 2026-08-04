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


# Above this many unflagged points, a scatter stops attaching a hover tooltip to
# each one. It is a file-size decision: the tooltip markup outweighs the marker
# itself, and on a crowded plot the individual value is not what is being read.
SCATTER_TOOLTIP_MAX_POINTS = 150


def histogram_svg(distribution: dict[str, Any], compact: bool = False) -> Markup:
    """A histogram with a box-plot strip beneath, flagged values highlighted.

    ``compact`` renders the card-sized variant: same data, shorter, and without
    the box strip, which is unreadable at that height.
    """
    histogram = distribution.get("histogram", {})
    counts = histogram.get("counts", [])
    edges = histogram.get("edges", [])
    box = {} if compact else distribution.get("box", {})
    flagged = distribution.get("flagged_values", []) or []
    if not counts or len(edges) != len(counts) + 1:
        return Markup("")

    width, height = (300.0, 96.0) if compact else (680.0, 184.0)
    pad_l, pad_r, pad_t = 14.0, 14.0, 12.0
    hist_h = 62.0 if compact else 104.0
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
    if compact and edges:
        parts.append(
            f'<text class="chart-label" x="{pad_l}" y="{height - 4}" '
            f'text-anchor="start">{escape(_format_number(edges[0]))}</text>'
            f'<text class="chart-label" x="{width - pad_r}" y="{height - 4}" '
            f'text-anchor="end">{escape(_format_number(edges[-1]))}</text>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def dual_histogram_svg(distribution: dict[str, Any]) -> Markup:
    """A dual-series histogram comparing two datasets with different colors."""
    edges = distribution.get("edges", [])
    base_counts = distribution.get("base_counts", [])
    comp_counts = distribution.get("compare_counts", [])
    if not edges or not base_counts or not comp_counts:
        return Markup("")

    width, height = 680.0, 184.0
    pad_l, pad_r, pad_t, pad_b = 14.0, 14.0, 12.0, 24.0
    hist_h = height - pad_t - pad_b
    x_lo, x_hi = float(edges[0]), float(edges[-1])
    max_count = max(max(base_counts), max(comp_counts)) or 1

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Comparison Distribution">',
        f'<line class="chart-axis" x1="{pad_l}" y1="{pad_t + hist_h}" '
        f'x2="{width - pad_r}" y2="{pad_t + hist_h}"></line>',
    ]

    bar_gap = 1.0
    for index in range(len(base_counts)):
        b_count = base_counts[index]
        c_count = comp_counts[index]
        left = _scale(float(edges[index]), x_lo, x_hi, pad_l, width - pad_r)
        right = _scale(float(edges[index + 1]), x_lo, x_hi, pad_l, width - pad_r)

        # We split the bin width in two for side-by-side bars
        bin_w = max(right - left, 1.2)
        bar_w = max((bin_w - bar_gap * 2) / 2.0, 0.4)

        b_h = (b_count / max_count) * hist_h if b_count else 0.0
        c_h = (c_count / max_count) * hist_h if c_count else 0.0

        b_top = pad_t + hist_h - b_h
        c_top = pad_t + hist_h - c_h

        # Base series (blue-ish)
        if b_count > 0:
            parts.append(
                f'<rect class="chart-bar" style="fill: #2563eb; opacity: 0.8;" '
                f'x="{left + bar_gap / 2:.1f}" y="{b_top:.1f}" '
                f'width="{bar_w:.1f}" height="{b_h:.1f}">'
                f"<title>Base: {b_count}</title></rect>"
            )
        # Compare series (red-ish)
        if c_count > 0:
            parts.append(
                f'<rect class="chart-bar" style="fill: #dc2626; opacity: 0.8;" '
                f'x="{left + bar_gap / 2 + bar_w:.1f}" y="{c_top:.1f}" '
                f'width="{bar_w:.1f}" height="{c_h:.1f}">'
                f"<title>Compare: {c_count}</title></rect>"
            )

    parts.append(
        f'<text class="chart-label" x="{pad_l}" y="{height - 4}" '
        f'text-anchor="start">{escape(_format_number(edges[0]))}</text>'
        f'<text class="chart-label" x="{width - pad_r}" y="{height - 4}" '
        f'text-anchor="end">{escape(_format_number(edges[-1]))}</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def scatter_svg(scatter: dict[str, Any], compact: bool = False) -> Markup:
    """A scatter of the most relevant numeric pair, flagged rows highlighted."""
    points = scatter.get("points", [])
    if not points:
        return Markup("")
    width, height = (480.0, 300.0) if compact else (680.0, 300.0)
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
    # A tooltip per point roughly triples the markup, and on a dense plot nobody
    # hovers an individual dot — the shape is the message. Above this many
    # points the unflagged dots drop their tooltips; flagged ones always keep
    # theirs, because those are the ones a reader goes looking for.
    label_points = len(normal) <= SCATTER_TOOLTIP_MAX_POINTS
    for point in normal:
        marker = (
            f'<circle class="chart-dot" cx="{px(float(point["x"])):.1f}" '
            f'cy="{py(float(point["y"])):.1f}" r="3.4"'
        )
        if label_points:
            title = (
                f"{x_col}={_format_number(point['x'])}, "
                f"{y_col}={_format_number(point['y'])}"
            )
            parts.append(f"{marker}><title>{escape(title)}</title></circle>")
        else:
            parts.append(f"{marker}></circle>")
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


def residual_scatter_svg(residuals: dict[str, Any]) -> Markup:
    """Residuals against fitted values, with the zero rule and a spread band.

    The plot that decides whether a regression fit is trustworthy. Two things
    are drawn behind the points because stating them in prose does not let a
    reader check them: the zero rule, which a well-behaved cloud straddles
    evenly, and the per-bin spread band, which fans out when the error grows
    with the prediction. A reader who sees the band widen has understood
    heteroscedasticity without the word.
    """
    points = residuals.get("points", [])
    if not points:
        return Markup("")
    width, height = 680.0, 310.0
    # The top padding leaves a clear band for the off-scale note, so it never
    # lands on top of the very markers it is describing.
    pad_l, pad_r, pad_t, pad_b = 52.0, 16.0, 28.0, 38.0
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    bins = residuals.get("bins", []) or []

    x_lo, x_hi = min(xs), max(xs)
    # The vertical scale must include the band, or the band clips at the edges.
    band_values = [
        value
        for item in bins
        for value in (
            item["mean_residual"] + item["std_residual"],
            item["mean_residual"] - item["std_residual"],
        )
    ]
    # A residual plot is exactly the chart most likely to contain a handful of
    # enormous residuals — that is what it is for. Scaling to them compresses
    # every other point into an unreadable strip, which loses the pattern the
    # chart exists to show. So the axis is set from a robust range and the
    # extremes are pinned to the edge rather than dropped: the shape stays
    # readable, the outliers stay visible, and the count of pinned points is
    # stated on the chart so nothing is quietly hidden.
    ordered = sorted(ys)
    q1 = ordered[int(len(ordered) * 0.25)]
    q3 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.75))]
    iqr = q3 - q1
    if iqr > 0:
        y_lo = max(min(ys), q1 - 3.0 * iqr)
        y_hi = min(max(ys), q3 + 3.0 * iqr)
    else:
        y_lo, y_hi = min(ys), max(ys)
    y_lo = min([y_lo, 0.0, *band_values])
    y_hi = max([y_hi, 0.0, *band_values])
    x_pad = (x_hi - x_lo) * 0.05 or 1.0
    y_pad = (y_hi - y_lo) * 0.08 or 1.0
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad
    off_scale = sum(1 for value in ys if value < y_lo or value > y_hi)

    def px(value: float) -> float:
        return _scale(value, x_lo, x_hi, pad_l, width - pad_r)

    def py(value: float) -> float:
        return _scale(min(max(value, y_lo), y_hi), y_lo, y_hi, height - pad_b, pad_t)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Residuals against fitted values">',
        f'<line class="chart-axis" x1="{pad_l}" y1="{height - pad_b}" '
        f'x2="{width - pad_r}" y2="{height - pad_b}"></line>',
        f'<line class="chart-axis" x1="{pad_l}" y1="{pad_t}" '
        f'x2="{pad_l}" y2="{height - pad_b}"></line>',
    ]

    if len(bins) >= 2:
        upper = " ".join(
            f"{px((item['low'] + item['high']) / 2):.1f},"
            f"{py(item['mean_residual'] + item['std_residual']):.1f}"
            for item in bins
        )
        lower = " ".join(
            f"{px((item['low'] + item['high']) / 2):.1f},"
            f"{py(item['mean_residual'] - item['std_residual']):.1f}"
            for item in reversed(bins)
        )
        parts.append(
            f'<polygon class="chart-band" points="{upper} {lower}"></polygon>'
            f'<polyline class="chart-band-edge" points="{upper}"></polyline>'
        )

    parts.append(
        f'<line class="chart-zero" x1="{pad_l}" y1="{py(0.0):.1f}" '
        f'x2="{width - pad_r}" y2="{py(0.0):.1f}"></line>'
    )

    normal = [point for point in points if not point.get("flagged")]
    flagged = [point for point in points if point.get("flagged")]
    label_points = len(normal) <= SCATTER_TOOLTIP_MAX_POINTS

    def clipped(value: float) -> bool:
        return value < y_lo or value > y_hi

    for point in normal:
        value = float(point["y"])
        css = "chart-dot chart-dot-clipped" if clipped(value) else "chart-dot"
        marker = (
            f'<circle class="{css}" cx="{px(float(point["x"])):.1f}" '
            f'cy="{py(value):.1f}" r="3.4"'
        )
        if label_points or clipped(value):
            suffix = " — beyond the axis range" if clipped(value) else ""
            title = (
                f"predicted {_format_number(point['x'])}, "
                f"residual {_format_number(value)}{suffix}"
            )
            parts.append(f"{marker}><title>{escape(title)}</title></circle>")
        else:
            parts.append(f"{marker}></circle>")
    for point in flagged:
        value = float(point["y"])
        suffix = " — beyond the axis range" if clipped(value) else ""
        title = (
            f"predicted {_format_number(point['x'])}, "
            f"residual {_format_number(value)} — under review{suffix}"
        )
        css = "chart-dot chart-dot-flagged"
        if clipped(value):
            css += " chart-dot-clipped"
        parts.append(
            f'<circle class="{css}" cx="{px(float(point["x"])):.1f}" '
            f'cy="{py(value):.1f}" r="5"><title>{escape(title)}</title></circle>'
        )

    mid_x = (pad_l + width - pad_r) / 2
    mid_y = (pad_t + height - pad_b) / 2
    parts.append(
        f'<text class="chart-label" x="{mid_x:.0f}" y="{height - 8}" '
        f'text-anchor="middle">predicted value</text>'
        f'<text class="chart-label" x="14" y="{mid_y:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {mid_y:.0f})">residual</text>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{py(0.0) + 3.5:.1f}" '
        f'text-anchor="end">0</text>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{pad_t + 4:.1f}" '
        f'text-anchor="end">{escape(_format_number(y_hi))}</text>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{height - pad_b:.1f}" '
        f'text-anchor="end">{escape(_format_number(y_lo))}</text>'
    )
    if off_scale:
        noun = "point" if off_scale == 1 else "points"
        parts.append(
            f'<text class="chart-label chart-note-mark" x="{width - pad_r}" '
            f'y="12" text-anchor="end">{off_scale} {noun} beyond this range, '
            f"pinned to the edge</text>"
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def conditional_bias_svg(bias: dict[str, Any]) -> Markup:
    """Mean residual per fitted bin, diverging from the zero rule.

    A fit can have an unbiased average and still be wrong everywhere: too low at
    the bottom of the range and too high at the top, averaging to nothing. This
    chart is the one that catches it. Bars run above or below the zero rule, so
    the sign is carried by direction as well as hue and never depends on colour
    alone.
    """
    bins = bias.get("bins", [])
    spread = float(bias.get("overall_std_residual", 0.0) or 0.0)
    if not bins:
        return Markup("")

    width = 680.0
    pad_l, pad_r, pad_t, pad_b = 52.0, 16.0, 14.0, 40.0
    height = 210.0
    plot_h = height - pad_t - pad_b
    means = [float(item["mean_residual"]) for item in bins]
    extent = max(abs(min(means)), abs(max(means)), spread * 0.5) or 1.0
    zero_y = pad_t + plot_h / 2

    def py(value: float) -> float:
        return zero_y - (value / extent) * (plot_h / 2)

    slot = (width - pad_l - pad_r) / len(bins)
    # A 2px surface gap between adjacent fills, per the shared mark spec.
    bar_w = max(3.0, slot - 2.0)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Mean residual by fitted range">'
    ]
    # A +/- 1 standard-deviation reference makes the bars readable as an effect
    # size rather than raw units.
    if spread > 0:
        for reference in (spread, -spread):
            if abs(reference) > extent:
                continue
            parts.append(
                f'<line class="chart-axis" x1="{pad_l}" y1="{py(reference):.1f}" '
                f'x2="{width - pad_r}" y2="{py(reference):.1f}" '
                f'stroke-dasharray="3 3"></line>'
            )

    for index, item in enumerate(bins):
        mean = float(item["mean_residual"])
        left = pad_l + index * slot + (slot - bar_w) / 2
        top = min(py(mean), zero_y)
        bar_h = max(1.5, abs(py(mean) - zero_y))
        css = "chart-bar-pos" if mean >= 0 else "chart-bar-neg"
        direction = "under-predicts" if mean >= 0 else "over-predicts"
        title = (
            f"{_format_number(item['low'])} to {_format_number(item['high'])}: "
            f"{direction} by {_format_number(abs(mean))} on average "
            f"({item['count']} rows)"
        )
        parts.append(
            f"<g><title>{escape(title)}</title>"
            f'<rect class="{css}" x="{left:.1f}" y="{top:.1f}" '
            f'width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2"></rect></g>'
        )

    parts.append(
        f'<line class="chart-zero" x1="{pad_l}" y1="{zero_y:.1f}" '
        f'x2="{width - pad_r}" y2="{zero_y:.1f}"></line>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{zero_y + 3.5:.1f}" '
        f'text-anchor="end">0</text>'
    )
    if spread > 0 and spread <= extent:
        parts.append(
            f'<text class="chart-label" x="{pad_l - 6}" y="{py(spread) + 3.5:.1f}" '
            f'text-anchor="end">+1σ</text>'
            f'<text class="chart-label" x="{pad_l - 6}" y="{py(-spread) + 3.5:.1f}" '
            f'text-anchor="end">−1σ</text>'
        )
    parts.append(
        f'<text class="chart-label" x="{pad_l}" y="{height - 20}" '
        f'text-anchor="start">{escape(_format_number(bins[0]["low"]))}</text>'
        f'<text class="chart-label" x="{width - pad_r}" y="{height - 20}" '
        f'text-anchor="end">{escape(_format_number(bins[-1]["high"]))}</text>'
        f'<text class="chart-label" x="{(pad_l + width - pad_r) / 2:.0f}" '
        f'y="{height - 5}" text-anchor="middle">predicted value</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def _series_extent(*series: list[dict[str, Any]]) -> tuple[float, float]:
    values = [
        float(point["y"])
        for points in series
        for point in points
        if point.get("y") is not None
    ]
    if not values:
        return 0.0, 1.0
    low, high = min(values), max(values)
    if low == high:
        return low - 1.0, high + 1.0
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _polyline(
    points: list[dict[str, Any]],
    px: Any,
    py: Any,
    total: int,
) -> list[str]:
    """Line segments that break at gaps rather than drawing through them.

    A single polyline across a missing period draws a straight line over the
    outage, which is the one thing the chart must not imply happened.
    """
    runs: list[list[str]] = [[]]
    for index, point in enumerate(points):
        if point.get("y") is None:
            if runs[-1]:
                runs.append([])
            continue
        runs[-1].append(f"{px(index, total):.1f},{py(float(point['y'])):.1f}")
    return [" ".join(run) for run in runs if len(run) > 1]


def series_line_svg(data: dict[str, Any]) -> Markup:
    """The series over time, with gaps left open and change points marked."""
    points = data.get("points", [])
    if not points:
        return Markup("")
    trend = data.get("trend_points", []) or []
    changes = set(data.get("change_points", []) or [])

    width, height = 680.0, 240.0
    pad_l, pad_r, pad_t, pad_b = 52.0, 16.0, 16.0, 34.0
    total = len(points)
    y_lo, y_hi = _series_extent(points, trend)

    def px(index: int, count: int) -> float:
        if count <= 1:
            return pad_l
        return pad_l + (index / (count - 1)) * (width - pad_l - pad_r)

    def py(value: float) -> float:
        return _scale(value, y_lo, y_hi, height - pad_b, pad_t)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="{escape(str(data.get("value", "series")))} over time">',
        f'<line class="chart-axis" x1="{pad_l}" y1="{height - pad_b}" '
        f'x2="{width - pad_r}" y2="{height - pad_b}"></line>',
        f'<line class="chart-axis" x1="{pad_l}" y1="{pad_t}" '
        f'x2="{pad_l}" y2="{height - pad_b}"></line>',
    ]

    # Change points first, so the series draws over the rules rather than under.
    stamps = [point["t"] for point in points]
    for stamp in changes:
        if stamp not in stamps:
            continue
        x = px(stamps.index(stamp), total)
        parts.append(
            f'<line class="chart-changepoint" x1="{x:.1f}" y1="{pad_t}" '
            f'x2="{x:.1f}" y2="{height - pad_b}"><title>level shift: '
            f"{escape(str(stamp)[:10])}</title></line>"
        )

    for run in _polyline(points, px, py, total):
        parts.append(f'<polyline class="chart-line" points="{run}"></polyline>')
    for run in _polyline(trend, px, py, len(trend)):
        parts.append(f'<polyline class="chart-line-trend" points="{run}"></polyline>')

    first = str(points[0]["t"])[:10]
    last = str(points[-1]["t"])[:10]
    parts.append(
        f'<text class="chart-label" x="{pad_l}" y="{height - 12}" '
        f'text-anchor="start">{escape(first)}</text>'
        f'<text class="chart-label" x="{width - pad_r}" y="{height - 12}" '
        f'text-anchor="end">{escape(last)}</text>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{pad_t + 4}" '
        f'text-anchor="end">{escape(_format_number(y_hi))}</text>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{height - pad_b}" '
        f'text-anchor="end">{escape(_format_number(y_lo))}</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def acf_stems_svg(data: dict[str, Any]) -> Markup:
    """Autocorrelation by lag, with the band inside which a lag means nothing."""
    lags = data.get("lags", [])
    if not lags:
        return Markup("")
    band = float(data.get("confidence_band", 0.0))
    width, height = 680.0, 190.0
    pad_l, pad_r, pad_t, pad_b = 44.0, 16.0, 14.0, 30.0
    plot_h = height - pad_t - pad_b
    zero_y = pad_t + plot_h / 2
    extent = max([abs(float(item["acf"])) for item in lags] + [band, 0.2])

    def py(value: float) -> float:
        return zero_y - (value / extent) * (plot_h / 2)

    slot = (width - pad_l - pad_r) / max(1, len(lags))
    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Autocorrelation by lag">',
        f'<rect class="chart-band" x="{pad_l}" y="{py(band):.1f}" '
        f'width="{width - pad_l - pad_r:.1f}" '
        f'height="{max(1.0, py(-band) - py(band)):.1f}"></rect>',
    ]
    for index, item in enumerate(lags):
        value = float(item["acf"])
        x = pad_l + index * slot + slot / 2
        css = "chart-stem" if item.get("significant") else "chart-stem-quiet"
        parts.append(
            f"<g><title>lag {item['lag']}: {value:+.3f}</title>"
            f'<line class="{css}" x1="{x:.1f}" y1="{zero_y:.1f}" '
            f'x2="{x:.1f}" y2="{py(value):.1f}"></line>'
            f'<circle class="{css}-dot" cx="{x:.1f}" cy="{py(value):.1f}" '
            f'r="2.2"></circle></g>'
        )
    parts.append(
        f'<line class="chart-zero" x1="{pad_l}" y1="{zero_y:.1f}" '
        f'x2="{width - pad_r}" y2="{zero_y:.1f}"></line>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{zero_y + 3.5:.1f}" '
        f'text-anchor="end">0</text>'
        f'<text class="chart-label" x="{pad_l}" y="{height - 10}" '
        f'text-anchor="start">lag 1</text>'
        f'<text class="chart-label" x="{width - pad_r}" y="{height - 10}" '
        f'text-anchor="end">lag {lags[-1]["lag"]}</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


#: Weekday names for a 7-period seasonal profile, which is what daily data has.
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def seasonal_profile_svg(data: dict[str, Any]) -> Markup:
    """The seasonal effect of each position in the cycle, diverging from zero."""
    profile = data.get("seasonal_profile", [])
    if not profile:
        return Markup("")
    width = 680.0
    pad_l, pad_r, pad_t, pad_b = 44.0, 16.0, 12.0, 28.0
    height = 160.0
    plot_h = height - pad_t - pad_b
    zero_y = pad_t + plot_h / 2
    effects = [float(item["effect"]) for item in profile]
    extent = max(abs(min(effects)), abs(max(effects))) or 1.0

    def py(value: float) -> float:
        return zero_y - (value / extent) * (plot_h / 2)

    slot = (width - pad_l - pad_r) / len(profile)
    bar_w = max(3.0, slot - 6.0)
    weekly = len(profile) == 7

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Seasonal effect by position in the cycle">'
    ]
    for index, item in enumerate(profile):
        effect = float(item["effect"])
        left = pad_l + index * slot + (slot - bar_w) / 2
        top = min(py(effect), zero_y)
        bar_h = max(1.5, abs(py(effect) - zero_y))
        css = "chart-bar-pos" if effect >= 0 else "chart-bar-neg"
        label = (
            _WEEKDAYS[int(item["position"]) % 7] if weekly else str(item["position"])
        )
        parts.append(
            f"<g><title>{escape(label)}: {effect:+.4g}</title>"
            f'<rect class="{css}" x="{left:.1f}" y="{top:.1f}" '
            f'width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2"></rect>'
            f'<text class="chart-label" x="{left + bar_w / 2:.1f}" '
            f'y="{height - 10}" text-anchor="middle">{escape(label)}</text></g>'
        )
    parts.append(
        f'<line class="chart-zero" x1="{pad_l}" y1="{zero_y:.1f}" '
        f'x2="{width - pad_r}" y2="{zero_y:.1f}"></line>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{zero_y + 3.5:.1f}" '
        f'text-anchor="end">0</text>'
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


def image_dimension_svg(distribution: dict[str, Any]) -> Markup:
    """Width against height, one dot per distinct size, sized by how many files.

    A single dot means every image already shares one shape. A smear along a
    diagonal means mixed resolutions at a constant aspect ratio; a scattered
    cloud means the shapes disagree too, which is the case that quietly breaks a
    fixed-size resize.
    """
    points = distribution.get("scatter_points", [])
    if not points:
        return Markup("")
    width, height = 680.0, 300.0
    pad_l, pad_r, pad_t, pad_b = 52.0, 18.0, 18.0, 38.0
    widths = [float(point["width"]) for point in points]
    heights = [float(point["height"]) for point in points]
    x_lo, x_hi = min(widths), max(widths)
    y_lo, y_hi = min(heights), max(heights)
    # Pad a degenerate axis so a single distinct size still lands mid-plot.
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 1, x_hi + 1
    if y_hi == y_lo:
        y_lo, y_hi = y_lo - 1, y_hi + 1
    max_count = max(int(point["count"]) for point in points) or 1

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Image width against height">',
        f'<line class="chart-axis" x1="{pad_l}" y1="{height - pad_b}" '
        f'x2="{width - pad_r}" y2="{height - pad_b}"></line>',
        f'<line class="chart-axis" x1="{pad_l}" y1="{pad_t}" '
        f'x2="{pad_l}" y2="{height - pad_b}"></line>',
    ]
    for point in points:
        px = _scale(float(point["width"]), x_lo, x_hi, pad_l, width - pad_r)
        py = _scale(float(point["height"]), y_lo, y_hi, height - pad_b, pad_t)
        count = int(point["count"])
        radius = 3.0 + 7.0 * math.sqrt(count / max_count)
        css = "chart-dot-flagged" if point.get("is_outlier") else "chart-dot"
        title = f"{int(point['width'])}x{int(point['height'])}: {count} image(s)" + (
            " — outlier" if point.get("is_outlier") else ""
        )
        parts.append(
            f"<g><title>{escape(title)}</title>"
            f'<circle class="{css}" cx="{px:.1f}" cy="{py:.1f}" '
            f'r="{radius:.1f}"></circle></g>'
        )
    parts.append(
        f'<text class="chart-label" x="{pad_l}" y="{height - 12}" '
        f'text-anchor="start">{escape(_format_number(x_lo))}</text>'
        f'<text class="chart-label" x="{width - pad_r}" y="{height - 12}" '
        f'text-anchor="end">width {escape(_format_number(x_hi))}</text>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{pad_t + 8}" '
        f'text-anchor="end">{escape(_format_number(y_hi))}</text>'
        f'<text class="chart-label" x="{pad_l - 6}" y="{height - pad_b}" '
        f'text-anchor="end">{escape(_format_number(y_lo))}</text>'
        "</svg>"
    )
    return Markup("".join(parts))


def _bars_svg(
    rows: list[dict[str, Any]],
    *,
    aria_label: str,
    noun: str,
    flagged: set[int],
    limit: int,
    label_width: float = 120.0,
) -> Markup:
    """Horizontal count bars — the shared body of the label and category charts."""
    if not rows:
        return Markup("")
    visible = rows[:limit]
    counts = [int(row["count"]) for row in visible]
    max_count = max(counts) or 1
    width = 680.0
    row_h = 26.0
    track_w = width - label_width - 74.0
    height = len(visible) * row_h + 6.0

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="{escape(aria_label)}">'
    ]
    for index, row in enumerate(visible):
        count = int(row["count"])
        y = index * row_h + 3.0
        bar_w = max(2.0, (count / max_count) * track_w)
        name = str(row.get("value", ""))
        css = "chart-bar-flagged" if index in flagged else "chart-bar"
        share = float(row.get("rate", 0.0))
        title = f"{name}: {count:,} {noun}, {share:.1%}"
        parts.append(
            f"<g><title>{escape(title)}</title>"
            f'<text class="chart-label" x="0" y="{y + 15:.1f}">'
            f"{escape(_truncate(name, 16))}</text>"
            f'<rect class="{css}" x="{label_width}" y="{y + 4:.1f}" '
            f'width="{bar_w:.1f}" height="14" rx="3"></rect>'
            f'<text class="chart-label" x="{label_width + bar_w + 8:.1f}" '
            f'y="{y + 15:.1f}">{count:,}</text></g>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def dual_bar_svg(distribution: dict[str, Any]) -> Markup:
    """A dual-series bar chart for comparing categorical frequencies."""
    labels = distribution.get("labels", [])
    base_counts = distribution.get("base_counts", [])
    comp_counts = distribution.get("compare_counts", [])
    if not labels:
        return Markup("")

    width = 680.0
    bar_h = 16.0
    group_pad = 8.0
    # Two bars per label + padding
    group_h = (bar_h * 2) + group_pad
    pad_t, pad_b, pad_r = 16.0, 16.0, 16.0
    label_w = 160.0
    height = pad_t + pad_b + (len(labels) * group_h)

    max_count = max(max(base_counts), max(comp_counts)) or 1

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" '
        'role="img" class="chart" aria-label="Comparison Categories">'
    ]

    for i, label in enumerate(labels):
        y_top = pad_t + i * group_h

        # Label text
        display_label = escape(_truncate(str(label), 22))
        parts.append(
            f'<text class="chart-label" x="{label_w - 8}" y="{y_top + bar_h + 4}" '
            f'text-anchor="end">{display_label}</text>'
        )

        # Base bar (blue)
        b_count = base_counts[i]
        if b_count > 0:
            b_w = max((b_count / max_count) * (width - label_w - pad_r), 2.0)
            parts.append(
                f'<rect class="chart-bar" style="fill: #2563eb; opacity: 0.8;" '
                f'x="{label_w}" y="{y_top}" '
                f'width="{b_w:.1f}" height="{bar_h}">'
                f"<title>Base: {b_count}</title></rect>"
                f'<text class="chart-label" style="fill: white; font-weight: bold;" '
                f'x="{label_w + b_w - 6}" y="{y_top + 11}" '
                f'text-anchor="end">{b_count}</text>'
            )

        # Compare bar (red)
        c_count = comp_counts[i]
        if c_count > 0:
            c_w = max((c_count / max_count) * (width - label_w - pad_r), 2.0)
            parts.append(
                f'<rect class="chart-bar" style="fill: #dc2626; opacity: 0.8;" '
                f'x="{label_w}" y="{y_top + bar_h + 2}" '
                f'width="{c_w:.1f}" height="{bar_h}">'
                f"<title>Compare: {c_count}</title></rect>"
                f'<text class="chart-label" style="fill: white; font-weight: bold;" '
                f'x="{label_w + c_w - 6}" y="{y_top + bar_h + 13}" '
                f'text-anchor="end">{c_count}</text>'
            )

    parts.append("</svg>")
    return Markup("".join(parts))


def label_bars_svg(rows: list[dict[str, Any]]) -> Markup:
    """Class balance as horizontal bars, smallest class highlighted."""
    if not rows:
        return Markup("")
    visible = rows[:16]
    counts = [int(row["count"]) for row in visible]
    max_count = max(counts) or 1
    smallest = min(counts)
    flagged = set()
    for index, row in enumerate(visible):
        count = int(row["count"])
        # Only call out the smallest class when it is genuinely starved, not
        # merely last in a balanced set.
        starved = count == smallest and max_count >= 5 * max(smallest, 1)
        # Missing labels are always a review item, even when their class count
        # is not the smallest in the dataset.
        unlabeled = str(row.get("value", "")).strip().casefold() == "unlabeled"
        if starved or unlabeled:
            flagged.add(index)
    return _bars_svg(
        rows,
        aria_label="Images per label",
        noun="image(s)",
        flagged=flagged,
        limit=16,
    )


def category_bars_svg(frequency: dict[str, Any]) -> Markup:
    """Label frequencies for a categorical column, long tail folded into one bar."""
    rows = list(frequency.get("counts", []))
    if not rows:
        return Markup("")
    other = int(frequency.get("other_count", 0) or 0)
    flagged: set[int] = set()
    if other:
        remaining = int(frequency.get("other_category_count", 0) or 0)
        rows = [
            *rows,
            {
                "value": f"other ({remaining:,} more)",
                "count": other,
                "rate": other / max(1, sum(int(row["count"]) for row in rows) + other),
            },
        ]
        flagged.add(len(rows) - 1)
    return _bars_svg(
        rows,
        aria_label=f"Value frequencies for {frequency.get('column', '')}",
        noun="row(s)",
        flagged=flagged,
        limit=len(rows),
    )


def missing_bars_svg(missingness: dict[str, Any]) -> Markup:
    """Missing share per column, so structural gaps stand out from stray nulls."""
    columns = [
        row for row in missingness.get("columns", []) if float(row["missing_rate"]) > 0
    ]
    if not columns:
        return Markup("")
    columns.sort(key=lambda row: -float(row["missing_rate"]))
    width = 680.0
    row_h = 24.0
    label_w = 150.0
    track_w = width - label_w - 66.0
    height = len(columns) * row_h + 6.0
    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Missing values per column">'
    ]
    for index, row in enumerate(columns):
        rate = float(row["missing_rate"])
        y = index * row_h + 3.0
        bar_w = max(2.0, rate * track_w)
        # 20% is where the profile promotes missingness to a finding, so the bar
        # and the findings list agree about what counts as a lot.
        css = "chart-bar-flagged" if rate >= 0.2 else "chart-bar"
        title = f"{row['column']}: {int(row['missing_count']):,} missing ({rate:.1%})"
        parts.append(
            f"<g><title>{escape(title)}</title>"
            f'<text class="chart-label" x="0" y="{y + 14:.1f}">'
            f"{escape(_truncate(str(row['column']), 20))}</text>"
            f'<rect class="chart-rail" x="{label_w}" y="{y + 4:.1f}" '
            f'width="{track_w:.1f}" height="12" rx="3"></rect>'
            f'<rect class="{css}" x="{label_w}" y="{y + 4:.1f}" '
            f'width="{bar_w:.1f}" height="12" rx="3"></rect>'
            f'<text class="chart-label" x="{width:.0f}" y="{y + 14:.1f}" '
            f'text-anchor="end">{rate:.1%}</text></g>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def _heatmap_svg(
    labels: list[str],
    cells: dict[tuple[str, str], dict[str, Any]],
    *,
    aria_label: str,
    diagonal: str,
) -> Markup:
    """A square matrix of intensity cells, with the number printed when it fits.

    Colour alone would make the strongest pairs invisible to a reader who cannot
    distinguish the shades, so every cell carries its value as text at readable
    grid sizes and as a tooltip at all sizes.
    """
    if len(labels) < 2:
        return Markup("")
    count = len(labels)
    label_w = 118.0
    cell = max(24.0, min(46.0, 620.0 / count))
    width = label_w + cell * count + 8
    height = label_w + cell * count + 8
    show_text = cell >= 34.0

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart heat" '
        f'aria-label="{escape(aria_label)}">'
    ]
    for index, name in enumerate(labels):
        y = label_w + index * cell + cell / 2 + 3.5
        parts.append(
            f'<text class="chart-label" x="{label_w - 8:.1f}" y="{y:.1f}" '
            f'text-anchor="end">{escape(_truncate(name, 16))}</text>'
        )
        x = label_w + index * cell + cell / 2
        parts.append(
            f'<text class="chart-label" x="{x:.1f}" y="{label_w - 8:.1f}" '
            f'text-anchor="start" transform="rotate(-90 {x:.1f} '
            f'{label_w - 8:.1f})">{escape(_truncate(name, 16))}</text>'
        )

    for row, left in enumerate(labels):
        for column, right in enumerate(labels):
            x = label_w + column * cell
            y = label_w + row * cell
            if left == right:
                parts.append(
                    f'<rect class="heat-diagonal" x="{x:.1f}" y="{y:.1f}" '
                    f'width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2">'
                    f"<title>{escape(f'{left}: {diagonal}')}</title></rect>"
                )
                continue
            entry = cells.get((left, right)) or cells.get((right, left))
            if entry is None:
                parts.append(
                    f'<rect class="heat-empty" x="{x:.1f}" y="{y:.1f}" '
                    f'width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2">'
                    f"<title>{escape(f'{left} vs {right}: not comparable')}</title>"
                    f"</rect>"
                )
                continue
            strength = max(0.0, min(1.0, float(entry["strength"])))
            css = "heat-cell-negative" if entry.get("negative") else "heat-cell"
            # A faint floor keeps a near-zero cell visible as "measured, weak"
            # rather than looking like a hole in the matrix.
            opacity = 0.07 + 0.88 * strength
            parts.append(
                f"<g><title>{escape(str(entry['title']))}</title>"
                f'<rect class="{css}" x="{x:.1f}" y="{y:.1f}" '
                f'width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2" '
                f'style="fill-opacity:{opacity:.3f}"></rect>'
            )
            if show_text:
                text_css = (
                    "heat-value heat-value-light" if strength >= 0.6 else "heat-value"
                )
                parts.append(
                    f'<text class="{text_css}" x="{x + (cell - 2) / 2:.1f}" '
                    f'y="{y + (cell - 2) / 2 + 3.5:.1f}" text-anchor="middle">'
                    f"{escape(entry['label'])}</text>"
                )
            parts.append("</g>")
    parts.append("</svg>")
    return Markup("".join(parts))


def association_heatmap_svg(matrix: dict[str, Any]) -> Markup:
    """Every measured column pair, coloured by association strength."""
    labels = [str(name) for name in matrix.get("columns", [])]
    method_names = {
        "spearman": "Spearman rho",
        "cramers_v": "Cramer's V",
        "correlation_ratio": "correlation ratio",
    }
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in matrix.get("pairs", []):
        strength = float(pair["strength"])
        signed = pair.get("method") == "spearman"
        rho = float(pair["spearman"]) if signed else None
        label = f"{rho:+.2f}"[:5] if signed and rho is not None else f"{strength:.2f}"
        method = method_names.get(str(pair.get("method")), str(pair.get("method")))
        cells[(str(pair["left"]), str(pair["right"]))] = {
            "strength": strength,
            "negative": bool(signed and rho is not None and rho < 0),
            "label": label,
            "title": (f"{pair['left']} vs {pair['right']}: {label} ({method})"),
        }
    return _heatmap_svg(
        labels,
        cells,
        aria_label="Association strength between columns",
        diagonal="a column against itself",
    )


def co_missing_heatmap_svg(co_missing: dict[str, Any]) -> Markup:
    """How often two columns are missing on the same row."""
    labels = [str(name) for name in co_missing.get("columns", [])]
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in co_missing.get("pairs", []):
        jaccard = float(pair["jaccard"])
        cells[(str(pair["left"]), str(pair["right"]))] = {
            "strength": jaccard,
            "negative": False,
            "label": f"{jaccard:.2f}",
            "title": (
                f"{pair['left']} and {pair['right']}: "
                f"{int(pair['both_missing']):,} rows missing both "
                f"(overlap {jaccard:.0%})"
            ),
        }
    return _heatmap_svg(
        labels,
        cells,
        aria_label="Columns whose missing values coincide",
        diagonal="a column against itself",
    )


def timeline_svg(timeline: dict[str, Any]) -> Markup:
    """Row counts over time, so collection gaps are visible rather than inferred."""
    buckets = timeline.get("buckets", [])
    if not buckets:
        return Markup("")
    width, height = 680.0, 150.0
    pad_l, pad_r, pad_t = 14.0, 14.0, 12.0
    plot_h = 100.0
    counts = [int(bucket["count"]) for bucket in buckets]
    max_count = max(counts) or 1
    span = max(1, len(buckets))
    bar_w = max(1.0, (width - pad_l - pad_r) / span - 1.2)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" class="chart" '
        f'aria-label="Rows over time for {escape(timeline.get("column", ""))}">',
        f'<line class="chart-axis" x1="{pad_l}" y1="{pad_t + plot_h}" '
        f'x2="{width - pad_r}" y2="{pad_t + plot_h}"></line>',
    ]
    for index, bucket in enumerate(buckets):
        count = int(bucket["count"])
        left = pad_l + index * (width - pad_l - pad_r) / span
        bar_h = (count / max_count) * plot_h
        start = str(bucket["start"])[:10]
        parts.append(
            f"<g><title>{escape(f'{start}: {count:,} row(s)')}</title>"
            f'<rect class="chart-bar" x="{left:.1f}" '
            f'y="{pad_t + plot_h - bar_h:.1f}" width="{bar_w:.1f}" '
            f'height="{bar_h:.1f}" rx="1.5"></rect></g>'
        )
    granularity = str(timeline.get("granularity", ""))
    parts.append(
        f'<text class="chart-label" x="{pad_l}" y="{height - 6}" '
        f'text-anchor="start">{escape(str(timeline.get("min", ""))[:10])}</text>'
        f'<text class="chart-label" x="{width / 2:.0f}" y="{height - 6}" '
        f'text-anchor="middle">rows per {escape(granularity)}</text>'
        f'<text class="chart-label" x="{width - pad_r}" y="{height - 6}" '
        f'text-anchor="end">{escape(str(timeline.get("max", ""))[:10])}</text>'
        "</svg>"
    )
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
