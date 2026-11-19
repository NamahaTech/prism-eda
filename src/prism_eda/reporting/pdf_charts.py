"""Static charts for the PDF report, drawn with matplotlib.

Every function returns PNG bytes, or ``None`` when there is nothing worth
drawing. The look is fixed here once: thin marks, a recessive grid, text in
ink rather than in the data colour, one hue for magnitude, and the same few
categorical colours in the same order everywhere.
"""

from __future__ import annotations

import functools
import io
import math
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Categorical slots, used in this order and never cycled.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
# Neutrals.
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e7e6e2"
TRACK = "#dfe9f7"
SURFACE = "#ffffff"
# One hue, light to dark, for magnitude.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "magnitude", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

_PERIOD_RULE = {"hour": "h", "day": "D", "week": "W-MON", "month": "MS", "year": "YS"}
DPI = 150
#: Width of the report's text column in inches (A4 with 18 mm margins).
FULL_WIDTH = 6.9

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 6.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}


def _styled(function: Any) -> Any:
    """Draw the whole chart under the report's style, not just its creation."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with plt.rc_context(_RC):
            return function(*args, **kwargs)

    return wrapper


def _figure(width: float, height: float, **kwargs: Any) -> tuple[Figure, Any]:
    figure, axes = plt.subplots(figsize=(width, height), **kwargs)
    return figure, axes


def _png(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    with plt.rc_context(_RC):
        figure.savefig(
            buffer, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.06
        )
    plt.close(figure)
    return buffer.getvalue()


def compact(value: float, _position: Any = None) -> str:
    """1,284 → 1,284; 12,900 → 12.9K; 4,200,000 → 4.2M."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    magnitude = abs(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e4, "K")):
        if magnitude >= limit:
            scaled = value / (limit if suffix != "K" else 1e3)
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    if magnitude >= 100 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:.3g}"


def _shorten(text: Any, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------


@_styled
def horizontal_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "",
    value_format: Any = compact,
    color: str = BLUE,
    width: float = FULL_WIDTH,
    log: bool = False,
    label_limit: int = 38,
) -> bytes | None:
    if not labels:
        return None
    count = len(labels)
    figure, axis = _figure(width, max(1.1, 0.24 * count + 0.55))
    positions = np.arange(count)[::-1]
    plotted = [max(value, 0.8) if log else value for value in values]
    axis.barh(positions, plotted, height=0.62, color=color, linewidth=0)
    if log:
        axis.set_xscale("log")
    axis.set_yticks(positions)
    axis.set_yticklabels([_shorten(label, label_limit) for label in labels])
    axis.xaxis.set_major_formatter(FuncFormatter(compact))
    axis.grid(axis="y", visible=False)
    axis.spines["left"].set_visible(False)
    limit = axis.get_xlim()[1]
    for position, value, drawn in zip(positions, values, plotted, strict=True):
        axis.text(
            drawn * (1.08 if log else 1.0) + (0 if log else limit * 0.01),
            position,
            value_format(value),
            va="center",
            ha="left",
            fontsize=6.5,
            color=INK_2,
        )
    axis.set_xlim(right=limit * (3 if log else 1.12))
    if title:
        axis.set_title(title)
    return _png(figure)


@_styled
def bucket_bars(
    panels: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    width: float = FULL_WIDTH,
    height: float = 2.0,
) -> bytes | None:
    """Side-by-side bucketed count distributions (e.g. in- and out-degree)."""
    panels = [panel for panel in panels if panel[1]]
    if not panels:
        return None
    figure, axes = _figure(width, height, ncols=len(panels), squeeze=False)
    for axis, (title, buckets) in zip(axes[0], panels, strict=True):
        # Trailing empty buckets are just empty space on the right.
        last = max(
            (index for index, bucket in enumerate(buckets) if bucket["count"]),
            default=0,
        )
        shown = list(buckets[: last + 1])
        total = sum(bucket["count"] for bucket in shown) or 1
        shares = [bucket["count"] / total for bucket in shown]
        axis.bar(range(len(shown)), shares, width=0.7, color=BLUE, linewidth=0)
        axis.set_xticks(range(len(shown)))
        axis.set_xticklabels([bucket["label"] for bucket in shown], fontsize=5.6)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
        axis.grid(axis="x", visible=False)
        axis.set_title(title)
    figure.tight_layout()
    return _png(figure)


@_styled
def share_bar(
    segments: Sequence[tuple[str, float]], *, width: float = FULL_WIDTH
) -> bytes | None:
    """One whole split into two or three labelled parts."""
    segments = [(label, share) for label, share in segments if share > 0]
    if not segments:
        return None
    colors = [BLUE, TRACK, AQUA]
    figure, axis = _figure(width, 0.72)
    start = 0.0
    for index, (label, share) in enumerate(segments):
        axis.barh(
            0,
            share,
            left=start,
            height=0.5,
            color=colors[index % 3],
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        if share >= 0.08:
            axis.text(
                start + share / 2,
                0,
                f"{share:.0%}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#ffffff" if index % 3 != 1 else INK,
            )
        start += share
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.5, 0.5)
    axis.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[index % 3])
        for index in range(len(segments))
    ]
    axis.legend(
        handles,
        [f"{label} ({share:.1%})" for label, share in segments],
        loc="upper left",
        bbox_to_anchor=(0, -0.02),
        ncol=len(segments),
        handlelength=1,
        handleheight=1,
        columnspacing=1.6,
    )
    return _png(figure)


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


@_styled
def timeline(
    buckets: Sequence[dict[str, Any]],
    *,
    title: str = "",
    gaps: Sequence[dict[str, Any]] = (),
    granularity: str | None = None,
    width: float = FULL_WIDTH,
    height: float = 1.75,
) -> bytes | None:
    if len(buckets) < 2:
        return None
    series = pd.Series(
        [float(bucket["count"]) for bucket in buckets],
        index=pd.to_datetime([bucket["start"] for bucket in buckets]),
    ).sort_index()
    # Placeholder dates far from the real range flatten the chart; when they
    # stretch the axis badly, draw the window that holds the data and say so.
    total = series.sum()
    cumulative = series.cumsum() / total if total else series
    inside = series[(cumulative >= 0.002) & (cumulative <= 0.998)]
    trimmed = False
    if len(inside) >= 2:
        full = (series.index[-1] - series.index[0]).total_seconds()
        kept = (inside.index[-1] - inside.index[0]).total_seconds()
        if full > 0 and kept / full < 0.6:
            series, trimmed = inside, True
    # A period with no rows is absent from the counts; put it back as zero so
    # the line drops instead of bridging the hole.
    rule = _PERIOD_RULE.get(granularity or "")
    if rule and len(series) >= 2:
        expected = pd.date_range(series.index[0], series.index[-1], freq=rule)
        if 2 <= len(expected) <= 20_000:
            series = series.reindex(expected, fill_value=0.0)
    stamps, counts = series.index.to_numpy(), series.to_numpy()
    figure, axis = _figure(width, height)
    axis.fill_between(stamps, counts, color=BLUE, alpha=0.12, linewidth=0, step="mid")
    axis.plot(stamps, counts, color=BLUE, linewidth=1.3, drawstyle="steps-mid")
    for gap in gaps:
        axis.axvspan(
            np.datetime64(gap["start"]),
            np.datetime64(gap["end"]),
            color=ORANGE,
            alpha=0.16,
            linewidth=0,
        )
    if gaps:
        axis.plot([], [], color=ORANGE, alpha=0.4, linewidth=5, label="No rows")
        axis.legend(loc="lower right")
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_formatter(FuncFormatter(compact))
    axis.grid(axis="x", visible=False)
    note = "  (outlying dates not shown)" if trimmed else ""
    axis.set_title(f"{title}{note}" if title else note.strip())
    figure.autofmt_xdate(rotation=0, ha="center")
    return _png(figure)


@_styled
def coverage_chart(
    rows: Sequence[dict[str, Any]], *, width: float = FULL_WIDTH, limit: int = 40
) -> bytes | None:
    """The period each table covers, on one shared time axis."""
    spans = []
    for row in rows:
        try:
            start, end = np.datetime64(row["min"]), np.datetime64(row["max"])
        except (ValueError, TypeError):
            continue
        spans.append((f"{row['table']}  ·  {row['column']}", start, end))
    if not spans:
        return None
    spans.sort(key=lambda item: item[1])
    spans = spans[:limit]
    count = len(spans)
    figure, axis = _figure(width, max(1.2, 0.22 * count + 0.5))
    for index, (_label, start, end) in enumerate(spans):
        position = count - 1 - index
        if end == start:
            axis.plot([start], [position], marker="o", markersize=4, color=BLUE)
        else:
            axis.plot(
                [start, end],
                [position, position],
                color=BLUE,
                linewidth=5,
                solid_capstyle="round",
            )
    axis.set_yticks(range(count))
    axis.set_yticklabels([_shorten(label, 46) for label, _s, _e in reversed(spans)])
    axis.set_ylim(-0.7, count - 0.3)
    axis.grid(axis="y", visible=False)
    axis.spines["left"].set_visible(False)
    figure.autofmt_xdate(rotation=0, ha="center")
    return _png(figure)


@_styled
def weekday_hour_heatmap(
    grid: Sequence[Sequence[int]],
    weekdays: Sequence[str],
    *,
    title: str = "",
    width: float = FULL_WIDTH,
) -> bytes | None:
    array = np.asarray(grid, dtype="float64")
    if array.size == 0 or array.sum() == 0:
        return None
    figure, axis = _figure(width, 1.75)
    image = axis.imshow(array, aspect="auto", cmap=SEQUENTIAL, vmin=0)
    axis.set_xticks(range(0, 24, 2))
    axis.set_xticklabels([f"{hour:02d}" for hour in range(0, 24, 2)])
    axis.set_yticks(range(len(weekdays)))
    axis.set_yticklabels(weekdays)
    axis.set_xlabel("Hour of day")
    axis.grid(visible=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    # White seams separate the cells without drawing a border on any of them.
    axis.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(weekdays), 1), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=1.2)
    axis.tick_params(which="minor", length=0)
    bar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    bar.outline.set_visible(False)  # type: ignore[operator]
    bar.ax.yaxis.set_major_formatter(FuncFormatter(compact))
    bar.ax.tick_params(labelsize=6)
    if title:
        axis.set_title(title)
    return _png(figure)


# --------------------------------------------------------------------------
# Small multiples
# --------------------------------------------------------------------------


@_styled
def histogram_grid(
    panels: Sequence[dict[str, Any]], *, columns: int = 4, width: float = FULL_WIDTH
) -> bytes | None:
    """One small histogram per numeric column."""
    if not panels:
        return None
    rows = math.ceil(len(panels) / columns)
    figure, axes = _figure(
        width, 1.25 * rows + 0.1, nrows=rows, ncols=columns, squeeze=False
    )
    flat = axes.ravel()
    for axis, panel in zip(flat, panels, strict=False):
        counts = np.asarray(panel["counts"], dtype="float64")
        true_edges = np.asarray(panel["edges"], dtype="float64")
        log_scale = bool(panel.get("log")) and true_edges[0] > 0
        edges = np.log10(true_edges) if log_scale else true_edges
        axis.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=BLUE,
            linewidth=0.4,
            edgecolor=SURFACE,
        )
        name = _shorten(panel["name"], 22) + ("  (log scale)" if log_scale else "")
        axis.set_title(name, fontsize=7)
        axis.set_yticks([])
        axis.grid(visible=False)
        axis.spines["left"].set_visible(False)
        axis.set_xticks([edges[0], edges[-1]])
        axis.set_xticklabels([compact(true_edges[0]), compact(true_edges[-1])])
        labels = axis.get_xticklabels()
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
        median = panel.get("median")
        if median is not None and (not log_scale or median > 0):
            axis.axvline(
                np.log10(median) if log_scale else median, color=INK_2, linewidth=0.8
            )
    for axis in flat[len(panels) :]:
        axis.axis("off")
    figure.tight_layout(h_pad=0.9, w_pad=0.8)
    return _png(figure)


@_styled
def category_grid(
    panels: Sequence[dict[str, Any]], *, columns: int = 3, width: float = FULL_WIDTH
) -> bytes | None:
    """One small frequency bar chart per categorical column."""
    if not panels:
        return None
    bars = max(2, min(6, max(len(panel["counts"]) for panel in panels)))
    rows = math.ceil(len(panels) / columns)
    figure, axes = _figure(
        width,
        (0.2 * bars + 0.32) * rows + 0.1,
        nrows=rows,
        ncols=columns,
        squeeze=False,
    )
    flat = axes.ravel()
    for axis, panel in zip(flat, panels, strict=False):
        entries = list(panel["counts"][:bars])
        positions = np.arange(len(entries))[::-1]
        shares = [entry["rate"] for entry in entries]
        axis.barh(positions, shares, height=0.62, color=BLUE, linewidth=0)
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [_shorten(entry["value"], 14) for entry in entries], fontsize=6
        )
        axis.set_xlim(0, max(max(shares, default=0) * 1.3, 0.05))
        axis.set_ylim(len(entries) - bars - 0.6, len(entries) - 0.4)
        axis.set_xticks([])
        axis.grid(visible=False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_visible(False)
        for position, share in zip(positions, shares, strict=True):
            axis.text(
                share,
                position,
                f" {share:.0%}",
                va="center",
                ha="left",
                fontsize=6,
                color=INK_2,
            )
        extra = panel.get("other_category_count") or 0
        suffix = f"  (+{extra} more)" if extra else ""
        axis.set_title(_shorten(panel["name"], 22) + suffix, fontsize=7)
    for axis in flat[len(panels) :]:
        axis.axis("off")
    figure.tight_layout(h_pad=0.9, w_pad=1.0)
    return _png(figure)


@_styled
def association_heatmap(
    columns: Sequence[str], pairs: Sequence[dict[str, Any]], *, limit: int = 18
) -> bytes | None:
    """Pairwise association strength (0-1), one hue for magnitude."""
    if len(columns) < 3 or not pairs:
        return None
    strongest: dict[str, float] = {}
    for pair in pairs:
        for name in (pair["left"], pair["right"]):
            strongest[name] = max(strongest.get(name, 0.0), float(pair["strength"]))
    chosen = sorted(columns, key=lambda name: -strongest.get(name, 0.0))[:limit]
    chosen = [name for name in columns if name in set(chosen)]
    index = {name: position for position, name in enumerate(chosen)}
    size = len(chosen)
    matrix = np.full((size, size), np.nan)
    for pair in pairs:
        if pair["left"] in index and pair["right"] in index:
            a, b = index[pair["left"]], index[pair["right"]]
            matrix[a, b] = matrix[b, a] = float(pair["strength"])
    np.fill_diagonal(matrix, np.nan)
    side = min(FULL_WIDTH * 0.72, 0.3 * size + 1.6)
    figure, axis = _figure(side + 0.6, side)
    image = axis.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1)
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    axis.set_xticklabels(
        [_shorten(name, 16) for name in chosen], rotation=45, ha="right"
    )
    axis.set_yticklabels([_shorten(name, 20) for name in chosen])
    axis.grid(visible=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks(np.arange(-0.5, size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, size, 1), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=1.2)
    axis.tick_params(which="minor", length=0)
    if size <= 10:
        for a in range(size):
            for b in range(size):
                if not np.isnan(matrix[a, b]) and matrix[a, b] >= 0.3:
                    axis.text(
                        b,
                        a,
                        f"{matrix[a, b]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="#ffffff" if matrix[a, b] >= 0.55 else INK,
                    )
    bar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    bar.outline.set_visible(False)  # type: ignore[operator]
    bar.ax.tick_params(labelsize=6)
    return _png(figure)


# --------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------


@_styled
def reach_chart(
    hops: Sequence[dict[str, Any]], *, width: float = FULL_WIDTH
) -> bytes | None:
    """Two views of a walk: who gets anywhere, and how far."""
    if not hops:
        return None
    labels = [f"{hop['hop']} step{'s' if hop['hop'] > 1 else ''}" for hop in hops]
    figure, axes = _figure(width, 1.9, ncols=2)
    shares = [hop["seeds_reaching"] for hop in hops]
    axes[0].bar(labels, shares, width=0.5, color=BLUE, linewidth=0)
    axes[0].set_ylim(0, 1.12)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
    axes[0].grid(axis="x", visible=False)
    axes[0].set_title("Starting records that reach this far")
    for index, share in enumerate(shares):
        axes[0].text(
            index, share + 0.02, f"{share:.0%}", ha="center", fontsize=7, color=INK_2
        )
    medians = [max(hop["median_reach"], 0.5) for hop in hops]
    upper = [max(hop["p90_reach"], 0.5) for hop in hops]
    positions = np.arange(len(hops))
    axes[1].bar(
        positions - 0.19, medians, width=0.34, color=BLUE, linewidth=0, label="Median"
    )
    axes[1].bar(
        positions + 0.19,
        upper,
        width=0.34,
        color=AQUA,
        linewidth=0,
        label="90th percentile",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(FuncFormatter(compact))
    axes[1].grid(axis="x", visible=False)
    axes[1].set_title("Records reachable within this many steps")
    axes[1].legend(loc="upper left")
    figure.tight_layout()
    return _png(figure)


@_styled
def relationship_graph(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    *,
    width: float = FULL_WIDTH,
    limit: int = 28,
) -> bytes | None:
    """Tables on a circle, joined where their columns share identifiers."""
    edges = [edge for edge in edges if edge.get("tier") != "possible"]
    if not edges:
        return None
    degree: dict[str, int] = {}
    for edge in edges:
        for name in (edge["source"], edge["target"]):
            degree[name] = degree.get(name, 0) + 1
    names = sorted(degree, key=lambda name: (-degree[name], name))[:limit]
    kept = set(names)
    edges = [e for e in edges if e["source"] in kept and e["target"] in kept]
    count = len(names)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.pi / 2
    points = {
        name: (math.cos(angle), math.sin(angle))
        for name, angle in zip(names, angles, strict=True)
    }
    size = min(width, 5.2)
    figure, axis = _figure(size, size * 0.86)
    drawn: set[tuple[str, str]] = set()
    looped = {edge["source"] for edge in edges if edge["source"] == edge["target"]}
    for edge in edges:
        a, b = edge["source"], edge["target"]
        if a == b:
            x, y = points[a]
            loop = plt.Circle(
                (x * 1.09, y * 1.09), 0.09, fill=False, color=ORANGE, linewidth=1.4
            )
            axis.add_patch(loop)
            continue
        pair = (min(a, b), max(a, b))
        if pair in drawn:
            continue
        drawn.add(pair)
        (x1, y1), (x2, y2) = points[a], points[b]
        coverage = float(edge.get("coverage") or 0)
        axis.plot(
            [x1, x2],
            [y1, y2],
            color=BLUE,
            alpha=0.35 + 0.5 * coverage,
            linewidth=0.8 + 1.6 * coverage,
            solid_capstyle="round",
            zorder=1,
        )
    for name in names:
        x, y = points[name]
        axis.scatter(
            [x], [y], s=64, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3
        )
        outward = 1.3 if name in looped else 1.16
        axis.text(
            x * outward,
            y * outward,
            _shorten(name, 24),
            fontsize=6.8,
            color=INK,
            ha="left" if x > 0.15 else "right" if x < -0.15 else "center",
            va="bottom" if y > 0.15 else "top" if y < -0.15 else "center",
        )
    axis.plot(
        [],
        [],
        color=BLUE,
        linewidth=2,
        label="Shared identifiers (thicker = more covered)",
    )
    if any(edge["source"] == edge["target"] for edge in edges):
        axis.plot(
            [], [], color=ORANGE, linewidth=1.4, label="Links between rows of one table"
        )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1)
    axis.set_xlim(-1.75, 1.75)
    axis.set_ylim(-1.45, 1.45)
    axis.set_aspect("equal")
    axis.axis("off")
    return _png(figure)


@_styled
def log_histogram(
    counts: Sequence[int],
    edges: Sequence[float],
    *,
    title: str = "",
    width: float = FULL_WIDTH,
) -> bytes | None:
    """Histogram of a quantity binned on a log10 scale."""
    if not counts or sum(counts) == 0:
        return None
    figure, axis = _figure(width, 1.75)
    edges_array = np.asarray(edges, dtype="float64")
    axis.bar(
        edges_array[:-1],
        counts,
        width=np.diff(edges_array),
        align="edge",
        color=BLUE,
        linewidth=0.4,
        edgecolor=SURFACE,
    )
    ticks = np.arange(math.floor(edges_array[0]), math.ceil(edges_array[-1]) + 1)
    axis.set_xticks(ticks)
    axis.set_xticklabels([compact(10.0**tick) for tick in ticks])
    axis.yaxis.set_major_formatter(FuncFormatter(compact))
    axis.grid(axis="x", visible=False)
    axis.set_xlabel("Value (log scale)")
    if title:
        axis.set_title(title)
    return _png(figure)
