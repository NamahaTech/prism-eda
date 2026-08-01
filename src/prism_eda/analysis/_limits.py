"""Caps that keep a profile report readable, and honest about what it cut.

A report of a 300-column table cannot show every chart, every correlation, and
every scatter pair without becoming both enormous and useless. These limits
decide what is worth the space. The rule that goes with them: whenever a limit
actually removes something, the profile records a warning and the report says so
on the page. A truncated report that looks complete is worse than a large one.
"""

from __future__ import annotations

from dataclasses import dataclass

from prism_eda.config import DetailLevel


@dataclass(frozen=True, slots=True)
class ProfileLimits:
    """Per-detail-level caps for the chart and association work."""

    #: Columns that get a distribution chart on their profile card.
    chart_columns: int
    #: Columns entering the association matrix (quadratic in this number).
    correlation_columns: int
    #: Numeric columns offered in the scatter explorer (quadratic in pairs).
    explorer_columns: int
    #: Highlighted scatter pairs shown by default.
    highlight_pairs: int
    #: Pairs pre-rendered into the scatter explorer. Every pair costs file size
    #: whether or not the reader opens it, so this is capped independently of
    #: ``explorer_columns``.
    explorer_pairs: int
    #: Points drawn in one highlighted scatter before deterministic sampling.
    scatter_points: int
    #: Points per explorer scatter. Lower than the highlights: dozens of these
    #: are embedded at once and their job is the shape, not every row.
    explorer_points: int
    #: Rows shown from the head and from the tail of each table.
    sample_rows: int
    #: Columns shown in the sample-rows table before it stops being readable.
    sample_columns: int
    #: Rows used when fitting a distribution family (fitting is the slow part).
    fit_rows: int
    #: Rows used for correlations and scatters. Pairwise association over
    #: millions of rows costs minutes and changes no conclusion.
    association_rows: int

    @classmethod
    def for_detail(cls, detail: DetailLevel | str) -> ProfileLimits:
        if detail == "full":
            return cls(
                chart_columns=250,
                correlation_columns=80,
                explorer_columns=20,
                highlight_pairs=12,
                explorer_pairs=190,
                scatter_points=10_000,
                explorer_points=1_000,
                sample_rows=25,
                sample_columns=60,
                fit_rows=50_000,
                association_rows=200_000,
            )
        return cls(
            chart_columns=60,
            correlation_columns=30,
            explorer_columns=10,
            highlight_pairs=6,
            explorer_pairs=45,
            scatter_points=2_000,
            explorer_points=300,
            sample_rows=10,
            sample_columns=30,
            fit_rows=20_000,
            association_rows=50_000,
        )
