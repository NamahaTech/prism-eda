"""Which sections a report contains, in order.

The template used to number its sections with a running counter and decide what
to render with conditions spread through the markup. Once the report grew a
navigation bar those two had to agree, and a counter cannot be read ahead of
time. So the decision moved here: this module answers "which sections does this
result have, in what order, numbered how", and the template renders from that
answer. One list drives the nav, the anchors, and the numbering, so a section
cannot appear in the nav and be missing from the page.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_eda.results import AnalysisResult


@dataclass(frozen=True, slots=True)
class ReportSection:
    """One numbered, linkable section of a report."""

    id: str
    label: str
    number: int

    @property
    def anchor(self) -> str:
        return f"section-{self.id}"

    @property
    def display_number(self) -> str:
        return f"{self.number:02d}"


class SectionIndex:
    """The ordered sections of one report, addressable by id from the template."""

    def __init__(self, sections: list[ReportSection]) -> None:
        self._sections = tuple(sections)
        self._by_id = {section.id: section for section in sections}

    def __iter__(self) -> Iterator[ReportSection]:
        return iter(self._sections)

    def __len__(self) -> int:
        return len(self._sections)

    def __contains__(self, section_id: object) -> bool:
        return section_id in self._by_id

    def number(self, section_id: str) -> str:
        section = self._by_id.get(section_id)
        return section.display_number if section else ""

    def anchor(self, section_id: str) -> str:
        return f"section-{section_id}"

    def label(self, section_id: str) -> str:
        section = self._by_id.get(section_id)
        return section.label if section else ""


def _has_evidence(result: AnalysisResult, kind: str) -> bool:
    return any(item.kind == kind for item in result.evidence)


def _metric_tables(
    result: AnalysisResult, *, skip_linked: bool = False
) -> list[tuple[str, str]]:
    """Metric tables that get a section of their own.

    ``skip_linked`` mirrors the image report, which renders a metric table
    inside the finding that cites it rather than twice: once in the finding and
    again as a standalone section.
    """
    cited: set[str] = set()
    if skip_linked:
        for finding in result.findings:
            cited.update(finding.evidence_ids)
    return [
        (f"metrics-{index}", artifact.title)
        for index, artifact in enumerate(result.artifacts)
        if artifact.kind == "metric_table"
        and not (skip_linked and cited.intersection(artifact.evidence_ids))
    ]


def report_sections(result: AnalysisResult) -> SectionIndex:
    """Build the ordered section list for this result."""
    from prism_eda.evidence.models import split_findings

    goal = result.goal
    issues, observations = split_findings(result.findings)
    entries: list[tuple[str, str]] = []

    if result.warnings:
        entries.append(("warnings", "Warnings"))

    if goal == "schema_discovery":
        if any(artifact.kind == "schema_graph" for artifact in result.artifacts):
            entries.append(("erd", "Diagram"))
        entries.append(("keys", "Keys"))
        entries.append(("relationships", "Relationships"))

    # Image, anomaly, and regression reports defer their reference tables until
    # after the findings; every other goal leads with them. For regression the
    # probe scores and the VIF table are the working out, not the verdict.
    if goal not in {"anomaly_detection", "image_profile", "regression"}:
        entries.extend(_metric_tables(result))

    entries.append(("issues", "Issues" if goal == "profile" else "Findings"))
    if observations:
        entries.append(("alerts", "Alerts"))

    if goal == "anomaly_detection" and any(
        item.kind == "anomaly_consensus_review" and item.value.get("rows")
        for item in result.evidence
    ):
        entries.append(("rows", "Rows to review"))

    if goal == "regression":
        # Ordered the way the report is argued: which rows to open, then what
        # the residuals say about the fit, then the target's own shape, and
        # only then the reference tables behind all three.
        if any(
            item.kind == "regression_review_rows" and item.value.get("rows")
            for item in result.evidence
        ):
            entries.append(("rows", "Rows to review"))
        if _has_evidence(result, "regression_residual_scatter") or _has_evidence(
            result, "regression_conditional_bias"
        ):
            entries.append(("residuals", "Residuals"))
        if _has_evidence(result, "regression_target_shape"):
            entries.append(("target", "Target shape"))
        entries.extend(_metric_tables(result))

    if goal == "image_profile":
        if any(artifact.kind == "image_contact_sheet" for artifact in result.artifacts):
            entries.append(("flagged-images", "Flagged images"))
        if result.metadata.get("image_count"):
            entries.append(("image-shape", "Shape and exposure"))
        entries.extend(_metric_tables(result, skip_linked=True))

    columns = [
        (f"columns-{index}", table.name)
        for index, table in enumerate(result.catalog.tables)
    ]
    if goal == "profile":
        # The profile is read column-first: what is in here, then how the
        # columns relate. Other goals lead with their own analysis and keep the
        # per-column detail as reference at the end.
        entries.extend(columns)
        if _has_evidence(result, "profile_association_matrix"):
            entries.append(("correlations", "Correlations"))
        if _has_evidence(result, "profile_scatter"):
            entries.append(("interactions", "Interactions"))
        if any(
            item.kind == "profile_missingness"
            and any(row["missing_count"] for row in item.value["columns"])
            for item in result.evidence
        ):
            entries.append(("missing", "Missing values"))
        if _has_evidence(result, "profile_sample_rows"):
            entries.append(("sample", "Sample rows"))

    if goal == "anomaly_detection":
        if _has_evidence(result, "anomaly_distribution_shape"):
            entries.append(("distributions", "Distributions"))
        if any(
            item.kind == "anomaly_scatter_pair" and item.value.get("points")
            for item in result.evidence
        ):
            entries.append(("scatter", "Context"))
        entries.extend(_metric_tables(result))

    if goal != "profile":
        entries.extend(columns)

    if result.transformation_plan.steps:
        entries.append(("plan", "Plan"))

    return SectionIndex(
        [
            ReportSection(id=section_id, label=label, number=index)
            for index, (section_id, label) in enumerate(entries, start=1)
        ]
    )
