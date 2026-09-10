"""The result of planning and running a feature set.

A sibling of :class:`~prism_eda.comparison_results.ComparisonResult` rather
than an :class:`~prism_eda.results.AnalysisResult`: this capability produces
engineered data as well as an account of it, so it does not pretend to be one
of the analysis recipes. It reuses the same evidence, finding and artifact
contracts, so lineage discipline and the issue/observation split carry over
unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from prism_eda._serialization import to_jsonable
from prism_eda.artifacts import Artifact
from prism_eda.evidence.models import Evidence, Finding, split_findings
from prism_eda.features.contract import FeatureContract
from prism_eda.results import AnalysisStatus, AnalysisWarning, SamplingRecord

__all__ = ["FeaturePlanResult", "FeatureRun", "build_summary"]


@dataclass(frozen=True, slots=True)
class FeaturePlanResult:
    """What the plan produced, and what is worth knowing about it."""

    goal: str
    status: AnalysisStatus
    summary: str
    entity: str
    contract: FeatureContract
    time: str | None = None
    reference_time: str | None = None
    plan_description: str = ""
    findings: tuple[Finding, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    assumptions: tuple[str, ...] = ()
    warnings: tuple[AnalysisWarning, ...] = ()
    sampling: tuple[SamplingRecord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def issues(self) -> tuple[Finding, ...]:
        """Findings that report something wrong."""
        return tuple(split_findings(self.findings)[0])

    @property
    def observations(self) -> tuple[Finding, ...]:
        """Findings that report something true but not broken."""
        return tuple(split_findings(self.findings)[1])

    def to_dict(self) -> dict[str, Any]:
        """The complete machine-readable result."""
        payload = to_jsonable(self)
        payload["contract"] = self.contract.to_dict()
        return payload

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        """Compatibility-friendly typed-model style serialization."""
        if mode != "json":
            raise ValueError("Prism EDA currently supports only mode='json'")
        return self.to_dict()

    def render_html(self) -> str:
        """Render a self-contained HTML report without writing to disk."""
        from prism_eda.reporting.renderer import render_feature_plan_html

        return render_feature_plan_html(self)

    def to_html(self, path: str | Path) -> Path:
        """Write a self-contained HTML report and return its path."""
        target = Path(path)
        target.write_text(self.render_html(), encoding="utf-8")
        return target

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the machine-readable result and return its path."""
        target = Path(path)
        target.write_text(
            json.dumps(self.to_dict(), indent=indent, sort_keys=True),
            encoding="utf-8",
        )
        return target


@dataclass(frozen=True, slots=True)
class FeatureRun:
    """The engineered features, and the report about them.

    Returned together because computing the report means computing the
    features; handing back only one would mean doing the work twice.
    """

    features: pd.DataFrame
    report: FeaturePlanResult


def build_summary(
    *,
    feature_count: int,
    entity_count: int,
    findings: tuple[Finding, ...],
    shared_nodes: int,
    total_nodes: int,
    has_warnings: bool,
) -> str:
    """Lead with the verdict, then what would change it."""
    issues, observations = split_findings(findings)
    sharing = (
        f"{shared_nodes} of {total_nodes} operations are shared between features"
        if total_nodes
        else "no operations to share"
    )
    head = (
        f"{feature_count} feature(s) computed for {entity_count:,} entities; {sharing}."
    )

    if not issues:
        verdict = " Nothing in the definitions or the results looks wrong."
    else:
        counted = []
        for severity in ("critical", "high", "medium", "low"):
            number = sum(1 for item in issues if item.severity == severity)
            if number:
                counted.append(f"{number} {severity}")
        verdict = f" {len(issues)} issue(s) ({', '.join(counted)})."
        verdict += f" Top issue — {issues[0].title}."

    if observations:
        verdict += f" {len(observations)} alert(s)."
    if has_warnings:
        verdict += " Sampling or recoverable caveats apply."
    return head + verdict
