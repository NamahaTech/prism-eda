"""Comparison result and export APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prism_eda._serialization import to_jsonable
from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog
from prism_eda.evidence.models import Evidence, Finding
from prism_eda.results import (
    AnalysisFailure,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    goal: str
    status: AnalysisStatus
    summary: str
    base_catalog: DatasetCatalog
    compare_catalog: DatasetCatalog
    findings: tuple[Finding, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    assumptions: tuple[str, ...] = ()
    warnings: tuple[AnalysisWarning, ...] = ()
    failures: tuple[AnalysisFailure, ...] = ()
    sampling: tuple[SamplingRecord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete machine-readable result."""
        return to_jsonable(self)

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        """Compatibility-friendly typed-model style serialization."""
        if mode != "json":
            raise ValueError("Prism EDA currently supports only mode='json'")
        return self.to_dict()

    def render_html(self) -> str:
        """Render a self-contained HTML report without writing to disk."""
        from prism_eda.reporting.renderer import render_comparison_html

        return render_comparison_html(self)

    def to_html(self, path: str | Path, *, interactive: bool = False) -> Path:
        """Write a self-contained HTML report and return its path."""
        target = Path(path)
        html = self.render_html()
        if interactive:
            try:
                import plotly  # type: ignore[import-not-found]  # noqa: F401
            except ImportError:
                html = html.replace(
                    "</body>",
                    '<div class="runtime-warning">Interactive charts were requested '
                    "but Plotly is not installed; static charts are "
                    "shown.</div></body>",
                )
        target.write_text(html, encoding="utf-8")
        return target

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the machine-readable result and return its path."""
        target = Path(path)
        target.write_text(
            json.dumps(self.to_dict(), indent=indent, sort_keys=True),
            encoding="utf-8",
        )
        return target
