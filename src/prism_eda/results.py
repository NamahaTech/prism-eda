"""Analysis result and export APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prism_eda._compat import StrEnum
from prism_eda._serialization import to_jsonable
from prism_eda.artifacts import Artifact
from prism_eda.branding import BrandingLike
from prism_eda.catalog.models import DatasetCatalog
from prism_eda.evidence.models import Evidence, Finding
from prism_eda.transformations.models import TransformationPlan


class AnalysisStatus(StrEnum):
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_MEANINGFUL_STRUCTURE = "no_meaningful_structure"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AnalysisWarning:
    code: str
    message: str
    table: str | None = None
    column: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisFailure:
    stage: str
    message: str
    recoverable: bool
    table: str | None = None
    column: str | None = None


@dataclass(frozen=True, slots=True)
class SamplingRecord:
    operation: str
    source_rows: int
    sampled_rows: int
    strategy: str
    seed: int
    reason: str
    limitations: tuple[str, ...] = ()
    user_override: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    goal: str
    status: AnalysisStatus
    summary: str
    catalog: DatasetCatalog
    findings: tuple[Finding, ...]
    evidence: tuple[Evidence, ...]
    artifacts: tuple[Artifact, ...] = ()
    assumptions: tuple[str, ...] = ()
    warnings: tuple[AnalysisWarning, ...] = ()
    failures: tuple[AnalysisFailure, ...] = ()
    sampling: tuple[SamplingRecord, ...] = ()
    transformation_plan: TransformationPlan = field(default_factory=TransformationPlan)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete machine-readable result."""
        return to_jsonable(self)

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        """Compatibility-friendly typed-model style serialization."""
        if mode != "json":
            raise ValueError("Only mode='json' is supported")
        return self.to_dict()

    def render_html(
        self, *, branding: BrandingLike = None, title: str | None = None
    ) -> str:
        """Render a self-contained HTML report without writing to disk.

        ``branding=False`` renders the report with no product name, logo,
        favicon, or footer; a :class:`~prism_eda.branding.Branding` supplies a
        custom identity; ``None`` uses the process default.
        """
        from prism_eda.reporting.renderer import render_html

        return render_html(self, branding=branding, title=title)

    def to_html(
        self,
        path: str | Path,
        *,
        branding: BrandingLike = None,
        title: str | None = None,
    ) -> Path:
        """Write a self-contained HTML report and return its path."""
        target = Path(path)
        target.write_text(
            self.render_html(branding=branding, title=title), encoding="utf-8"
        )
        return target

    def render_pdf(
        self,
        *,
        branding: BrandingLike = None,
        title: str | None = None,
        subtitle: str | None = None,
        detailed_tables: int = 80,
    ) -> bytes:
        """Render a PDF report and return its bytes (needs the ``pdf`` extra).

        ``title`` and ``subtitle`` replace the cover's defaults. ``branding``
        works as it does for HTML. ``detailed_tables`` caps how many tables
        (largest first) get charts; the rest keep their column statistics.
        """
        from prism_eda.reporting.pdf import render_pdf

        return render_pdf(
            self,
            branding=branding,
            title=title,
            subtitle=subtitle,
            detailed_tables=detailed_tables,
        )

    def to_pdf(
        self,
        path: str | Path,
        *,
        branding: BrandingLike = None,
        title: str | None = None,
        subtitle: str | None = None,
        detailed_tables: int = 80,
    ) -> Path:
        """Write a PDF report and return its path (needs the ``pdf`` extra)."""
        target = Path(path)
        target.write_bytes(
            self.render_pdf(
                branding=branding,
                title=title,
                subtitle=subtitle,
                detailed_tables=detailed_tables,
            )
        )
        return target

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the machine-readable result and return its path."""
        target = Path(path)
        target.write_text(
            json.dumps(self.to_dict(), indent=indent, sort_keys=True),
            encoding="utf-8",
        )
        return target
