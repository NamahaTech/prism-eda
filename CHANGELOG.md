# Changelog

All notable changes to Prism EDA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow semantic versioning once its public API
stabilizes.

## [Unreleased]

### Added

- Initial `prism_eda` package and session-based public API.
- DataFrame, CSV, Parquet, multi-table mapping, and directory loading.
- Deterministic dataset fingerprints and baseline table/column catalogs.
- Evidence-linked findings and declarative transformation recommendations.
- Self-contained Jinja2 HTML reports and machine-readable JSON exports.
- Framework-neutral progress and lifecycle events.
- Minimal single-column and composite candidate-key discovery.
- Typed, name-aware candidate foreign-key relationships with orphan analysis.
- Structured report artifacts and a self-contained candidate schema graph.
- Layered ER diagram rendering with table cards, inferred PK/FK roles, routed
  relationships, confidence badges, and one/many cardinality marks.
- Deterministic anomaly-detection diagnostics with univariate, multivariate,
  conditional, rare-category, and optional label-summary evidence.
- Isolation Forest, Local Outlier Factor, detector-agreement, and seed-stability
  anomaly evidence with optional expected-contamination review sizing.
- Deterministic classification diagnostics with target balance, association,
  missingness-by-class, high-cardinality, conflicting-label, and leakage
  evidence.
- Leakage-screened classification probe diagnostics with fold-local
  preprocessing, cross-validated separability metrics, and hard-example review
  candidates.
- Generic metric-table artifacts in HTML reports.
- Detailed implementation plan and roadmap handoff documentation.
