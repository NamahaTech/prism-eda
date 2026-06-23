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
- Identifier-like classification features are now flagged for exclusion instead
  of being mislabeled as generic high-cardinality risks.
- Findings are now ordered by severity (`critical` first) across every recipe so
  reports lead with what blocks a decision.
- Decision-first summaries for classification ("not ready to model …") and
  anomaly detection (top candidate signal) instead of a raw finding count.
- `critical` finding severity and report badge for confirmed-style leakage.
- Privacy `PrivacyPolicy`/`ColumnPolicy` controls are now tested.

### Fixed

- Target-leakage detection no longer misses near-perfect value rules on
  imbalanced targets. The screen previously required an accuracy above
  `majority_rate + 0.15`, which exceeds 1.0 for imbalanced data and made the
  most common leakage case undetectable (`deterministic_leakage_screen_v2`).
- High-cardinality risk is no longer reported for ordinary numeric columns; only
  genuine categorical/text columns can carry encoding-cardinality risk.
- Univariate anomaly tails only become findings when a value is genuinely
  extreme or the tail is heavy, instead of flagging the ordinary tail every
  numeric column has.
- Conditional-anomaly findings are capped to the strongest pairs rather than
  emitting one per ordered feature combination.
- One-to-one relationship candidates now require real key-name agreement, so
  coincidental ID-range overlap between unrelated unique columns is suppressed.
- Relationship finding titles now name the participating tables and columns.
- Fixed lint and type errors in the privacy module.
