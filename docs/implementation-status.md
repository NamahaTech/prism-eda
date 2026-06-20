# Implementation Status

Last updated: 2026-06-20

This file is the living scope ledger. Update it whenever a capability is added,
removed, or materially re-scoped.

## Implemented

### Foundation

- Python 3.11+ package using Hatchling and a `src` layout
- DataFrame, CSV, Parquet, path-list, named mapping, and directory loading
- Optional recursive directory discovery
- Dataset and table fingerprints
- Physical and initial semantic column typing
- Framework-neutral events and callbacks
- Stable evidence IDs, findings, warnings, failures, sampling records, and
  artifacts
- Explicit JSON and self-contained HTML export
- Static responsive report design without CDNs or JavaScript requirements
- Declarative transformation recommendations

### Baseline profile

- Dataset/table shape and memory summaries
- Missingness, distinctness, duplicates, constants, top values, and basic robust
  numeric summaries
- Initial semantic-role candidates
- Prioritized findings with evidence lineage

### Schema discovery

- Minimal single-column and composite candidate keys
- Mode-based key-width and sampling budgets
- Typed/name-aware inclusion-dependency search
- One-to-one and one-to-many candidate cardinality
- Orphan and unreferenced-parent counts
- Candidate confidence and sampling disclosure
- Layered inline SVG ER diagram with entity cards, PK/FK roles, routed
  relationships, confidence badges, and one/many cardinality marks

### Anomaly detection

- Optional rare-label summary when a target is supplied
- Univariate robust numeric tail candidates using IQR and modified z-score style
  evidence
- Multivariate robust-scaled numeric candidate scores
- Conditional numeric anomaly candidates for surprising feature combinations
- Rare categorical value candidates
- Metric-table report artifact for candidate anomaly signals
- Evidence-linked findings and non-mutating review recommendations

### Classification

- Target validation, class counts, entropy, majority/minority rates, and
  imbalance ratio
- Duplicate feature signatures with conflicting labels
- Numeric target association using eta-squared
- Categorical target association using Cramer's V
- Class-conditional missingness gaps
- High-cardinality feature risk
- Deterministic leakage candidates from exact copies, target-name overlap, and
  highly predictive value rules
- Class-balance and feature-signal report artifacts

### Engineering

- Ruff, mypy, pytest, coverage configuration, and CI for Python 3.11–3.13
- Wheel/sdist build and packaged-template verification
- Product research, architecture, maintainer, roadmap, handoff, and feature
  documentation

## Next

### Anomaly detection improvements

- Isolation Forest multivariate candidates
- Local-density candidates where applicable
- Rare categorical combinations
- Detector agreement, stability, and per-row explanations
- Threshold-free ranked output with optional expected contamination

### Classification improvements

- Diagnostic probe models with cross-validation
- Class overlap and hard-example detection
- Group/time split guidance and opt-in fairness coverage
- Train/test comparison when both are supplied

## Later

- Regression, time-series, and clustering recipes
- Chunked CSV execution and a general execution planner
- Functional dependencies and denormalization analysis
- Plotly interactive artifact implementations
- Gemini-assisted investigation through LangChain and LangGraph
- Privacy policy and keyed-HMAC aliasing
- Persistent investigation checkpoints
- Additional DataFrame backends

## Known technical debt

- The shared report template will eventually benefit from recipe-specific partials
  as more report types are added.
- Baseline profiling is eager and should move behind reusable metric stages before
  large-scale chunked execution.
- Public result models use dataclasses; schema-version and migration policy must be
  defined before 1.0.
- Method-level performance benchmarks are not yet part of CI.
