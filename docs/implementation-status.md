# Implementation Status

Last updated: 2026-06-23 (signal-quality hardening pass)

This file is the living scope ledger. Update it whenever a capability is added,
removed, or materially re-scoped.

## Implemented

### Foundation

- Python 3.11+ package using Hatchling and a `src` layout
- DataFrame, CSV, Parquet, Excel (via the `excel` extra), path-list, named
  mapping, and directory loading
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
- Isolation Forest ranked review candidates with deterministic seed-stability
  disclosure
- Local Outlier Factor ranked review candidates where row count and
  dimensionality are suitable
- Detector agreement evidence across ranked anomaly review sets
- Optional expected-contamination parameter for review sizing
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
  highly predictive value rules (`deterministic_leakage_screen_v2`, reachable on
  imbalanced targets; near-perfect rules escalate to `critical`)
- Identifier-like features flagged for exclusion instead of generic
  high-cardinality risk; high-cardinality risk limited to categorical/text
- Leakage-screened logistic-regression diagnostic probe with fold-local
  preprocessing and cross-validated separability metrics
- Cross-validated hard-example candidates from probe errors
- Class-balance and feature-signal report artifacts

### Report quality

- Findings ordered by severity (`critical` > `high` > `medium` > `low`) across
  every recipe so reports lead with what blocks a decision
- Decision-first summaries (classification readiness verdict, top anomaly signal)
- Univariate anomaly tails gated on genuine extremity; conditional-anomaly
  findings capped to the strongest pairs
- One-to-one relationship candidates require key-name agreement, suppressing
  coincidental ID-range overlap; relationship titles name their tables/columns

### Privacy

- `PrivacyPolicy`/`ColumnPolicy` allow/redact/alias/exclude controls with keyed
  HMAC aliasing, now lint-clean, type-clean, and tested
- Wired into the AI-assisted layer: governs the dataset overview / schema
  description sent to a provider; raw values withheld by default

### AI-assisted investigation (`ai-gemini` extra)

- `assisted_analysis/` leaf package; the deterministic core imports no LLM library
- Provider-neutral `LLMProvider` interface with neutral request/decision types
- `GeminiProvider` over the `google-genai` SDK using a portable prompted-JSON
  protocol (works with Gemma and Gemini; default model `gemma-4-31b-it`)
- `FakeProvider` for deterministic, offline tests and docs
- Deterministic tool registry the model may call (`list_tables`, `describe_table`,
  `profile_dataset`, `discover_schema`, `detect_anomalies`,
  `assess_classification`); tools return compact summaries, never raw rows
- LangGraph flow: intake → bounded agent/tool loop → citation validation →
  synthesis; returns the standard `AnalysisResult`
- Evidence-citation validation drops any finding that doesn't cite real evidence;
  `insufficient_evidence` and non-convergence fallback handled
- Event emission through the existing callback system; report footer shows AI
  provenance
- Tests: FakeProvider flow, citation rejection, privacy, insufficient/unknown-tool
  paths, mocked-SDK provider, and core-import isolation (no LLM deps leak in)

### Engineering

- Ruff, mypy, pytest, coverage configuration, and CI for Python 3.11–3.13
- Wheel/sdist build and packaged-template verification
- CI wheel-install smoke test that imports the built wheel into a clean
  environment and renders a report from the packaged template
- Signal-quality regression tests (leakage on imbalanced targets, numeric vs
  high-cardinality, identifier exclusion, univariate/conditional gating,
  spurious one-to-one suppression) and edge-case tests (all-null, single-row,
  mixed dtype, single-class)
- Product research, architecture, maintainer, roadmap, handoff, and feature
  documentation

## Next

### Anomaly detection improvements

- Rare categorical combinations

### Classification improvements

- Class overlap and neighborhood-disagreement detection
- Group/time split guidance and opt-in fairness coverage
- Train/test comparison when both are supplied

## Later

- Regression, time-series, and clustering recipes
- Chunked CSV execution and a general execution planner
- Functional dependencies and denormalization analysis
- Plotly interactive artifact implementations
- Assisted-analysis follow-ups: critique/clarification nodes, an interactive
  question/answer loop, async `astart`/`arun`, and additional provider adapters
- Extending the privacy policy into a fuller model-payload builder (beyond the
  overview/schema description it governs today)
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
- Schema discovery can still propose a coincidental cross-named one-to-many
  relationship when two unrelated unique ID columns share a value range. The
  one-to-one gate is name-aware, but the one-to-many confidence model weights
  inclusion heavily; a proper fix needs join-cardinality/fan-out signals and
  ideally real labeled relationships rather than synthetic tuning.
