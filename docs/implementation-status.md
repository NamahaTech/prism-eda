# Implementation Status

Last updated: 2026-08-04 (regression readiness, time series)

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
- Missingness, distinctness, duplicates, constants, top values, and robust
  numeric summaries including quantiles, skewness, kurtosis, zero/negative/
  infinite counts, and per-column memory
- Initial semantic-role candidates
- Prioritized findings with evidence lineage, split into **issues** (data-quality
  defects) and **alerts** (true-but-not-broken observations) via
  `Finding.category`
- Data-quality detectors: case/whitespace label variants, disguised missing
  values (text placeholders and repeated negative sentinels), numbers stored as
  text, mixed Python types, text dates, mixed date layouts, ambiguous day/month
  order, placeholder and implausible dates, future-dated rows, reversed
  start/end date pairs, duplicate columns, unnamed columns, and mangled
  duplicate headers — all computed on every row, never sampled
- Per-column distribution shape labels, and best-fit distribution family
  (Normal, Log-normal, Exponential, Gamma, Weibull, Uniform, Beta, Poisson)
  selected by AIC with explicit abstention and no p-values
- Mixed-type association matrix (Spearman/Pearson, bias-corrected Cramér's V,
  correlation ratio) with per-pair method recorded
- Scatter interactions: ranked highlight pairs plus a pre-rendered explorer
- Missingness bars and pairwise co-missingness structure
- Head/tail sample rows and duplicate-row groups
- `detail="standard" | "full"` budgets, with every truncation reported as a
  warning and on the page

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
- Leakage-screened nearest-neighbor class-overlap candidates, with deterministic
  local label-disagreement review rows
- Context-aware group/time split guidance from `entity_id` and `timestamp`
- Class-balance and feature-signal report artifacts

### Regression

- `regression()` / `Dataset.regression()` public API for one numeric target
- Target summary, shape label, and `log1p`/`sqrt`/Yeo-Johnson candidates whose
  skew reduction is **measured on the data**, with abstention when none helps
- Value-spike scan for censoring, floors, defaults, and zero inflation in an
  otherwise continuous target; round-number heaping scan
- Feature association measured three ways (Pearson, Spearman, binned eta-squared)
  so a curved relationship is not reported as no relationship
- Redundant-pair detection with VIF and design condition number reported as
  measurements, never thresholded against a universal VIF cutoff
- Deterministic leakage screen: affine copies of the target, near-perfect
  univariate fit, and shared **name tokens** (never raw substrings)
- Leakage-screened cross-validated Ridge and Huber probes against a median
  baseline; the robust-versus-conventional gap on the typical row distinguishes
  weak features from a few distorting rows
- Residual shape with a KS *distance* and no p-value; binned residual spread with
  Breusch-Pagan; conditional bias per fitted decile
- Scale-normalized, peer-relative subgroup error concentration
- OLS leverage and Cook's distance, with a ranked review-row table carrying
  per-row reasons and robust feature deviations
- Weak-support scan for genuine holes in the target range and feature gaps
- Context-aware group/time split guidance from `entity_id` and `timestamp`
- Report sections (rows to review, residuals, target shape) plus new
  residual-scatter and diverging conditional-bias charts
- `assess_regression` registered in the assisted-analysis tool registry

### Time series

- `time_series()` / `Dataset.time_series()` public API taking `value`, optional
  `timestamp` (inferred when unambiguous), `entity_id`, and `horizon`
- Frequency inference from the modal gap, tolerant of gaps and duplicates where
  `pd.infer_freq` returns `None`; calendar frequencies matched by range
- Hygiene checks computed on the **raw** rows, structural analysis on a
  regularized reconstruction disclosed as a `SamplingRecord` plus a warning
- Entity-aware duplicate-timestamp detection, with conflicting values counted
- Unrecorded periods and blank periods reported separately, as contiguous blocks
- Irregular-spacing scan against the dominant interval
- Per-entity panel coverage, absolute seasonal-history floor, relative imbalance,
  and **panel composition changes** so a total whose membership changes is not
  read as a change in demand
- STL decomposition with variance-share trend and seasonal strength, plus the
  day-of-week seasonal profile
- ACF/PACF with confidence band and candidate seasonal periods from local peaks
- ADF **and** KPSS with the four-way agreement/disagreement classification
  reported rather than resolved to one verdict
- Level and variance change points via Theil-Sen-detrended binary segmentation,
  gated on the step exceeding the series' own noise
- Temporal outliers on the STL remainder against an interquartile fence, with
  interpolated periods and change-point neighbourhoods excluded, and the list
  suppressed when the flag rate indicates changing spread rather than anomalies
- Syntetos-Boylan intermittent-demand classification, only when zeros are present
- History-versus-horizon adequacy and an expanding-window backtest plan
- Leakage-safe lagged cross-correlation, marking which lags are usable at
  forecast time
- Report sections (the series, trend and seasonality, memory and stationarity,
  coverage) with new series-line, ACF-stem, and seasonal-profile charts
- `analyze_time_series` registered in the assisted-analysis tool registry

### Image dataset profile

- `ImageDataset`, `load_images()`, and `profile_images()` public API
- Recursive image-folder discovery with glob-style include/exclude filters
- Directory-derived labels *and* splits (`root/split/label/file`), so the split
  folder is not mistaken for a class
- Decode validation with unreadable/corrupt file evidence
- Dimension, aspect-ratio, megapixel, format, mode, animation, EXIF, and
  orientation summaries
- Exact duplicate groups via SHA-256 file hashes
- Perceptual near-duplicate candidates via average/difference image hashes, with
  deterministic hash-window blocking for larger analyzed sets
- Train/validation/test split leakage and conflicting-label detection from
  duplicate groups that cross a cohort boundary
- Loader traps: EXIF rotation (including orientations that transpose width and
  height), extension/encoding mismatch, greyscale stored in colour mode, used
  alpha channels, and truncated-but-decodable files
- Lightweight visual-quality flags for very dark, very bright, low-contrast,
  low-sharpness, and low-entropy imagesw
- Robust outlier candidates for file size, resolution, and aspect ratio, with a
  mean-absolute-deviation fallback so uniform datasets still surface odd files
- Per-label dimension and brightness profiles, plus deviating-label findings
- Embedded base64 thumbnail contact sheets for flagged images (duplicates shown
  as pairs), a width-against-height scatter, a brightness histogram, and class
  balance bars — all offline and self-contained; `thumbnails=False` opts out
- Label imbalance findings, metric-table artifacts, HTML/JSON export, sampling
  disclosure, and evidence-linked review recommendations

### Report quality

- Findings ordered by severity (`critical` > `high` > `medium` > `low`) across
  every recipe so reports lead with what blocks a decision
- Decision-first summaries (classification readiness verdict, top anomaly signal)
- Univariate anomaly tails gated on genuine extremity; conditional-anomaly
  findings capped to the strongest pairs
- One-to-one and one-to-many relationship candidates require key-name agreement
  or strong parent-coverage, suppressing coincidental ID-range overlap;
  relationship titles name their tables/columns

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

- Opt-in fairness coverage
- Train/test comparison when both are supplied

### Time-series improvements

- Per-entity structural analysis for panels, capped, alongside the aggregate
- Multiple seasonal periods at once (weekly *and* yearly on daily data)
- Holiday and calendar-effect regressors as candidate explanations for outliers
- Sub-daily timezone and DST handling beyond reporting the index timezone

### Regression improvements

- Quantile-regression probe for targets where the conditional median is the
  quantity of interest
- Interaction screening, so a group-specific slope is distinguished from a
  group-specific intercept
- Partial-dependence style summaries for the strongest non-linear features

### Image dataset improvements

- Optional deep visual embeddings for stronger near-duplicate and semantic
  outlier review (two photographs of the same object are not caught by
  perceptual hashes)
- Label-file joins for COCO/YOLO/classification manifests, so leakage and
  per-label checks work without a directory layout
- Per-channel mean/std normalization constants
- Domain-specific image quality profiles for OCR, medical imaging, and remote
  sensing

## Later

- Clustering recipe
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
