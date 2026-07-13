# Changelog

All notable changes to Prism EDA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow semantic versioning once its public API
stabilizes.

## [Unreleased]

### Added

- **Image dataset profiling.** New `ImageDataset`, `load_images()`, and
  `profile_images()` APIs profile image folders without forcing them through the
  tabular loader. The recipe reports unreadable files, dimensions and
  aspect-ratio spread, formats/modes/animation, EXIF/orientation presence,
  directory-label balance, exact SHA-256 duplicates, perceptual near-duplicate
  candidates, file-size outliers, and lightweight visual-quality flags for dark,
  bright, low-contrast, low-sharpness, and low-entropy images. Results use the
  standard `AnalysisResult`, evidence lineage, metric-table artifacts,
  transformation recommendations, sampling records, JSON export, and
  self-contained HTML report.
- **Train/test leakage detection for images.** Labels *and* splits are inferred
  from the `root/split/label/file` layout, so a duplicate or near-duplicate that
  crosses a split is reported as a `critical` finding — it inflates every metric
  you report without changing the model. The same check across labels catches a
  single image filed under two classes.
- **Loader traps.** Images that decode cleanly but reach a pipeline changed:
  non-default EXIF orientation (orientations 5-8 also swap width and height),
  file extensions that disagree with the actual encoding, greyscale images stored
  in three identical colour channels, used alpha channels, and truncated files —
  which are now profiled and reported rather than discarded as unreadable.
- **Per-label breakdown.** Dimension and brightness statistics are computed per
  label, and labels that do not look like the rest of the dataset are called out,
  so collection bias in one class is not averaged away.
- **The image report shows the images.** Flagged files are rendered as embedded
  base64 thumbnail contact sheets, with duplicate candidates paired side by side,
  alongside a width-against-height scatter, a brightness histogram, and class
  balance bars. Reports stay single, portable, offline files. `thumbnails=False`
  turns the pictures off without changing the findings. Thumbnails are stored in
  artifacts, never in evidence, and image datasets are not exposed to the
  optional AI layer: raw pixels never reach a model provider.

### Fixed

- Image profiling no longer reports a readable file as corrupt when Pillow merely
  *warns* about it. Pillow warns rather than raises on recoverable defects such
  as a corrupt EXIF block, and those warnings were being turned into decode
  failures, which would have called ordinary JPEGs unreadable.
- Robust outlier scoring for images no longer goes blind on a uniform dataset.
  When most images share one size the MAD and the IQR are both zero, and the
  previous scale returned no outliers at all — hiding the single panorama among
  the thumbnails in exactly the case where it matters most. The scale now falls
  back to the mean absolute deviation.
- **Interactive ER diagram.** The schema-discovery report's ERD is now a real
  interactive diagram (embedded, vendored Cytoscape.js — reports stay fully
  offline): drag table cards to rearrange with edges following, smooth
  scroll-zoom and pan, click a table to focus its relationships, click an edge
  for cardinality/confidence detail, and toggle tables to declutter.
  Relationship endpoints carry explicit `1`/`N` labels, and the legend now has
  a dedicated cardinality group explaining the notation. With JavaScript
  unavailable the report degrades to the previous static SVG with a visible
  notice.
- **Categorical columns show their categories.** The column profile's
  "Range / shape" cell now renders the top values of categorical and boolean
  columns with their share of rows (plus a "+N more · top 5 cover X% of rows"
  note), instead of the unhelpful "No numeric range".
- **Column warnings are visible in reports.** `ColumnCatalog.warnings`
  (half-missing, constant, and the new high-cardinality warning) render as
  amber chips next to the column's type.

### Changed

- **Categorical inference has an absolute cardinality cap.** A text column is
  categorical when it has ≤ 50 distinct values, or ≤ 5% distinct up to a cap
  of 200 distinct values. Previously the 5% rule alone could label name-like
  columns with tens of thousands of distinct values as categorical on large
  tables; such columns are now `text`. Columns that stay categorical with
  more than 100 distinct values get a high-cardinality warning.
- Polished the report logo into a geometrically exact prism mark and added it
  as the report favicon.

- **Row-centric anomaly review.** Anomaly detection now leads with the rows
  themselves instead of one finding per detector. A new cross-detector
  *consensus* (`anomaly_consensus_review` evidence) ranks rows by how many
  independent checks agree, and carries each flagged row's actual values, the
  per-column contribution vs. the typical baseline, and a plain-language *why*
  ("Salary 10,000,000 is 40× the typical 250,000"; conditional anomalies read as
  "unusual for its peer group"). The redundant per-detector findings
  (multivariate / Isolation Forest / LOF / conditional / agreement) now feed the
  consensus rather than each becoming its own finding.
- **AI investigator now adds a real interpretation layer — not a reword.** After
  the tool loop gathers deterministic evidence, a new grounded *interpretation*
  pass asks the model for the judgment statistics can't supply: a plain-English
  **semantic read of each column** (meaning / likely unit / caveat), **business-term
  naming of inferred relationships** (schema goal — e.g. *"Each order is placed by
  one customer"*), a **root-cause narrative** tying the findings together, and
  **prioritized next steps**. Each is a small focused prompt, reasoned in prose (JSON-mode degrades
  small models), grounded only in the gathered evidence and privacy-gated column
  aggregates, and allowed to **abstain** rather than fabricate — column reads that
  name a column we never disclosed are dropped. It renders as a visibly distinct
  "AI interpretation" panel so the added value is obvious versus the deterministic
  sections. Providers gained a `respond()` text endpoint (`FakeProvider` is
  scriptable via `responses=`); a provider without one simply yields no layer.
- **Opt-in richer LLM context.** The interpretation pass sends column *aggregates*
  (type, range, missingness, cardinality) always, but the actual category
  **labels/value samples** only when `PrivacyPolicy(allow_raw_values=True)` — off
  by default, preserving the strict "no raw values to the LLM" boundary unless you
  opt in.
- **Schema report overhaul — from a flat dump to a navigable map.** The schema
  graph now runs a lightweight FK-graph analysis: each table is classified
  (hub / dimension / fact / junction / bridge) from its referenced-by and
  references degree, and PageRank ranks structural importance — so the report
  leads with a synthesized **verdict** ("X and Y are the hubs — referenced by N
  of M tables") and role-tagged summary chips instead of an undifferentiated
  list. The **ER diagram is now interactive** (progressive-enhancement vanilla
  JS, still a single self-contained file with a static fallback): scroll/pinch to
  zoom (zoom-to-cursor), drag to pan, fit-to-screen, click a table to focus and
  dim the rest, toggle tables to declutter, and click "+N more" to open a popover
  with the full inferred-key-role list. Edges encode confidence (dashed, weight
  binned high/medium/low) with the evidence breakdown in the tooltip. The
  **candidate-relationships** table gained parent/child dropdown filters, a
  confidence sort, confidence badges, and a sticky header so a large schema's
  relationships are readable by pair instead of one long scroll. The whole report
  is now responsive.
- **Report recipe indicator.** The masthead's misleading Profile/Schema/…
  tab strip (which implied a multi-page app) is replaced by a single non-clickable
  recipe pill naming the current report.
- **Per-method evidence behind every review row.** Each consensus row now
  carries an `explanations` block so a *multivariate* tag is backed by the full
  per-column joint-deviation profile (every column's distance from its own
  typical value, not just the dominant spike) and a *conditional* tag by the
  peer group it stands out from — the condition bin, the peer band (median and
  middle 50%), and where the row's own value sits. The conditional detector's
  evidence was extended with that peer-band context. The report renders these as
  a per-column σ chart and a peer-group strip, so a "multivariate outlier" is
  actually shown to be multivariate instead of collapsing to one univariate bar.
- **Synthesized verdict headline** (`metadata["verdict"]`): one plain-language
  sentence leading with the strongest reframing — a two-population split when
  present, otherwise the row-review count with a concrete example — surfaced as
  the report's hero.
- **Redesigned HTML report.** A refraction-themed design system (verdict-led
  hero, dataset stat strip, severity-ruled findings with confidence meters,
  collapsible deep sections, self-contained system-font typography) and an
  Agreement filter (All / 2+ / 3+ / 4+ checks) on the review table via a small
  amount of progressive-enhancement inline JavaScript — the file stays a single
  self-contained document and the rows are server-rendered, so it degrades
  cleanly with JavaScript disabled.
- **Distribution-shape diagnostics** (`anomaly_distribution_shape` evidence):
  per-column histogram and box summary, plus a robust two-population (regime
  split) detector that reports "looks like two populations" instead of
  mislabelling a bimodal column's upper cluster as tail outliers.
- **Inline-SVG charts in the HTML report** (no JavaScript, fully self-contained):
  per-column histograms with box strips and flagged values marked, a scatter of
  the most relevant numeric pair with flagged rows highlighted, per-row "why"
  bars, and a flagged-rows table with the unusual cells highlighted.
- Finding *summaries* now carry only counts and rates — never raw cell values —
  so the AI-assisted investigator can forward them to an LLM without leaking
  data; exact values stay in evidence and the locally-rendered charts. The
  investigator prompt also forbids speculating about columns or relationships no
  tool surfaced.
- **Excel input support** (`.xlsx`, `.xlsm`, `.xls`) for file, list, mapping, and
  directory loading. The Excel engine is an optional extra
  (`pip install "prism-eda[excel]"`); loading an Excel file without it raises a
  clear `DataLoadError`, so CSV/Parquet users need no extra dependency. The first
  sheet is read by default; choose another with
  `read_options={"excel": {"sheet_name": ...}}`.
- **Optional AI-assisted investigation** (`prism_eda.assisted_analysis`, behind
  the `ai-gemini` extra): an LLM plans and explains an analysis by calling only
  Prism's deterministic tools. The model never sees raw data, never runs code,
  and every reported finding is dropped unless it cites real evidence. Returns
  the standard `AnalysisResult`.
  - Provider-neutral `LLMProvider` interface; `GeminiProvider` over the
    `google-genai` SDK using a portable prompted-JSON protocol that works with
    both Gemma and Gemini models (default `gemma-4-31b-it`).
  - `FakeProvider` for deterministic, offline tests and documentation examples.
  - Deterministic tool registry (`list_tables`, `describe_table`,
    `profile_dataset`, `discover_schema`, `detect_anomalies`,
    `assess_classification`) returning compact, privacy-filtered summaries.
  - LangGraph flow: intake → bounded agent/tool loop → evidence-citation
    validation → synthesis, with `insufficient_evidence` and non-convergence
    fallback handling.
  - `PrivacyPolicy` now governs the dataset overview/schema description sent to a
    provider; raw cell values are withheld by default and HMAC/API keys never
    leave memory.
  - Report footer shows AI provenance (provider, model, tool-call count).
  - `GeminiProvider` retries transient API errors (429/5xx/timeouts) with
    exponential backoff and otherwise raises a clean `ProviderError` instead of a
    raw SDK traceback. Excel "no default style" reader warnings are suppressed.
- Usage documentation for the AI-assisted layer and privacy controls.
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
- Spurious one-to-many relationship candidates driven purely by ID-range overlap
  are now suppressed when they lack name similarity and adequate parent coverage.
