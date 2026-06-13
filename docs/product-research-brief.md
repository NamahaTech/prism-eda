# Prism EDA Product Research Brief

Date: 2026-06-12

Status: Product discovery complete. Product, API, and execution decisions were
confirmed on 2026-06-13.

## 1. Product thesis

Prism EDA should not be another fixed data profile with an LLM-generated summary.
Its useful differentiator is a task-aware investigation system:

1. A deterministic evidence engine computes statistics, tests, candidate issues,
   and visual artifacts locally.
2. Task recipes decide which evidence matters for classification, regression,
   anomaly detection, clustering, time series, or schema discovery.
3. An optional AI investigator asks for missing context, selects safe tools,
   evaluates whether the evidence is sufficient, and writes conclusions that
   link back to computed evidence.
4. A report renderer turns the same structured result into a concise, beautiful,
   self-contained HTML report.

The AI should plan and explain analysis. It should not be the source of numeric
truth, silently execute arbitrary code, or receive the full raw dataset by
default.

## 2. What the current landscape already covers

### Profiling tools

- YData Profiling provides broad descriptive profiling, alerts, interactions,
  time-series support, dataset comparison, metadata, and HTML/JSON output. Its
  large-data guidance explicitly recommends sampling, minimal mode, or disabling
  expensive computations.
- Sweetviz adds target analysis and train/test or subgroup comparison, along with
  mixed-type associations and self-contained HTML output. This is closer to
  goal-oriented EDA, but its analysis path is still largely fixed.
- Evidently organizes metrics into data-quality, drift, classification,
  regression, and ranking evaluations. It is strongest when there is a model,
  prediction column, reference dataset, or monitoring workflow.
- Great Expectations is primarily a validation system: users express expected
  properties and validate data against them. It is adjacent to EDA rather than a
  replacement for exploratory investigation.

### Research systems

- Lux recommends visualizations from dataframe context and user intent. This
  supports the idea that EDA should react to what the analyst is doing, rather
  than display every possible chart.
- LIDA separates dataset summarization, goal exploration, visualization
  generation, and infographic generation. Its compact summarizer and explicit
  goal-exploration stage are especially relevant to token-efficient AI EDA.
- InsightPilot selects analysis intents and translates them into intentional
  queries that are executed by an insight engine. This is close to the proposed
  split between an AI planner and deterministic analysis tools.
- QUIS uses iterative question generation to improve exploration coverage. It
  reinforces the value of a question-driven loop, but full autonomy is not a
  sufficient reliability strategy on its own.
- Data Formulator separates high-level visualization intent from data
  transformation and exposes transformed data for inspection. This is a strong
  precedent for keeping AI transformations reviewable.
- Recent data-analysis-agent benchmarks such as LongDA and DataClaw report large
  reliability gaps on realistic, noisy, documentation-heavy analysis. This is a
  warning against presenting an autonomous analyst as authoritative.

### Product gap

The defensible gap is not "more statistics" or "an LLM can chat with a CSV."
It is:

- task-specific evidence selection;
- context acquisition before analysis;
- cross-column and cross-table reasoning;
- explicit assumptions and confidence;
- evidence lineage for every reported claim;
- local-first privacy and bounded AI access;
- adaptive compute and token budgets;
- reports that prioritize decisions and risks instead of metric volume.

## 3. Proposed conceptual model

### 3.1 Dataset catalog

Loading should produce a catalog rather than only a dictionary of DataFrames.
The catalog should contain:

- source identity and fingerprint;
- table names, row counts, column counts, and memory estimates;
- physical and inferred semantic types;
- candidate identifier, timestamp, target, measure, dimension, and free-text
  roles;
- sampling metadata;
- candidate primary keys, foreign keys, and functional dependencies;
- privacy classifications and redaction rules;
- user-provided data dictionary and domain context.

### 3.2 Analysis context

Every analysis should receive an explicit context object, even in non-AI mode.
Likely fields include:

- goal: profile, classification, regression, anomaly detection, clustering,
  time series, schema discovery, or custom;
- target column or target table;
- entity identifier and row granularity;
- timestamp, grouping, segment, and ordering columns;
- whether the data is training, scoring, historical, reference, or production
  data;
- known constraints and domain rules;
- cost level: quick, standard, or deep;
- row/column sampling and compute budgets;
- privacy policy and whether raw values may leave the process;
- assumptions confirmed by the user.

This context is the main contract between deterministic and AI-assisted modes.

### 3.3 Evidence model

Analysis functions should return structured evidence rather than HTML fragments.
A useful evidence record needs:

- stable evidence ID;
- analysis recipe and algorithm version;
- dataset/table/column/row scope;
- metric or finding type;
- value, uncertainty, effect size, and sample size where applicable;
- assumptions and applicability conditions;
- severity and confidence as separate concepts;
- computation cost and sampling information;
- references to supporting tables, charts, and candidate rows;
- suggested next actions;
- human-readable explanation generated deterministically when possible.

An insight is a conclusion over one or more evidence records. AI-generated
insights must cite evidence IDs and may not introduce uncited numeric claims.

## 4. Baseline report shared by every goal

The common report should stay concise and include:

- dataset fingerprint, source, scope, and sampling disclaimer;
- inferred row granularity and semantic roles, with uncertainty;
- schema and type mismatches;
- completeness, distinctness, duplicate rows, duplicate entities, and constant
  or near-constant columns;
- numeric distributions using robust summaries as well as mean and standard
  deviation;
- categorical frequencies, rare levels, cardinality, entropy, and unseen/other
  risk;
- string length and dominant pattern summaries, not only top values;
- missingness patterns across columns and important groups;
- typed associations, with nonlinear and categorical associations separated from
  Pearson correlation;
- candidate constraints and data-quality issues;
- prioritized findings, not a flat list of warnings.

## 5. Task-specific report recipes

### 5.1 Anomaly detection

The first question is whether the task is outlier detection on contaminated
historical data, novelty detection against known-normal data, or supervised rare
event detection. These are different problems and should not share one default.

Recommended report sections:

- expected anomaly definition, entity, time window, and approximate prevalence;
- class balance and label quality when labels exist;
- invalid values and hard domain-rule violations;
- univariate robust tail candidates using quantiles, IQR, and median absolute
  deviation;
- multivariate global candidates, initially using Isolation Forest;
- local-density candidates using Local Outlier Factor where dimensionality and
  sample size are suitable;
- robust covariance/Mahalanobis candidates only when numeric data is compatible
  with its distributional assumptions;
- conditional anomalies, where a value is surprising given other features or a
  subgroup, which addresses cases such as an individually plausible age and
  weight forming an implausible pair;
- rare categorical combinations and unusual group transitions;
- score stability across seeds, subsamples, and reasonable hyperparameters;
- agreement and disagreement between detectors;
- per-row explanations showing the features, conditions, or neighbors that made
  a candidate unusual;
- a ranked review table that calls records "candidates," not confirmed anomalies.

Important limitation: high-dimensional unsupervised anomaly detection is
intrinsically difficult. A report should expose method assumptions and detector
disagreement instead of manufacturing a single authoritative anomaly label.

### 5.2 Classification

Recommended report sections:

- target validity, class counts, imbalance ratio, entropy, and minimum examples
  per class;
- exact duplicates and near-duplicates, especially rows with conflicting labels;
- missingness and representation by class and important subgroup;
- typed feature-target associations with effect sizes;
- class-conditional distributions and category lift;
- overlap and neighborhood disagreement between classes;
- a small, clearly labeled diagnostic probe model to estimate separability,
  surface suspiciously predictive columns, and identify hard examples;
- target leakage candidates, including direct encodings, post-outcome timestamps,
  identifiers, and implausibly predictive fields;
- high-cardinality and unseen-category risk;
- train/test comparison when both are supplied;
- group/time-aware split recommendations based on identifiers and timestamps;
- fairness-relevant subgroup coverage when the user opts into that analysis.

Class imbalance alone is not enough. Classification difficulty also comes from
class overlap, label noise, small disjuncts, sparsity, and insufficient subgroup
coverage.

### 5.3 Regression

Recommended report sections:

- target range, skew, zeros, censoring, heaping, tails, and transformation
  candidates;
- typed and nonlinear feature-target associations;
- redundant features and multicollinearity diagnostics, avoiding universal VIF
  cutoffs;
- a robust and a conventional baseline probe;
- residual shape, conditional bias, and heteroscedasticity from the probes;
- high-leverage and influential observations, including Cook-style influence
  diagnostics where applicable;
- error concentration by subgroup and target range;
- leakage candidates and time/group split risks;
- regions with weak support or extrapolation risk.

Regression diagnostics derived from a linear probe must be labeled as
model-conditional; they are not universal properties of the dataset.

### 5.4 Time series

Recommended report sections:

- time-index inference, timezone, frequency, span, duplicate timestamps, gaps,
  and irregular sampling;
- entity count for panel or hierarchical series and coverage per entity;
- trend, seasonality, remainder, and seasonal strength using decomposition where
  enough history exists;
- autocorrelation and partial autocorrelation summaries;
- stationarity tests with assumptions and disagreement shown;
- change points, level shifts, variance shifts, and temporal outliers;
- intermittent demand, zero runs, burstiness, and event sparsity;
- lagged and rolling associations with leakage-safe directionality;
- missing blocks rather than only total missingness;
- history length relative to the requested forecast horizon;
- candidate evaluation and cross-validation strategy based on temporal order.

The catch22 work is useful inspiration for a compact set of interpretable time
series characteristics, but Prism EDA should select features based on the task and
avoid computing a large library by default.

### 5.5 Clustering

Recommended report sections:

- feature type, scaling, cardinality, redundancy, missingness, and distance-metric
  compatibility;
- duplicate and near-duplicate observations;
- intrinsic dimensionality and distance concentration warnings;
- cluster tendency, for example a Hopkins-style diagnostic with repeated samples;
- candidate embeddings used as visual aids, not proof of cluster structure;
- candidate algorithms based on geometry and data types;
- multiple values of cluster count evaluated with silhouette, gap-style, and
  algorithm-appropriate criteria;
- resampling stability and sensitivity to scaling/features;
- cluster sizes, distinguishing features, and representative examples;
- an explicit "no stable cluster structure found" result when appropriate.

Clustering has no universal ground truth. A report should emphasize stability,
interpretability, and sensitivity rather than optimizing one internal score.

### 5.6 Multi-table schema discovery

Recommended report sections:

- candidate primary keys using uniqueness, null rate, stability, and semantic
  plausibility;
- candidate foreign keys using inclusion coverage, orphan rate, type
  compatibility, name similarity, and join cardinality;
- one-to-one, one-to-many, and many-to-many relationship candidates;
- composite-key discovery as a deeper optional pass;
- approximate functional dependencies and likely denormalized dimensions;
- cycles, bridge tables, disconnected tables, and suspicious fan-out joins;
- a schema graph where every edge exposes confidence and supporting evidence;
- user confirmation before a candidate relationship is treated as truth.

Value overlap alone is not sufficient to infer a foreign key. Names, semantic
types, uniqueness, coverage, null behavior, and join behavior must contribute to
the confidence score.

## 6. AI-assisted analysis design

### 6.1 Recommended workflow

1. Intake: ask only questions that materially change the analysis.
2. Deterministic scan: create a compact catalog and baseline evidence locally.
3. Semantic proposal: infer likely roles, domain, granularity, and relationships.
4. Human checkpoint: confirm high-impact ambiguities such as target, entity,
   timestamp, or whether a value is an identifier.
5. Structured plan: the model returns a validated analysis plan with tool calls,
   budget, rationale, and stopping conditions.
6. Execution: whitelisted deterministic tools run locally.
7. Critique: check evidence coverage, contradictions, detector stability, and
   whether another question or tool call is justified.
8. Report: produce findings with evidence IDs, assumptions, confidence, and next
   actions.

LangGraph fits this workflow because its persistence and interrupts support
resumable investigation and human-in-the-loop clarification. LangChain's current
agent API, structured output strategies, tools, and ToolRuntime can provide the
model and tool integration. The graph should remain small and explicit; it does
not need a large multi-agent hierarchy for an initial release.

### 6.2 High-information tools

Tools should return bounded, typed summaries rather than dataframe dumps. A
possible initial tool set:

- `inspect_catalog`: compact table, role, type, and quality manifest;
- `inspect_columns`: deeper profile for selected columns;
- `compare_groups`: effect sizes and distributions across target/segment groups;
- `rank_anomaly_candidates`: detector scores, agreement, stability, and evidence;
- `inspect_candidate_rows`: explicitly scoped row values with privacy controls;
- `test_relationships`: candidate key, inclusion, and join-cardinality evidence;
- `run_task_probe`: bounded classification/regression/clustering probe;
- `inspect_time_structure`: frequency, gaps, seasonality, dependence, and shifts;
- `get_evidence`: retrieve selected evidence records by ID;
- `render_artifact`: create a chart/table from existing evidence without asking
  the model to write plotting code.

Arbitrary Python execution by the model should not be in the first release. If
added later, it needs a separate sandbox, resource limits, filesystem isolation,
network denial, provenance, and an explicit user opt-in.

### 6.3 Token and cost controls

- send a compact catalog first, never the full dataset;
- expose only tools relevant to the current task and state;
- cap rows, categories, correlations, findings, and tool-result payload sizes;
- return evidence IDs and retrieve detail lazily;
- use structured output schemas to prevent verbose prose between tool calls;
- summarize conversation state and persist confirmed facts separately;
- cache stable dataset context and deterministic tool results;
- use Gemini token counting before calls and record response usage metadata;
- set per-run call, input-token, output-token, compute, and wall-time budgets;
- make "why another call is worth its cost" part of the planner state;
- prefer one rich tool result over many tiny dataframe queries.

Gemini supports function calling, structured output, token accounting, and
context caching. Structured JSON still requires semantic validation in the
application.

### 6.4 Privacy and trust

- local-only deterministic analysis must remain fully useful without an API key;
- raw rows should not leave the process by default;
- add PII and secret detection before any model payload is built;
- allow column exclusion, hashing, bucketing, and representative-value redaction;
- show a payload preview or auditable payload log in a privacy/debug mode;
- distinguish facts, model interpretations, assumptions, and user-confirmed
  domain rules in the report;
- never store API keys in report artifacts, graph state, or logs;
- document that Gemini unpaid services may use submitted content and generated
  responses to improve Google products and may involve human review; Google says
  not to submit sensitive, confidential, or personal information to unpaid
  services;
- recommend paid-service configuration for real organizational data, while still
  enforcing Prism EDA's own minimization and redaction controls.

## 7. Reporting and rendering

The renderer should consume the structured report model and remain independent
from analysis algorithms.

Recommended core approach:

- Jinja2 templates;
- semantic HTML with strong accessibility defaults;
- one embedded stylesheet with CSS custom properties;
- small, vendored vanilla JavaScript only for navigation, filtering, and
  disclosure controls;
- inline SVG for sparklines, bars, distributions, missingness matrices, and
  schema graphs;
- no CDN requirement and a self-contained HTML option;
- print-friendly styles and a useful static view when JavaScript is disabled;
- optional Plotly or other interactive rendering as an extra, not a core
  dependency;
- deterministic snapshots or image-based visual tests for report layouts.

The report hierarchy should be:

1. decision summary;
2. assumptions and confidence;
3. prioritized findings;
4. task-specific evidence;
5. column/table explorer;
6. methods, sampling, and reproducibility details.

## 8. Python library architecture

The public package is `prism_eda`; its internal boundaries should look roughly
like this:

```text
src/<package>/
  catalog/          loading, fingerprints, semantic roles, relationships
  evidence/         typed result and artifact models
  profiling/        reusable deterministic metrics
  recipes/          task-specific planners and analysis pipelines
  reporting/        report model, Jinja templates, CSS, SVG renderers
  assistant/        optional LangChain/LangGraph integration
  providers/        Gemini and future model adapters
  policies/         privacy, budgets, sampling, and execution controls
  cli/              optional command-line interface
```

Key engineering choices:

- retain a `src` layout and Hatchling/PEP 621 packaging;
- keep core dependencies deliberate; PyArrow is included for first-class Parquet
  support, while AI and interactive visualization stay behind optional extras;
- use typed, JSON-serializable domain models that do not require LangChain;
- keep analysis recipes independent from rendering and model providers;
- define plugin protocols for new goals, metrics, renderers, and dataframe
  backends;
- seed every stochastic operation and record versions/configuration;
- support cancellation and explicit resource budgets;
- test algorithms with synthetic fixtures that contain known pathologies;
- use property tests for invariants and golden/snapshot tests for reports;
- test built wheels, editable installs, supported Python versions, and the public
  API in CI;
- publish API stability and deprecation rules before a 1.0 release.

### Foundation changes completed

- Hatchling now builds the real `src/prism_eda` package.
- The local Windows-path profiling experiment was removed.
- `pyproject.toml` is the canonical dependency source; the old environment freeze
  was removed.
- Matplotlib, Seaborn, and `tqdm` are not core dependencies. Static reports use
  HTML/CSS/SVG and progress is exposed through events.
- The foundation includes tests, coverage configuration, Ruff, mypy, package
  build verification, CI, API documentation, and a changelog.

## 9. Recommended initial scope

Do not launch with five shallow specialized reports and a general autonomous
agent. Build two strong vertical slices:

### Milestone A: deterministic foundation

- CSV and Parquet catalog loading;
- one-table and multi-table catalog;
- semantic type/role inference with user overrides;
- concise baseline EDA;
- evidence and report contracts;
- self-contained HTML renderer;
- reproducible quick/standard/deep budgets.

### Milestone B: first task recipes

- classification report;
- anomaly-detection report, including multivariate and conditional candidates;
- candidate key/foreign-key discovery for multi-file inputs.

These cover the clearest product examples and force the architecture to support
targets, labels, groups, multivariate behavior, and multiple tables.

### Milestone C: constrained AI investigator

- Gemini provider through LangChain;
- a small LangGraph with intake, plan, execute, clarify, critique, and report
  states;
- structured plans and evidence-cited findings;
- token/cost ledger and privacy controls;
- deterministic replay tests with mocked model responses.

### Later milestones

- regression, time-series, and clustering recipes;
- OpenAI and local-model providers;
- notebook widgets and richer interactivity;
- dataframe backends beyond pandas;
- sandboxed custom analysis, only if demand justifies the security complexity.

## 10. Research references

### Existing systems and official documentation

- [YData Profiling documentation](https://docs.profiling.ydata.ai/latest/)
- [YData Profiling: large datasets](https://docs.profiling.ydata.ai/latest/features/big_data/)
- [Sweetviz repository and feature overview](https://github.com/fbdesignpro/sweetviz)
- [Evidently metrics](https://docs.evidentlyai.com/metrics/all_metrics)
- [Great Expectations Core](https://docs.greatexpectations.io/docs/core/introduction/)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
- [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini token counting](https://ai.google.dev/gemini-api/docs/tokens)
- [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching)
- [Gemini API terms](https://ai.google.dev/gemini-api/terms)
- [PyPA packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [PyPA src-layout discussion](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)

### Research papers

- [Lux: Always-on Visualization Recommendations for Exploratory Dataframe Workflows](https://arxiv.org/abs/2105.00121)
- [LIDA: Automatic Generation of Grammar-Agnostic Visualizations](https://arxiv.org/abs/2303.02927)
- [InsightPilot: An LLM-Empowered Automated Data Exploration System](https://arxiv.org/abs/2304.00477)
- [QUIS: Question-guided Insights Generation for Automated EDA](https://arxiv.org/abs/2410.10270)
- [Data Formulator: AI-powered Concept-driven Visualization Authoring](https://arxiv.org/abs/2309.10094)
- [Data Formulator 2](https://arxiv.org/abs/2408.16119)
- [LongDA: Benchmarking LLM Agents for Long-Document Data Analysis](https://arxiv.org/abs/2601.02598)
- [DataClaw: A Process-Oriented Agent Benchmark](https://arxiv.org/abs/2605.02503)
- [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17)
- [LOF: Identifying Density-Based Local Outliers](https://dl.acm.org/doi/10.1145/342009.335388)
- [catch22: Canonical Time-series Characteristics](https://arxiv.org/abs/1901.10200)
- [Validation of Cluster Analysis Results on Validation Data](https://arxiv.org/abs/2103.01281)
- [A Lightweight Algorithm to Uncover Deep Relationships in Data Tables](https://arxiv.org/abs/2009.03358)

## 11. Confirmed product decisions

The following decisions were confirmed on 2026-06-13:

- The primary initial users are data scientists.
- The product is a Python library first, in the spirit of pandas or TensorFlow,
  rather than a standalone UI product.
- Deterministic report generation and interactive AI investigation are equally
  important product experiences.
- The public API will accept pandas DataFrames as well as individual file and
  directory paths.
- Standard mode should be designed for datasets up to approximately 10 million
  rows. This is a resource target, not a promise that every quadratic or
  model-based analysis runs over all rows.
- Recursive directory discovery will be available as an explicit option and
  disabled by default.
- Multi-file inputs may contain arbitrary related tables, not only train/test
  pairs.
- Schema discovery should include candidate composite keys.
- Classification, unlabeled historical anomaly detection, and schema discovery
  are the first specialized capabilities.
- Specialized reports may train lightweight diagnostic probe models, with clear
  labeling that they are analysis instruments rather than production models.
- Data transformations remain a separate subpackage. Analysis may recommend and
  return transformation plans but must not silently mutate user data.
- Gemini is the only initial AI provider. The provider boundary must support
  future OpenAI-compatible and local models without changing the analysis core.
- AI support will be installed separately, for example with
  `pip install prism-eda[ai-gemini]`.
- Raw values are prohibited from model payloads by default. Privacy-preserving
  representations may retain equality and relationship evidence without
  disclosing real values.
- Exact model-payload inspection is deferred beyond the first release.
- AI-assisted analysis should conduct a meaningful context interview rather than
  ask only one or two high-impact questions.
- AI-generated findings must cite internal evidence IDs.
- Arbitrary AI-generated Python execution is excluded from the first release.
- User-configurable token, API-call, compute-time, and estimated-cost limits are
  deferred to a later release. Internal conservative limits are still required
  to prevent runaway execution.
- AI investigations are resumable while the in-memory session remains alive.
  Durable cross-process resume is deferred until checkpoint persistence is added.
- Reports must support a self-contained HTML artifact and a machine-readable JSON
  result.
- Plotly is an optional extra for interactive charts. Core reports should still
  contain lightweight static charts rendered with HTML, CSS, or inline SVG.
- Reports lead with a concise decision summary before detailed evidence.
- The project remains MIT licensed and supports Python 3.11 and newer.
- The distribution name is `prism-eda`. The Python import package is
  `prism_eda`, because Python package imports cannot contain hyphens.
- No representative real-world datasets are currently available from the project
  owner, so the initial test suite must use public datasets and purpose-built
  synthetic fixtures with known pathologies.

## 12. Confirmed API and execution decisions

The second product interview confirmed the following:

- The primary API is session-based, with convenience functions for common
  one-shot analyses.
- AI investigation uses an `Investigator` session object.
- Interviews are exposed through events and callbacks. An optional terminal
  adapter may handle those events, but core library calls never unexpectedly
  invoke `input()`.
- AI APIs support synchronous and asynchronous execution.
- Analysis returns an in-memory result and writes nothing until the user calls an
  explicit export method such as `to_html()` or `to_json()`.
- Cheap and streaming-compatible metrics may run over all rows. Expensive
  operations may sample automatically, must emit a visible warning, and must
  record the sampling strategy in evidence and reports.
- Users can disable automatic sampling. This may substantially increase runtime
  and memory use and should require an explicit configuration choice.
- `quick`, `standard`, and `deep` modes select algorithms, sampling limits, and
  compute budgets. Every stochastic operation is deterministic by default and
  records its seed.
- Standard composite-key discovery tests combinations of up to two columns;
  deep mode may test combinations of up to three columns.
- A foundational failure aborts analysis. Optional or individual metric failures
  are recorded in the report and analysis continues.
- Pandas is the only initial in-memory backend. CSV processing should use chunks
  where the metric permits it.
- PyArrow is a core dependency, so Parquet works in the base installation.
- Unsupervised anomaly results are ranked candidates rather than definitive
  anomaly labels.
- Users may supply an expected contamination rate. The default remains
  threshold-free and avoids inventing a binary cutoff.
- Initial anomaly analysis covers numeric features and rare categorical
  combinations. Free-text semantic anomalies are deferred.
- Deterministic classification requires an explicit target. Assisted analysis may
  propose a target but must obtain user confirmation.
- Probe models use cross-validation where dataset size, class support, and the
  chosen compute mode permit it.
- Fairness and sensitive-group analysis require explicit opt-in.
- Reports may return `insufficient_evidence` or `no_meaningful_structure` instead
  of forcing a conclusion. Users may explicitly request a best-effort report,
  which must retain prominent evidence-quality warnings.
- Table and column names may be sent to Gemini by default. Documentation must
  state this clearly and users can override the behavior through privacy rules.
- Joins and relationship discovery run locally. Hashed values are sent only when
  aggregate evidence cannot support the requested reasoning.
- Privacy-preserving aliases use keyed HMAC rather than plain hashes.
- Column privacy policies support `allow`, `redact`, `alias`, and `exclude`.
- Version 0.1 does not persist AI checkpoints. Session state is in memory and is
  lost when the Python process or object is lost.
- Future checkpoints must exclude complete raw rows and contain only graph state,
  evidence, aliases, and dataset fingerprints.
- A changed dataset fingerprint invalidates resume; the user must start a new
  investigation.
- Version 0.1 establishes the deterministic engine. Gemini-assisted analysis is
  introduced in version 0.2 after evidence and report contracts stabilize.
- Transformation plans are declarative JSON-serializable objects. Generated
  pandas code is not part of the initial release.
- The command-line interface is excluded from the first release.

The concrete public API and package boundaries are specified in
`docs/public-api-and-architecture.md`.
