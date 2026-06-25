# AI-assisted investigation

The recipes in the rest of this guide are fully deterministic — you choose which
one to run. The **assisted layer** adds an optional LLM "investigator" that
decides *which deterministic tools to run* to reach a goal you state in plain
terms, then writes up the findings.

The crucial design point: **the model never sees your raw data and never runs
code.** It can only call Prism's registered, deterministic tools (the same
recipes documented elsewhere), and every finding it reports is **dropped unless
it cites real evidence** those tools produced. The AI plans and explains; it never
invents numbers.

> This is an optional extra. Install it with `pip install "prism-eda[ai-gemini]"`.
> The deterministic core never imports any LLM library.

## Install

```bash
pip install "prism-eda[ai-gemini]"
```

This adds LangGraph (the orchestration state machine) and the Google
`google-genai` SDK. Get a Google AI Studio API key and expose it:

```bash
export GEMINI_API_KEY="your-key"
```

## A first investigation

```python
import prism_eda as pe
from prism_eda.assisted_analysis import Investigator, GeminiProvider
from examples.sample_data import load_sample

dataset = pe.load(load_sample())

investigator = Investigator(dataset, provider=GeminiProvider.from_env())
session = investigator.start(
    goal="classification",
    context={"target": "churned", "domain_notes": "Monthly subscriptions."},
)
result = session.run()

print(result.status)
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title} — cites {finding.evidence_ids}")

result.to_html("investigation.html")
```

`result` is the **same `AnalysisResult`** the deterministic recipes return — so
`status`, `findings`, `evidence`, `to_html()`, and `to_json()` all work exactly as
in [Results & evidence](results-and-evidence.md). The report footer shows it was
produced by the AI engine, with the provider and model.

## What it produces (reproducible example)

Because real LLM output varies run to run, the example below uses the built-in
**`FakeProvider`** — a deterministic, offline stand-in that drives the very same
flow with no network or API key. It's what powers the test suite, and it's the
best way to see the shape of a result:

```python
import prism_eda as pe
from prism_eda.assisted_analysis import Investigator, FakeProvider
from examples.sample_data import load_sample

dataset = pe.load(load_sample())
result = Investigator(dataset, provider=FakeProvider()).start(
    goal="classification", context={"target": "churned"}
).run()

print(result.status)
print(result.summary)
for f in result.findings:
    print(f"[{f.severity}] {f.title}  cites={list(f.evidence_ids)}")
print(result.metadata)
```

```text
completed
customers.churned: not ready to model. Top issue — Potential target leakage: exit_survey_sent. 4 prioritized finding(s) (1 critical, 1 high, 2 medium).
[critical] Potential target leakage: exit_survey_sent  cites=['ev_0b5b3e6d53fc6712']
[high] Identifier-like feature: customer_id  cites=['ev_e8b87a561ecfdf7f']
[medium] Weak classification separability in customers  cites=['ev_501376985e1008f1']
[medium] Probe hard examples in customers  cites=['ev_bc6a7fdd2276077c']
{'engine': 'assisted', 'provider': 'fake', 'model': 'fake-deterministic', 'tool_calls': 1, 'uncited_findings_dropped': 0, 'converged': True}
```

Notice `tool_calls` and `uncited_findings_dropped` in the metadata — the layer
records how many tools the model ran and how many proposed findings it threw away
for failing citation validation.

## How it works

```text
  intake ──▶ agent step ──▶ (tool call?) ──▶ run deterministic tool ──┐
                 ▲                                                     │
                 └──────────────── loop, banking evidence ◀───────────┘
                 │
            (finish?) ──▶ validate citations ──▶ synthesize AnalysisResult
```

1. **Intake** announces the run and gathers a privacy-safe dataset overview.
2. **Agent step** asks the provider for its next decision. If it asks to call a
   tool, Prism runs that *deterministic* tool, banks the evidence it produced, and
   loops. The loop is bounded by `max_steps`.
3. When the model decides to finish, **citation validation** keeps only the
   findings that cite evidence IDs which actually exist; the rest are dropped.
4. **Synthesis** assembles the standard `AnalysisResult`.

The orchestration is a [LangGraph](https://langchain-ai.github.io/langgraph/)
state machine; you don't interact with it directly.

### The tools the model may call

The model's entire action space is this fixed registry — nothing else is
possible:

| Tool | Wraps |
|------|-------|
| `list_tables` | Catalog overview (table names, shapes, columns) |
| `describe_table` | Column types, roles, missingness for one table |
| `profile_dataset` | [`profile()`](profile.md) |
| `discover_schema` | [`discover_schema()`](schema-discovery.md) |
| `detect_anomalies` | [`anomaly_detection()`](anomaly-detection.md) |
| `assess_classification` | [`classification()`](classification.md) |

Each tool returns a **compact, aggregate summary** (never raw rows) for the model
to reason over, plus the citable evidence it generated.

## Goals

`start(goal=...)` accepts the same goals as the deterministic recipes —
`"profile"`, `"schema_discovery"`, `"anomaly_detection"`, `"classification"` — and
their aliases. Provide task details through `context` (see
[Context & configuration](context-and-config.md)):

```python
investigator.start(
    goal="anomaly_detection",
    context={"target": "label", "domain_notes": "Card transactions; review fraud."},
)
```

## Configuration

```python
Investigator(
    dataset,
    provider=GeminiProvider.from_env(),
    privacy=None,        # PrivacyPolicy; see the privacy guide
    callbacks=[...],     # event observers, as in events-and-progress.md
    max_steps=8,         # max tool calls before the run must conclude
)
```

### Choosing a model

`GeminiProvider` defaults to a small, fast Gemma model (generous rate limits, good
for iteration). Override it, or use a larger Gemini model for harder reasoning:

```python
GeminiProvider.from_env(model="gemma-4-31b-it")     # default: small & fast
GeminiProvider.from_env(model="gemini-2.5-flash")   # larger, stronger reasoning
GeminiProvider.from_env(temperature=0.0)            # steadier tool use
```

To see which model ids your key can access:

```python
print(GeminiProvider.from_env().available_models())
```

Prism talks to the model through a portable **prompted-JSON protocol** rather than
native function-calling, so small open models like Gemma — which don't reliably
support function-calling — work just as well as Gemini models.

## Guarantees

The assisted layer keeps the same trust contract as the rest of Prism:

- **No raw data leaves your machine by default.** Tools send aggregate summaries
  and (per your [privacy policy](privacy.md)) column names — never cell values.
- **The model cannot run code.** Its only actions are the registered tools above.
- **Every finding is evidence-backed.** Uncited or fabricated claims are dropped
  before they reach the result; the count is recorded in `metadata`.
- **"I don't know" is a valid answer.** If the evidence supports nothing, the
  result status is `insufficient_evidence` rather than a forced conclusion.
- **It degrades gracefully.** If the model can't converge within `max_steps`, the
  run falls back to the deterministic findings it already gathered and flags this
  with a warning.
- **Determinism where it counts.** The evidence and recipe results are
  deterministic; only the model's narration varies between runs.

## Bringing your own provider

`GeminiProvider` and `FakeProvider` both implement a tiny `LLMProvider`
interface — given a `DecisionRequest`, return a `ProviderDecision`. To support a
different backend, implement that one method:

```python
from prism_eda.assisted_analysis import LLMProvider, ProviderDecision

class MyProvider(LLMProvider):
    name = "my-llm"
    model = "my-model"

    def decide(self, request) -> ProviderDecision:
        ...  # call your LLM, return the next ProviderDecision
```

Everything upstream — the graph, tools, validation, reporting — is unchanged.

## Privacy

What reaches the provider is governed by a `PrivacyPolicy`. See the dedicated
[privacy guide](privacy.md) for the controls (allow / redact / alias / exclude),
the defaults, and how to keep sensitive columns out of model context.
