# Privacy

When you use the [AI-assisted layer](ai-assisted-analysis.md), some dataset
context is sent to a third-party LLM provider. A **`PrivacyPolicy`** controls
exactly what — and Prism's defaults are conservative: aggregate summaries and
column names go out, **raw cell values do not**.

> Privacy controls only apply to the AI-assisted layer. The deterministic recipes
> ([profile](profile.md), [schema](schema-discovery.md), etc.) run entirely
> locally and send nothing anywhere.

```python
from prism_eda.privacy import PrivacyPolicy, ColumnPolicy
```

## The default policy

If you don't pass one, the investigator uses `PrivacyPolicy()`:

```python
PrivacyPolicy(
    default="allow",          # per-column default action
    send_column_names=True,   # column names provide useful semantic context
    allow_raw_values=False,   # cell values are NOT sent
)
```

So by default the provider sees the **shape and schema** of your data — table
names, column names, types, missingness rates, aggregate findings — but never the
values inside cells.

## Per-column actions

Set an action per column to tighten (or loosen) what's shared. The four actions:

| Action | Effect on what the model sees |
|--------|-------------------------------|
| `allow` | Column name and aggregates (and raw `top_values` only if `allow_raw_values=True`) |
| `redact` | Name kept, but content/shape replaced with a placeholder |
| `alias` | Name replaced with a stable, keyed HMAC alias (`column_<hash>`) |
| `exclude` | Column omitted from model context entirely |

```python
policy = PrivacyPolicy(
    columns={
        "churned": ColumnPolicy("exclude"),     # keep the label out entirely
        "signup_age": ColumnPolicy("alias"),    # share the column, hide its name
    },
)
```

Applied to the sample dataset, the privacy-safe overview the model receives
becomes (note `churned` is gone and `signup_age` is aliased):

```text
2 table(s), 320 rows, 11 columns total.
- customers (80 rows): customer_id:numeric, region:categorical, plan:categorical, tenure_months:numeric, monthly_spend:numeric, column_6b00dc7b8165f47e:numeric, exit_survey_sent:numeric
- orders (240 rows): order_id:numeric, customer_id:numeric, amount:numeric
```

## Hiding all column names

If even column names are sensitive, turn them off globally — every name is
replaced with a stable alias:

```python
policy = PrivacyPolicy(send_column_names=False)
```

```text
- customers (80 rows): column_dbd8e743699d200c:numeric, column_be913647f70c8227:categorical, ...
```

The model can still reason about structure ("the second categorical column
predicts the target") without ever learning what the columns are called.

## Using a policy

Pass it to the investigator:

```python
from prism_eda.assisted_analysis import Investigator, GeminiProvider

investigator = Investigator(
    dataset,
    provider=GeminiProvider.from_env(),
    privacy=policy,
)
```

## How aliasing works

`alias` (and `send_column_names=False`) use a **keyed HMAC**. Aliases are:

- **Stable within a policy** — the same value always maps to the same alias, so
  the model can still reason about repeated values…
- **…but not reversible** — the alias is a one-way hash, so the provider can't
  recover the original.

```python
policy = PrivacyPolicy()
policy.alias("alice")      # -> 'alias_740617b77c0375bc'
policy.alias("alice")      # -> 'alias_740617b77c0375bc'  (same every time)
```

The HMAC key is generated per `PrivacyPolicy` instance and lives only in memory.
**It is never written to reports, logs, events, or provider requests.** Likewise,
your API key stays on the SDK client and never enters investigation state or a
report.

## What's sent, at a glance

| Category | Sent by default? | Controlled by |
|----------|------------------|---------------|
| Table & column names | Yes | `send_column_names`, per-column `alias`/`exclude` |
| Column types, roles, missingness, aggregate findings | Yes | per-column `redact`/`exclude` |
| Raw cell values / `top_values` | **No** | `allow_raw_values` + per-column `allow` |
| API key, HMAC key | **Never** | — |

## Current scope

In this first release, the privacy policy governs the **dataset overview and
schema description** the model reasons over. Recipe findings (e.g. "high
missingness in `signup_age`") are aggregate by nature and may reference column
names per your policy. Wiring the policy into a fuller model-payload builder is
tracked on the [roadmap](../implementation-status.md); raw values are never sent
regardless.
