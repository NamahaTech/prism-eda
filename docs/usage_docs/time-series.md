# Time series

`time_series("value")` answers two questions at once: **can this be forecast**,
and **what is actually in it?**

It checks the time axis (frequency, gaps, duplicate timestamps, panel coverage),
then characterizes the series (trend, seasonality, memory, stationarity, change
points, outliers, intermittency), and finishes with a validation plan that
respects time order.

It is a *readiness and characterization* diagnostic, not a forecasting library.
It returns evidence and findings, never a fitted model or a prediction.

```python
import prism_eda as pe
from examples.sample_data import daily_orders

result = pe.load({"daily_orders": daily_orders()}).time_series(
    "orders", entity_id="store", horizon=28
)
```

As a one-liner:

```python
result = pe.time_series("data/sales.csv", value="units", timestamp="observed_at")
```

> The time column is **inferred** when the table has exactly one datetime column.
> With two — `ordered_at` and `shipped_at`, say — Prism asks rather than guesses,
> because picking the wrong one silently changes every number in the report.

## The verdict

```python
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
    print(f"    {finding.summary}")
```

```text
daily_orders.orders: not ready to forecast. Top issue — Duplicate timestamps in the series. 4 prioritized issue(s) (3 high, 1 medium). 6 alert(s) describe what is in the series. Sampling or recoverable caveats apply.
[high] Duplicate timestamps in the series
    8 row(s) across 4 timestamp(s) are recorded more than once for the same series.
[high] The aggregate is not made of the same series throughout
    The number of contributing store values moves between 2 and 3 across the history, changing 1 time(s) first at 2025-12-02.
[high] 9 period(s) were never recorded
    1 block(s) of the time axis have no row at all (1.2% of the span). The longest runs 9 daily period(s) from 2025-04-07.
[medium] 1 entity has far less history than the rest
    harbour has 61 period(s), 8% of the longest entity's history.
[medium] 2 level shifts in the series
    The most recent is 2025-12-06, where the level stepped up (+31%) and stayed there.
[low] The level moves over time
    Both tests agree the level moves. Difference the series, or use a model that handles a trend directly.
[low] The series became harder to predict
    Around 2025-02-10 the spread went wider, from 7.274 to 17.99.
[low] 8 one-off spikes in the series
    The largest is 2025-06-14, high and well outside the local range.
[low] Clear weekly seasonality
    A weekly cycle explains 66% of what is left after the trend, with a peak-to-trough swing of 102.
[low] The level is rising
    Trend explains 62% of the non-seasonal variation, moving the level by 123.2 across the history.
```

Four issues, six alerts. The split is the whole point.

## Structure is not a defect

The first four findings are things to **fix**: repeated timestamps, an aggregate
whose membership changes, a nine-day outage, an entity too short to model. The
last six **describe** the series — and a report that filed "your series is
non-stationary" as a defect would be putting a finding on almost every real
series ever analysed while telling you nothing you can act on.

```python
from prism_eda.evidence.models import split_findings

issues, alerts = split_findings(result.findings)
print(len(issues), "issues,", len(alerts), "alerts")
```

```text
4 issues, 6 alerts
```

Non-stationarity, trend, and seasonality are all properties. Each carries a
modelling consequence in its recommendation, and none of them is a problem.

## Absence comes in two forms

A single "missingness" percentage hides the distinction that matters:

```python
gaps = next(e for e in result.evidence if e.kind == "time_series_gaps")
print(gaps.value["unrecorded_period_count"], "periods with no row at all")
print(gaps.value["blank_period_count"], "periods with a row and a blank value")
print(gaps.value["unrecorded_blocks"][0])
```

A period with **no row** is a collection failure — the pipeline did not run. A
period with a row and a **blank value** is a measurement failure — the pipeline
ran and the value did not arrive. They have different causes and different
fixes, so they are counted separately and reported as contiguous blocks. A
forecaster cares that the outage lasted nine consecutive days, not that 1.2% of
rows are missing.

## Duplicate detection is entity-aware

In a panel, every date legitimately appears once per entity. Counting duplicates
on the timestamp column alone would report that essentially every row is
duplicated — true, and completely useless:

```python
aware = pe.load({"d": daily_orders()}).time_series("orders", entity_id="store")
naive = pe.load({"d": daily_orders()}).time_series("orders")

for label, res in (("entity-aware", aware), ("without entity", naive)):
    ev = next(e for e in res.evidence if e.kind == "time_series_duplicate_timestamps")
    print(f"{label:15s} {ev.value['duplicated_row_count']:>5} duplicated rows")
```

```text
entity-aware        8 duplicated rows
without entity   1507 duplicated rows
```

## The panel trap

Totalling an unbalanced panel produces a level shift on the day an entity joins
— not because demand changed, but because *the thing being measured* changed.
Every downstream check reads that as a regime change.

Prism counts the contributing series per period and reports it when it moves, so
the `+31%` "level shift" above is correctly attributable to `harbour` opening
rather than to a jump in demand.

```python
composition = next(
    e for e in result.evidence if e.kind == "time_series_panel_composition"
)
print(composition.value["min_active_entities"], "→", composition.value["max_active_entities"])
print(composition.value["changes"])
```

Structural analysis runs on the aggregate; coverage is reported per entity. Both
facts are disclosed as warnings rather than assumed.

## Trend, seasonality, and memory

```python
single = pe.load({"orders": daily_orders_single()}).time_series("orders", horizon=28)
d = next(e for e in single.evidence if e.kind == "time_series_decomposition")
print("seasonal strength", round(d.value["seasonal_strength"], 3))
print("trend strength   ", round(d.value["trend_strength"], 3))
```

```text
seasonal strength 0.433
trend strength    0.597
```

Strength is the share of variance the component removes, bounded in `[0, 1]` and
comparable across series — not the size of the seasonal swing, which is only
comparable to itself. The seasonal *profile* (`seasonal_profile`) gives the
effect of each position in the cycle, which for daily data is the day-of-week
shape.

## Stationarity: both tests, and their disagreement

ADF and KPSS have **opposite null hypotheses**, so running only one answers half
the question. Prism runs both and reports the four-way outcome:

```python
st = next(e for e in single.evidence if e.kind == "time_series_stationarity")
print(st.value["verdict"], "| tests agree:", st.value["tests_agree"])
print("ADF p", round(st.value["adf"]["p_value"], 4),
      "| KPSS p", round(st.value["kpss"]["p_value"], 4))
print(st.value["explanation"])
```

```text
non_stationary | tests agree: True
ADF p 0.7144 | KPSS p 0.01
Both tests agree the level moves. Difference the series, or use a model that handles a trend directly.
```

| Outcome | What it usually means |
|---|---|
| `stationary` | Both agree the level is stable. No differencing needed. |
| `non_stationary` | Both agree the level moves. Difference, or model the trend. |
| `trend_stationary` | Deterministic trend — **de-trend**, do not difference. |
| `difference_stationary` | Stationary around a shifting level — **difference**. |

The two middle rows only exist because both tests run. When they disagree, that
disagreement *is* the result, and it is what distinguishes the two fixes.

## Change points versus spikes

A spike is one bad day. A change point is every day afterwards. Confusing them
is expensive in both directions — deleting a change point as an outlier throws
away the most important fact about the series, and treating a promotion as a
regime change re-baselines a forecast onto a day that will never repeat.

They are detected separately, by different methods:

- **Change points** run on a *robustly detrended, seasonally adjusted* series.
  Without the detrend, binary segmentation chops a smooth trend into a staircase
  of "regime changes"; with a least-squares detrend, a couple of spikes drag the
  line and hide the real step. Theil–Sen fixes both. A shift must also move the
  level by more than the series' own noise.
- **Outliers** run on the STL remainder against an interquartile fence, with
  interpolated periods and the neighbourhood of a change point excluded — both
  produce artificial spikes that would otherwise be "discovered".

When more than 2% of periods fall outside the fence, the list is suppressed: the
series has a *changing spread*, not hundreds of anomalies, and one fence is the
wrong instrument. The rate is still reported.

## Horizon and validation

`horizon=` is optional. Supplied, it makes the history check and the backtest
plan concrete:

```python
plan = next(e for e in single.evidence if e.kind == "time_series_validation_plan")
print(plan.value["fold_count"], "folds x",
      plan.value["test_periods_per_fold"], "periods")
print("minimum training window:", plan.value["minimum_training_periods"])
```

```text
5 folds x 28 periods
minimum training window: 243
```

A random train/test split on a time series trains on the future and reports an
accuracy production cannot reproduce. Every fold here trains on everything
before its test window and nothing after it.

## Irregular data is regularized, and it says so

STL, ACF, and the stationarity tests all need a complete, evenly spaced series.
Filling is therefore unavoidable — but it manufactures observations that were
never recorded, so it is recorded as a `SamplingRecord` and stated on the page:

```python
for record in single.sampling:
    print(record.operation, record.source_rows, "→", record.sampled_rows)
```

The hygiene checks — gaps, duplicates, irregular spacing — always run on the
**raw** timestamps. Those are exact claims about defects, and a duplicate that
has already been collapsed cannot be counted.

## What stays quiet

A clean series produces no issues at all:

```python
import numpy as np, pandas as pd

rng = np.random.default_rng(5)
index = pd.date_range("2024-01-01", periods=730, freq="D")
weekly = np.where(index.dayofweek >= 5, 20, -8)
clean = pd.DataFrame(
    {"day": index, "units": 100 + np.arange(730) * 0.05 + weekly + rng.normal(0, 7, 730)}
)

result = pe.load({"t": clean}).time_series("units")
print(result.summary)
```

```text
t.units looks forecastable: no blocking time-axis problems were found. 3 alert(s) describe what is in the series.
```

Three alerts describe the trend, the weekly cycle, and the stationarity verdict.
Zero issues, zero change points, zero outliers — on a series where a naive
implementation would report a staircase of regime changes and a steady trickle
of anomalies.

## What it checks

| Area | Checks |
|---|---|
| Time axis | Span, timezone, ordering, future timestamps, inferred frequency and its regularity |
| Duplicates | Repeated timestamps per series, entity-aware, flagging conflicting values |
| Coverage | Unrecorded periods and blank periods, as contiguous blocks |
| Spacing | Departure from a single dominant interval |
| Panel | Per-entity history and coverage, imbalance, and changing membership |
| Decomposition | STL trend, seasonal, remainder; strength of each; seasonal profile |
| Memory | ACF and PACF with a confidence band, plus candidate seasonal periods |
| Stationarity | ADF **and** KPSS, with the four-way agreement reported |
| Events | Level shifts, variance shifts, and temporal outliers |
| Intermittency | Zero runs, burstiness, and Syntetos–Boylan demand classification |
| Horizon | History in seasonal cycles and as a multiple of the requested horizon |
| Validation | Expanding-window backtest folds sized to the series |
| Predictors | Lagged cross-correlation, restricted to lags usable at forecast time |

## Limits worth knowing

- Frequency is the **dominant observed spacing**, not a declared schedule. A
  series that changes cadence partway through is regularized to one grid.
- Panel structural analysis runs on the aggregate. Per-entity decomposition is
  not run; coverage is reported per entity so you can see who cannot be modelled.
- ADF and KPSS both lose power on short series, so a `stationary` verdict on a
  few dozen observations is weak evidence rather than reassurance.
- Change points are candidate boundaries. They mark where the series behaves
  differently, never why.
- Duplicate timestamps are averaged when building the grid. That is the safer
  default for a rate or a level, and arguably wrong for a count — which is why
  the duplicate finding asks you to resolve them at source.
- Lagged correlation is not causation, and two series sharing a trend correlate
  strongly at every lag for that reason alone.

## See also

- [Regression readiness](regression.md) — the same issue/alert split for a numeric target
- [The baseline profile](profile.md) — `profile()` also reports time coverage per column
- [Results & evidence](results-and-evidence.md) — the `AnalysisResult` contract
