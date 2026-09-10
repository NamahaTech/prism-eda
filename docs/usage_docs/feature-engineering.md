# Feature engineering

Feature code forces a choice, and both options are bad.

Write each feature as its own function and you keep the thing that makes
experimentation possible: add one, drop one, reorder them, and nothing else
moves. But every function re-scans the frame, so *n* features cost *n* passes.
Hand-fuse them into one function — one sort, masks built once, normalisations
reused — and you get the speed back by giving up the modularity you wanted.

Production feature code usually ends up on the second horn. A real 44-feature
account model this module was designed against lives in a single 250-line
function, with one `sort_values`, three time masks built once and reused, and
each string column normalised exactly once. A person performed, by hand,
precisely the common-subexpression elimination a query planner performs
automatically. The cost is that no one of those 44 features can now be removed
without editing shared code.

`prism_eda.features` exists to make that trade unnecessary. You write the
features separately; the planner recovers the performance, and proves the
numbers did not change.

```python
import prism_eda as pe
from examples.sample_data import account_transactions
from prism_eda.features import FeatureSet, safe_div, log1p, top_share

ledger = account_transactions()   # 4,059 transactions across 120 accounts

fs = FeatureSet(entity="account_id", time="txn_ts", reference_time="decided_at")
fs.window("w7d", days=7)
fs.window("w30d", days=30)
fs.window("last10", last_k=10)

@fs.derive                       # shared intermediate, not an output column
def failed(status):
    return status.str.upper().str.strip().isin({"FAILED", "DECLINED", "FAILURE"})

@fs.feature
def txn_count_30d(w30d):
    return w30d.count()

@fs.feature
def failed_ratio_30d(failed, w30d):
    return safe_div(w30d.count(where=failed), w30d.count())

@fs.feature
def failed_ratio_last10(failed, last10):
    return safe_div(last10.count(where=failed), last10.count())

@fs.feature
def max_consecutive_failures(failed, all):
    return all.run_length_max(failed)

@fs.feature
def amt_mean_log(amount, all):
    return log1p(all.mean(amount))

@fs.feature
def top_beneficiary_share_7d(beneficiary, w7d):
    return top_share(w7d, beneficiary)

run = pe.features({"account_transactions": ledger}, fs)
run.features          # one row per account, columns in declared order
run.report.to_html("features.html")
```

```text
6 feature(s) computed for 120 entities; 9 of 24 operations are shared between
features. 1 issue(s) (1 high). Top issue — The source frame contains rows from
after the decision moment.
```

That issue is not a defect in the features. It is prism noticing that the
extract itself was never filtered to the decision moment — the plan excluded
those rows, but anything else reading the same frame will not.

## It is not a faster loop, it is a different shape of computation

The obvious way to compute per-entity features is a loop: group by account,
hand each group to a function. That is what the production code this was built
against does, over roughly 3.1 million accounts, and it is where nearly all the
time goes — each call re-pays pandas' per-call overhead on a frame of a few
dozen rows.

The planner does not speed that loop up. It removes it, evaluating every
entity at once. On 20,000 accounts and 800,000 rows with the fifteen features
in the test suite:

| | |
|---|---|
| per-entity Python loop | 47.1 s |
| planned execution | 0.42 s |
| | **112x**, every value identical |

Three measurements shaped the design, and one of them is a warning:

- **Sharing the grouper** across six aggregations: 240 ms → 44 ms (5.5x).
- **Sharing the sort order** across four ordered features: 1479 ms → 368 ms (4x).
- **Fusing the aggregations** on top of a shared grouper: 44 ms → 42 ms. Worth
  almost nothing, so it is not attempted.
- **Fusing elementwise operations** made things *slower* (7 ms → 13 ms), so it
  is not attempted either.

The win is in reusing expensive physical artifacts — the sort order, the
grouper, each normalised column, each window mask — and above all in
vectorising across entities. It is a materialisation planner, not an
instruction-level fusion compiler, and that is why it is small enough to be
correct.

## Your functions are run, not read

A feature function is executed once, at plan time, with symbolic proxies
standing in for its inputs. Whatever it does to them is recorded as an
expression tree.

The alternative would be parsing your pandas source. That was rejected: to
share a grouper between two features the planner must *prove* they group
identically, and that proof is free from a recorded operation while being both
hard and unsound to extract from arbitrary Python.

The cost is that **branching on data cannot work**, because at trace time a
value has a shape but no contents:

```python
@fs.feature
def bad(amount, all):
    if amount > 100:          # TracingError, with an explanation
        return all.count()
```

Branching on the **schema** is fine, and is what real feature code actually
does to tolerate an optional column:

```python
@fs.feature
def non_upi_share(channel, all):
    if "channel" not in fs.columns:      # the schema is known at trace time
        return all.count() * 0.0
    return safe_div(all.count(where=channel != "UPI"), all.count())
```

For anything the algebra genuinely cannot express, `@fs.opaque` runs your code
unchanged, one entity at a time. It is correct, it is just unplanned — and the
cost report names it, because that is where the time will be.

## You write one entity; the planner writes all of them

The unit of definition is a single entity, because that is how people actually
reason about a feature: *this account's* failure rate over *its* last 30 days.
The planner is responsible for evaluating that definition across every entity
simultaneously.

This is also what lets one definition serve two paths. A training backfill is
the many-entity case and a serving request is the one-entity case of the same
expression tree, so they cannot drift apart the way two implementations do.

## Windows are declared, not inlined

```python
fs.window("w30d", days=30)     # a lookback
fs.window("last10", last_k=10) # the most recent ten rows
fs.all                         # the entity's whole history
```

Naming a window is what makes sharing *provable*. Two features that both ask
for `w30d` ask for one mask; two features with inline 30-day lookbacks only
happen to agree, and a planner would have to prove it.

## Point-in-time correctness is a column, not a promise

By default a lookback ends at the entity's own most recent row. That is exactly
right at serving time, where the newest row *is* the transaction being scored.
It is quietly wrong for a training backfill assembled later, where the newest
row is wherever the extract happened to stop.

Declare the decision moment and the difference is stark:

```python
fs = FeatureSet(entity="account", time="ts", reference_time="decided_at")
```

The sample ledger has this built in: 30% of its rows fall after the decision
they are attached to, across 116 of its 120 accounts. Anchoring decides what
`txn_count_30d` even means:

```text
            anchored on newest row   anchored on decided_at
account_id
ACC0000                        1.0                      0.0
ACC0001                        4.0                      3.0
ACC0002                        8.0                      2.0
ACC0003                        3.0                      3.0
ACC0004                        6.0                      1.0

107 of 120 accounts differ.
```

Nothing about the left column looks wrong. With `reference_time` declared,
every window — including `all` — ends at the decision moment, and prism also
tells you that the extract contained those rows at all, because any other
consumer of the same frame that does not exclude them is leaking.

## The planner is checked against itself on every build

Verification runs by default and raises on any disagreement:

```python
plan = fs.plan(frame)                # verifies, then returns
plan = fs.plan(frame, verify=False)  # skip once a plan is known-good
```

It executes the same features a second way — one entity at a time, no sharing,
no batching — and compares. That reference executor is slow and obviously
correct, and it is right by definition: a disagreement is a planner bug.

A planner that silently returns different numbers than the code it replaced is
worse than no planner, because the failure is invisible and arrives weeks later
as a model that underperforms for no visible reason. So the default is to find
out at build time.

Entities are sampled, never rows. A feature reads an entity's whole history, so
dropping rows would not sample the computation — it would change it, and then
compare two different questions.

## Porting existing code is a separate claim, checked separately

The plan agreeing with its own reference executor says nothing about whether it
agrees with the hand-written function you are replacing. That is a different
claim and gets a different check:

```python
report = plan.verify_against(compute_account_features, frame)
print(report.describe())
```

Your function receives one entity's rows and returns a mapping of name to
value — the shape hand-written feature code already has. Only features present
in both are compared, so an existing set can be ported a few at a time with
evidence at every step.

Tolerance defaults to a relative `1e-9`: tight enough to catch a logic error,
loose enough to survive the reassociation that legitimately differs between a
loop and a grouped reduction in `log1p`, `std`, `entropy` and division.

## Column order is part of the contract

A model consumes a feature vector by position. Train on one order and serve
another and nothing raises — every value is a plausible float in the wrong
slot.

```python
contract = plan.contract()
contract.validate_output(run.features)   # raises on a permuted vector
contract.to_json("feature_contract.json")
```

The contract carries the declared order, each feature's default, and the source
columns the features actually read — derived from the expression tree, so the
declared read set cannot drift from what the code touches.

## What the report says

Running a plan produces the features **and** a report, because computing one
means computing the other.

- **Duplicate definitions** — two names, one expression tree. Structural, so
  there are no false positives, and it holds on every dataset rather than
  just this one.
- **Redundancy** — features that are defined differently but move together
  almost exactly. Reported as an alert, not a defect, and suppressed for pairs
  already reported as structurally identical.
- **Constant features** — one value for every entity, so it cannot separate
  them.
- **Fragility** — features that fell back to their default. A fallback is a
  computation that produced *nothing usable*, which is deliberately not the
  same as a value that happens to equal the default; conflating the two
  produces a false alarm on perfectly healthy data.
- **Leakage** — a feature that reads the declared target, read off the
  expression tree rather than guessed from names.
- **Point-in-time** — rows in the frame from after the decision moment.
- **Cost** — which feature is doing the expensive work.

Every one of these was calibrated against a clean, deliberately varied feature
set and produces **nothing** on it. That is the repository's rule: a detector
that never stays quiet is not a detector, and one that fires on healthy data is
worse than absent because it teaches you to skip the section.

## Cost is measured, but the finding is not

The report shows measured milliseconds per feature. The *finding* about cost
concentration is computed from the operations in the plan instead, using static
weights.

This is not fastidiousness. The timing-based version of that detector fired on
clean data the first time the suite ran on a loaded machine. Wall-clock time is
a property of the machine, not of your data, so it is reported as a measurement
of one run and never banked as evidence — an evidence id is a hash of its
value, and a duration in there would give the same analysis a different lineage
every time.

## Limits worth knowing

- **pandas only.** A Polars executor is planned; the intermediate
  representation is semantic rather than pandas-shaped so it can be added
  without changing any feature definition.
- **Single table.** Cross-table features are planned. Join keys will be
  proposed and require confirmation — prism does not join on an inferred key,
  because an inferred key is a candidate until you confirm it.
- **No serving path yet.** The one-definition-two-paths claim is architectural
  today: the entity-scoped model makes a single-row path the n=1 case of the
  same tree, but the state artifact that a served grouped feature needs is not
  built.
- **The algebra is closed.** Eight families cover the 44-feature production set
  it was designed against. Anything outside them belongs in `@fs.opaque`, which
  is correct but unplanned rather than approximated by something that nearly
  fits.
- **Redundancy is measured on your data.** A pair that is redundant here may
  separate on data with a wider range.

## See also

- [Results & evidence](results-and-evidence.md) — findings, evidence, statuses
- [Exporting reports](exporting-reports.md) — HTML and JSON output
- [Regression readiness](regression.md) — leakage screening on a modelling target
