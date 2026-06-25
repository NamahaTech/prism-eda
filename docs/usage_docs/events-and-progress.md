# Events & progress

Prism's core analysis never prints, never prompts, and never assumes it's running
in a notebook or terminal. Instead, long-running operations **emit events**, and
you subscribe with callbacks. This keeps the core usable from a script, a web
server, a notebook, or a GUI without change.

## Subscribing with a callback

A callback is any callable that takes a single `Event`. Pass a list of them via
`callbacks=`:

```python
import prism_eda as pe
from examples.sample_data import load_sample

def on_event(event: pe.Event) -> None:
    print(f"{event.kind}: {event.message}")

dataset = pe.load(load_sample())
result = dataset.profile(callbacks=[on_event])
```

```text
run_started: Baseline profile started.
stage_started: Creating structured evidence.
evidence_created: Overall dataset shape across all loaded tables.
evidence_created: Shape, memory, and duplicate summary for customers.
...
stage_completed: Structured evidence created.
run_completed: Profiled 2 table(s), 320 rows, and 11 columns; found 1 prioritized issue(s).
```

Every recipe and the top-level convenience functions accept `callbacks=`.

## The `Event` object

| Field | Type | Meaning |
|-------|------|---------|
| `kind` | `EventKind` | What happened (see table below) |
| `message` | `str` | Human-readable description |
| `stage` | `str \| None` | Which stage emitted it |
| `progress` | `float \| None` | Progress in `[0, 1]` where applicable |
| `data` | `dict` | Extra structured payload (e.g. `evidence_id`) |
| `created_at` | `datetime` | UTC timestamp |

`EventKind` values:

| Kind | Emitted when |
|------|--------------|
| `run_started` / `run_completed` / `run_failed` | The run begins / ends / aborts |
| `stage_started` / `stage_completed` | A stage begins / ends |
| `progress_updated` | Incremental progress within a stage |
| `sampling_selected` | The planner decided to sample |
| `warning_raised` | A non-fatal caveat |
| `metric_failed` | An optional metric failed (the run continues) |
| `evidence_created` | A piece of evidence was produced |
| `question_asked` | (Reserved for future interactive/assisted analysis) |

## A simple progress display

Use `stage`/`progress` to drive a status line or progress bar:

```python
def progress(event: pe.Event) -> None:
    if event.progress is not None:
        pct = int(event.progress * 100)
        print(f"[{pct:3d}%] {event.stage}: {event.message}")
    elif event.kind == pe.EventKind.RUN_COMPLETED:
        print("done:", event.message)

dataset.discover_schema(callbacks=[progress])
```

## Counting or filtering events

Callbacks are just functions, so collect or filter however you like:

```python
from collections import Counter

events: list[pe.Event] = []
dataset.profile(callbacks=[events.append])

print(dict(Counter(e.kind.value for e in events)))
```

```text
{'run_started': 1, 'stage_started': 1, 'evidence_created': 14, 'stage_completed': 1, 'run_completed': 1}
```

## Callbacks are observers — they can't break a run

A callback that raises is **isolated**: the exception is swallowed and analysis
continues uninterrupted. Observers must never be able to corrupt a result.

```python
def broken(event: pe.Event) -> None:
    raise RuntimeError("observer failure")

# The run still completes normally despite the broken callback.
result = dataset.profile(callbacks=[broken, on_event])
print(result.status)        # completed
```

You can register multiple callbacks; each is called independently and one
failing doesn't stop the others.

## Why events instead of printing

This design keeps the deterministic core free of I/O assumptions: a CLI can map
events to a progress bar, a web backend can stream them to the browser, a notebook
can render them as widgets — all from the same analysis code. The `question_asked`
kind is the seam where future AI-assisted, interactive investigation will plug in
without changing the deterministic contract. See the
[roadmap](../implementation-status.md) for what's coming in 0.2.
