"""Run a deep exploration across one or more sources.

The run is a sequence of independent pieces — one per table, then
relationships, links, and hint-driven sections — and each finished piece can be
handed to a checkpoint store. A run that is interrupted at hour four resumes
from the pieces it already has instead of starting over.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from prism_eda._serialization import to_jsonable
from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import (
    Evidence,
    EvidenceScope,
    Finding,
    sort_findings,
    split_findings,
)
from prism_eda.exploration import checkpoint
from prism_eda.exploration.config import (
    CheckpointStore,
    ExploreConfig,
    Hints,
    Progress,
    ProgressCallback,
)
from prism_eda.exploration.hints import (
    HintResult,
    amount_evidence,
    entity_activity,
    hint_matches,
    label_analysis,
    time_pattern_evidence,
)
from prism_eda.exploration.network import NetworkResult, profile_network
from prism_eda.exploration.relationships import (
    RelationshipResult,
    discover_relationships,
)
from prism_eda.exploration.tables import (
    FINGERPRINT_METHOD,
    TableState,
    profile_one_table,
)
from prism_eda.results import (
    AnalysisFailure,
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)
from prism_eda.sources.base import Source, SourceError, TableInfo, select_tables

GOAL = "exploration"
#: Share of the progress bar given to each phase.
_PHASES = {"tables": 0.70, "relationships": 0.15, "links": 0.10, "hints": 0.05}


class _Reporter:
    """Turns internal steps into progress reports and events."""

    def __init__(
        self, callbacks: tuple[EventCallback, ...], progress: ProgressCallback | None
    ) -> None:
        self._callbacks = callbacks
        self._progress = progress

    def report(
        self, stage: str, message: str, fraction: float, done: int = 0, total: int = 0
    ) -> None:
        fraction = max(0.0, min(1.0, fraction))
        emit(
            self._callbacks,
            Event(
                EventKind.PROGRESS_UPDATED,
                message,
                stage=stage,
                progress=fraction,
                data={"done": done, "total": total},
            ),
        )
        if self._progress is not None:
            # Not guarded: raising from the progress callback cancels the run.
            self._progress(
                Progress(
                    stage=stage,
                    message=message,
                    done=done,
                    total=total,
                    fraction=fraction,
                )
            )


def _signature(config: ExploreConfig, hints: Hints) -> str:
    """Checkpoints made under different settings must not be mixed."""
    payload = json.dumps(
        {"config": to_jsonable(config), "hints": to_jsonable(hints), "v": 1},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _key(signature: str, kind: str, name: str = "") -> str:
    digest = hashlib.sha256(name.encode()).hexdigest()[:20] if name else "all"
    return f"explore-{signature}-{kind}-{digest}"


def _display_names(tables: Sequence[TableInfo]) -> dict[str, str]:
    """Shortest name that still tells every table apart."""
    bare = Counter(table.name for table in tables)
    qualified = Counter(table.qualified for table in tables)
    names = {}
    for table in tables:
        if bare[table.name] == 1:
            names[table.key] = table.name
        elif qualified[table.qualified] == 1:
            names[table.key] = table.qualified
        else:
            names[table.key] = table.key
    return names


def _sample_extras(state: TableState, sample: pd.DataFrame, hints: Hints) -> None:
    """Sections computed while the table's row sample is still in memory."""
    timestamp = hint_matches(hints.timestamp, state)
    if timestamp is None:
        # Without a hint, the busiest true timestamp column is the natural one.
        ranges = [
            entry
            for entry in state.time_ranges
            if state.column(entry["column"]).kind == "datetime"
        ]
        timestamp = ranges[0]["column"] if ranges else None
    if timestamp:
        item = time_pattern_evidence(state, sample, timestamp)
        if item is not None:
            state.evidence.append(item)
    amount = hint_matches(hints.amount, state)
    if amount:
        found = amount_evidence(state, sample, amount)
        if found is not None:
            state.evidence.append(found[0])
            state.findings.extend(found[1])


def _inventory(
    states: Sequence[TableState], sources: Sequence[Source]
) -> tuple[list[Evidence], list[Finding]]:
    evidence: list[Evidence] = []
    findings: list[Finding] = []
    rows = [
        {
            "table": state.display,
            "source": state.info.source,
            "schema": state.info.schema,
            "name": state.info.name,
            "kind": state.info.kind,
            "engine": state.info.engine,
            "rows": state.rows,
            "rows_exact": state.rows_exact,
            "columns": len(state.columns),
            "bytes": state.info.bytes,
            "status": state.status,
            "message": state.message,
            "identifier_columns": len(state.key_columns),
            "sample_rows": state.sample_rows,
        }
        for state in states
    ]
    inventory = Evidence.create(
        kind="explore_inventory",
        scope=EvidenceScope(),
        value={
            "sources": [source.describe() for source in sources],
            "tables": rows,
            "total_rows": sum(state.rows for state in states),
            "total_bytes": sum(state.info.bytes or 0 for state in states),
        },
        method="source_catalog_v1",
        description="Every table explored, with its size and status.",
        metadata={"basis": "full_table"},
    )
    evidence.append(inventory)

    empty = [state.display for state in states if state.status == "empty"]
    if empty:
        shown = ", ".join(empty[:12]) + (
            f" and {len(empty) - 12} more" if len(empty) > 12 else ""
        )
        findings.append(
            Finding.create(
                title="Empty tables",
                summary=f"{len(empty)} table(s) have no rows: {shown}.",
                severity="low",
                confidence=1.0,
                evidence_ids=(inventory.id,),
                recommendation=(
                    "Confirm whether these are unused or were expected to be "
                    "loaded; anything built on them will silently see nothing."
                ),
            )
        )

    ranges = [entry for state in states for entry in state.time_ranges[:1]]
    usable = []
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    for entry in ranges:
        try:
            start, end = pd.Timestamp(entry["min"]), pd.Timestamp(entry["max"])
        except (TypeError, ValueError):
            continue
        if end > now + pd.Timedelta(days=366) or start.year < 1900:
            continue  # placeholder dates would stretch the chart to nothing
        usable.append(entry)
    if usable:
        coverage = Evidence.create(
            kind="explore_time_coverage",
            scope=EvidenceScope(),
            value={
                "tables": [
                    {
                        "table": entry["table"],
                        "column": entry["column"],
                        "min": entry["min"],
                        "max": entry["max"],
                        "rows": entry["rows"],
                    }
                    for entry in usable
                ]
            },
            method="full_table_min_max_v1",
            description="The period each table covers, by its busiest time column.",
            metadata={"basis": "full_table"},
        )
        evidence.append(coverage)
    return evidence, findings


def explore_sources(
    sources: Sequence[Source],
    *,
    include: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
    config: ExploreConfig | None = None,
    hints: Hints | dict[str, Any] | None = None,
    callbacks: Sequence[EventCallback] = (),
    progress: ProgressCallback | None = None,
    checkpoints: CheckpointStore | None = None,
) -> AnalysisResult:
    """Explore every selected table of the given sources."""
    config = config or ExploreConfig()
    hints = Hints.parse(hints)
    reporter = _Reporter(tuple(callbacks), progress)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    signature = _signature(config, hints)
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("Each source needs a distinct name")
    by_name = {source.name: source for source in sources}
    emit(
        tuple(callbacks),
        Event(EventKind.RUN_STARTED, "Exploration started.", stage=GOAL),
    )

    failures: list[AnalysisFailure] = []
    warnings: list[AnalysisWarning] = []
    listed: list[TableInfo] = []
    reporter.report("inventory", "Listing tables", 0.0)
    for source in sources:
        try:
            listed.extend(source.tables())
        except SourceError as error:
            failures.append(
                AnalysisFailure(
                    stage="inventory",
                    message=f"{source.name}: {error}",
                    recoverable=False,
                )
            )
    if failures and not listed:
        raise SourceError(
            "No source could be listed: "
            + "; ".join(failure.message for failure in failures)
        )
    chosen = select_tables(listed, include=include, exclude=exclude)
    chosen.sort(key=lambda table: (table.source, table.schema or "", table.name))
    names = _display_names(chosen)

    # -- tables ------------------------------------------------------------
    states: list[TableState] = []
    total = len(chosen)
    for index, info in enumerate(chosen):
        display = names[info.key]
        base = _PHASES["tables"] * index / max(total, 1)
        # Computed the same way as the next table's base, so progress never
        # steps back by a rounding error.
        done = _PHASES["tables"] * (index + 1) / max(total, 1)
        key = _key(signature, "table", info.key)
        state = None
        if checkpoints is not None:
            saved = checkpoints.load(key)
            if saved is not None:
                state = checkpoint.loads(TableState, saved)
        if state is not None:
            reporter.report("tables", f"{display}: restored", done, index + 1, total)
        else:
            reporter.report("tables", f"{display}: starting", base, index, total)

            def tick(
                message: str,
                display: str = display,
                base: float = base,
                index: int = index,
            ) -> None:
                reporter.report("tables", f"{display}: {message}", base, index, total)

            state = profile_one_table(
                by_name[info.source],
                info,
                display,
                config,
                tick,
                extras=lambda state, sample: _sample_extras(state, sample, hints),
            )
            if checkpoints is not None:
                checkpoints.save(key, checkpoint.dumps(state))
            reporter.report("tables", f"{display}: done", done, index + 1, total)
        states.append(state)

    usable = [state for state in states if state.status in {"profiled", "partial"}]

    # -- relationships -----------------------------------------------------
    offset = _PHASES["tables"]
    relationships = RelationshipResult()
    if config.relationships and usable:
        key = _key(signature, "relationships", "|".join(s.key for s in usable))
        saved = checkpoints.load(key) if checkpoints is not None else None
        restored = checkpoint.loads(RelationshipResult, saved) if saved else None
        if restored is not None:
            relationships = restored
        else:
            reporter.report("relationships", "Comparing identifier columns", offset)
            relationships = discover_relationships(
                usable,
                by_name,
                config,
                lambda message: reporter.report("relationships", message, offset),
            )
            if checkpoints is not None:
                checkpoints.save(key, checkpoint.dumps(relationships))

    # -- links -------------------------------------------------------------
    offset += _PHASES["relationships"]
    network = NetworkResult()
    if config.network and usable:
        key = _key(signature, "network", "|".join(s.key for s in usable))
        saved = checkpoints.load(key) if checkpoints is not None else None
        restored_network = checkpoint.loads(NetworkResult, saved) if saved else None
        if restored_network is not None:
            network = restored_network
        else:
            reporter.report("links", "Profiling links between records", offset)
            network = profile_network(
                usable,
                by_name,
                relationships,
                config,
                lambda message: reporter.report("links", message, offset),
            )
            if checkpoints is not None:
                checkpoints.save(key, checkpoint.dumps(network))

    # -- hints -------------------------------------------------------------
    offset += _PHASES["links"]
    hinted: list[HintResult] = []
    if hints:
        reporter.report("hints", "Hint-driven analysis", offset)
        tick_hint = lambda message: reporter.report("hints", message, offset)  # noqa: E731
        hinted.append(
            label_analysis(
                usable, by_name, hints, relationships, network, config, tick_hint
            )
        )
        hinted.append(entity_activity(usable, by_name, hints, config, tick_hint))

    # -- assemble ----------------------------------------------------------
    reporter.report("assemble", "Assembling the result", 0.98)
    inventory_evidence, inventory_findings = _inventory(states, sources)
    evidence: list[Evidence] = list(inventory_evidence)
    findings: list[Finding] = list(inventory_findings)
    sampling: list[SamplingRecord] = []
    artifacts: list[Artifact] = list(relationships.artifacts)
    for state in states:
        evidence.extend(state.evidence)
        findings.extend(state.findings)
        warnings.extend(state.warnings)
        failures.extend(state.failures)
        sampling.extend(state.sampling)
        if state.status == "failed":
            warnings.append(
                AnalysisWarning(
                    code="table_not_profiled",
                    message=f"{state.display} could not be profiled: {state.message}",
                    table=state.display,
                )
            )
    evidence.extend(relationships.evidence)
    findings.extend(relationships.findings)
    failures.extend(relationships.failures)
    evidence.extend(network.evidence)
    findings.extend(network.findings)
    failures.extend(network.failures)
    for part in hinted:
        evidence.extend(part.evidence)
        findings.extend(part.findings)
        warnings.extend(part.warnings)
        failures.extend(part.failures)

    catalogs = tuple(state.catalog for state in states if state.catalog is not None)
    catalog = DatasetCatalog(
        fingerprint=hashlib.sha256(
            "|".join(table.fingerprint for table in catalogs).encode()
        ).hexdigest()[:32],
        fingerprint_method=FINGERPRINT_METHOD,
        table_count=len(catalogs),
        row_count=sum(table.row_count for table in catalogs),
        column_count=sum(table.column_count for table in catalogs),
        tables=catalogs,
    )
    findings = sort_findings(findings)
    issues, observations = split_findings(findings)
    strong = [item for item in relationships.relationships if item.tier != "possible"]

    if not states:
        status = AnalysisStatus.INSUFFICIENT_EVIDENCE
        summary = "No tables matched the selection, so nothing was explored."
        warnings.append(
            AnalysisWarning(
                "no_tables", "No tables matched the include/exclude patterns."
            )
        )
    else:
        problems = bool(failures) or any(state.status == "failed" for state in states)
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if problems
            else AnalysisStatus.COMPLETED
        )
        summary = (
            f"Explored {len(states)} table(s) holding {catalog.row_count:,} rows and "
            f"{catalog.column_count:,} columns. Found {len(strong)} relationship(s) "
            f"between columns, {len(network.links)} record-to-record link(s), "
            f"{len(issues)} data-quality issue(s) and {len(observations)} "
            "observation(s)."
        )
        if problems:
            summary += " Some steps could not be completed; see the warnings."

    result = AnalysisResult(
        goal=GOAL,
        status=status,
        summary=summary,
        catalog=catalog,
        findings=tuple(findings),
        evidence=tuple(evidence),
        artifacts=tuple(artifacts),
        warnings=tuple(warnings),
        failures=tuple(failures),
        sampling=tuple(sampling),
        metadata={
            "sources": [source.describe() for source in sources],
            "table_count": len(states),
            "tables_profiled": len(usable),
            "tables_empty": sum(1 for state in states if state.status == "empty"),
            "tables_failed": sum(1 for state in states if state.status == "failed"),
            "identifier_columns": relationships.columns_compared,
            "relationship_count": len(strong),
            "possible_relationship_count": len(relationships.relationships)
            - len(strong),
            "domain_count": len(relationships.domains),
            "link_count": len(network.links),
            "issue_count": len(issues),
            "observation_count": len(observations),
            "config": to_jsonable(dataclasses.asdict(config)),
            "hints": to_jsonable(hints),
            "started_at": started_at.isoformat(),
            "duration_seconds": round(time.monotonic() - started, 1),
            "random_seed": config.seed,
        },
    )
    emit(
        tuple(callbacks),
        Event(
            EventKind.RUN_COMPLETED,
            summary,
            stage=GOAL,
            progress=1.0,
            data={"status": status.value},
        ),
    )
    reporter.report("done", "Exploration finished", 1.0)
    return result
