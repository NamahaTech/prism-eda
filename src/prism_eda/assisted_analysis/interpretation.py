"""Grounded interpretation pass — the AI layer's actual value-add.

The deterministic engine owns detection, magnitude, and confidence. Restating
those numbers is exactly where an LLM adds nothing, so this pass instead asks the
model for the judgment a statistic can't supply: what the columns *mean*, what the
findings mean *together*, and what to do next. Each is a small, focused prompt
grounded only in evidence the engine already produced (plus privacy-gated column
aggregates), reasoned in prose (JSON-mode degrades small models), and allowed to
abstain rather than fabricate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from prism_eda.catalog.models import DatasetCatalog
from prism_eda.evidence.models import Evidence, Finding
from prism_eda.privacy.models import PrivacyPolicy

_MAX_FINDINGS = 8
_MAX_COLUMNS = 40
_MAX_COLUMN_READS = 14
_MAX_NEXT_STEPS = 4
_MAX_RELATIONSHIPS = 12


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if number.is_integer():
        return f"{int(number)}"
    return f"{number:.4g}"


def _stat_summary(statistics: dict[str, Any]) -> str:
    parts = [
        f"{key}={_fmt(statistics[key])}"
        for key in ("min", "max", "median", "mean", "std")
        if statistics.get(key) is not None
    ]
    return ", ".join(parts)


def _columns_block(
    catalog: DatasetCatalog, privacy: PrivacyPolicy
) -> tuple[str, set[str]]:
    """Privacy-safe per-column aggregate lines, plus the set of names disclosed.

    Aggregates (type, range, missingness, cardinality) are always safe to send.
    The actual category labels are values, so they are included only when the
    caller opted in via ``privacy.allow_raw_values``.
    """
    lines: list[str] = []
    disclosed: set[str] = set()
    for table in catalog.tables:
        for column in table.columns:
            safe = privacy.safe_column_name(column.name)
            if safe is None:
                continue
            disclosed.add(safe)
            bits = [f"{safe} ({column.semantic_type}/{column.physical_type})"]
            stat = _stat_summary(column.statistics)
            if stat:
                bits.append(stat)
            if column.unique_count is not None:
                bits.append(f"{column.unique_count} distinct")
            if column.missing_rate:
                bits.append(f"{column.missing_rate:.0%} missing")
            if privacy.allow_raw_values and column.top_values:
                labels = [
                    str(item.get("value"))
                    for item in column.top_values[:3]
                    if item.get("value") is not None
                ]
                if labels:
                    bits.append("top: " + ", ".join(labels))
            lines.append("- " + "; ".join(bits))
            if len(lines) >= _MAX_COLUMNS:
                return "\n".join(lines), disclosed
    return "\n".join(lines), disclosed


def _findings_block(findings: Sequence[Finding]) -> str:
    lines = []
    for index, finding in enumerate(findings[:_MAX_FINDINGS], start=1):
        lines.append(
            f"{index}. {finding.title} — {finding.summary} [{finding.severity}]"
        )
    return "\n".join(lines)


def _safe_columns(columns: Sequence[str], privacy: PrivacyPolicy) -> str:
    names = [privacy.safe_column_name(str(column)) for column in columns]
    if any(name is None for name in names):
        return "(key)"
    return " + ".join(name for name in names if name)


def _relationships_block(
    evidence: Sequence[Evidence],
    privacy: PrivacyPolicy,
    *,
    limit: int = _MAX_RELATIONSHIPS,
) -> tuple[str, list[dict[str, Any]]]:
    """The strongest candidate FKs as a numbered block, plus their mapping.

    Numbering keeps the model's reply easy to ground (we map ``N`` back to a real
    relationship) and caps the prompt so a hub-heavy schema with hundreds of
    candidates doesn't blow the context — we name the highest-confidence ones.
    """
    relationships = sorted(
        (item for item in evidence if item.kind == "candidate_relationship"),
        key=lambda item: item.confidence,
        reverse=True,
    )
    lines: list[str] = []
    ordered: list[dict[str, Any]] = []
    for item in relationships[:limit]:
        value = item.value
        child_cols = _safe_columns(value["child_columns"], privacy)
        parent_cols = _safe_columns(value["parent_columns"], privacy)
        cardinality = str(value.get("cardinality", "")).replace("_", "-")
        ordered.append(
            {
                "parent": value["parent_table"],
                "child": value["child_table"],
                "cardinality": cardinality,
            }
        )
        lines.append(
            f"{len(ordered)}. {value['child_table']}.{child_cols} -> "
            f"{value['parent_table']}.{parent_cols} ({cardinality})"
        )
    return "\n".join(lines), ordered


def _ask(provider: Any, prompt: str) -> str | None:
    """Call the provider's text endpoint, treating any failure as an abstain."""
    try:
        text = provider.respond(prompt)
    except NotImplementedError:
        raise
    except Exception:  # noqa: BLE001 - a flaky micro-task must not sink the report
        return None
    return text.strip() if text else None


def _read_columns(
    provider: Any, columns_block: str, disclosed: set[str]
) -> list[dict[str, str]]:
    if not columns_block:
        return []
    prompt = (
        "You are a senior data analyst. Below are columns from a dataset, described "
        "by aggregates only. For each column, infer in plain English what it most "
        "likely represents — its real-world meaning, a likely unit, and any data "
        "quality caveat you can justify from the aggregates. Be concrete but do NOT "
        "invent facts beyond what the aggregates support.\n\n"
        "Reply with one line per column, formatted exactly:\n"
        "<column_name>: <meaning / unit / caveat>\n"
        'If a column is genuinely unclear, write "<column_name>: unclear". '
        "Add no other commentary.\n\n"
        f"COLUMNS:\n{columns_block}"
    )
    text = _ask(provider, prompt)
    if not text:
        return []
    reads: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, _, meaning = line.partition(":")
        name = name.strip().lstrip("-*0123456789. ").strip()
        meaning = meaning.strip()
        # Ground it: only keep columns we actually disclosed, and skip abstentions.
        if name not in disclosed or name in seen or not meaning:
            continue
        if meaning.lower() in {"unclear", "unknown", "n/a", "none"}:
            continue
        seen.add(name)
        reads.append({"column": name, "meaning": meaning})
        if len(reads) >= _MAX_COLUMN_READS:
            break
    return reads


def _narrative(
    provider: Any, goal: str, summary: str, findings_block: str
) -> str | None:
    if not findings_block:
        return None
    prompt = (
        f"You are a senior data analyst writing the headline takeaway of an "
        f"automated {goal} report. The deterministic engine found:\n\n"
        f"SUMMARY: {summary}\n\nFINDINGS:\n{findings_block}\n\n"
        "In 2-4 sentences, explain what these findings MEAN taken together and the "
        "single most likely root cause or action. Ground every statement in the "
        "findings above; do not introduce specific numbers, columns, or "
        "relationships that are not listed. If the findings do not support a "
        "coherent interpretation, reply with exactly: NONE"
    )
    text = _ask(provider, prompt)
    if not text or text.strip().upper().rstrip(".") == "NONE":
        return None
    return text


def _next_steps(provider: Any, findings_block: str) -> list[str]:
    if not findings_block:
        return []
    prompt = (
        "Based on the findings below, list up to four concrete next analyses the "
        "analyst should run to confirm or act on them, most important first. One "
        'per line starting with "- ". Ground them in the findings; do not invent '
        "columns. If there is no useful next step, reply with exactly: NONE\n\n"
        f"FINDINGS:\n{findings_block}"
    )
    text = _ask(provider, prompt)
    if not text or text.strip().upper().rstrip(".") == "NONE":
        return []
    steps: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(("-", "*", "•")):
            continue
        step = line.lstrip("-*• ").strip()
        if step:
            steps.append(step)
        if len(steps) >= _MAX_NEXT_STEPS:
            break
    return steps


def _read_relationships(
    provider: Any, block: str, ordered: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not block:
        return []
    prompt = (
        "You are a senior data analyst. Below are candidate foreign-key "
        "relationships (child references parent), inferred from the data. For each, "
        "write one short sentence in plain BUSINESS terms describing what it means "
        "(for example: 'Each order belongs to one customer'). Base it only on the "
        "table and column names and the cardinality shown; if a relationship's "
        "meaning is genuinely unclear, write 'unclear'.\n\n"
        "Reply with one line per item, formatted exactly:\n"
        "<number>: <sentence>\nAdd no other commentary.\n\n"
        f"RELATIONSHIPS:\n{block}"
    )
    text = _ask(provider, prompt)
    if not text:
        return []
    reads: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in text.splitlines():
        number, _, sentence = line.strip().partition(":")
        number = number.strip().lstrip("-*. ").strip()
        if not number.isdigit():
            continue
        index = int(number) - 1
        sentence = sentence.strip()
        if index in seen or not (0 <= index < len(ordered)) or not sentence:
            continue
        if sentence.lower() in {"unclear", "unknown", "n/a", "none"}:
            continue
        seen.add(index)
        relationship = ordered[index]
        reads.append(
            {
                "parent": relationship["parent"],
                "child": relationship["child"],
                "cardinality": relationship["cardinality"],
                "reading": sentence,
            }
        )
    return reads


def interpret(
    provider: Any,
    *,
    catalog: DatasetCatalog,
    privacy: PrivacyPolicy,
    findings: Sequence[Finding],
    evidence: Sequence[Evidence],  # reserved for future evidence-level reads
    goal: str,
    summary: str,
) -> dict[str, Any]:
    """Run the grounded micro-tasks and return the interpretation layer.

    Returns an empty dict when the provider offers no text endpoint or nothing
    survived grounding/abstention — so the report simply omits the AI layer
    rather than showing an empty panel.
    """
    columns_block, disclosed = _columns_block(catalog, privacy)
    findings_block = _findings_block(findings)
    relationships_block, ordered_relationships = _relationships_block(evidence, privacy)
    try:
        column_reads = _read_columns(provider, columns_block, disclosed)
    except NotImplementedError:
        return {}  # provider can't do free-form text — no interpretation layer
    # Schema goal only: name the inferred relationships in business terms. Skipped
    # (no provider call) when the dataset has no candidate relationships.
    relationship_reads = _read_relationships(
        provider, relationships_block, ordered_relationships
    )
    narrative = _narrative(provider, goal, summary, findings_block)
    next_steps = _next_steps(provider, findings_block)

    if not (column_reads or relationship_reads or narrative or next_steps):
        return {}
    return {
        "narrative": narrative,
        "column_reads": column_reads,
        "relationship_reads": relationship_reads,
        "next_steps": next_steps,
        "shared_labels": bool(privacy.allow_raw_values),
    }
