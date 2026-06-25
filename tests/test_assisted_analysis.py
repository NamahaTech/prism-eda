"""Tests for the optional AI-assisted investigation layer.

These require the ``ai-gemini`` extra; the whole module is skipped when LangGraph
is not installed, so the core test job stays green without it.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("langgraph")

import prism_eda as pe
from prism_eda.assisted_analysis import (
    FakeProvider,
    GeminiProvider,
    Investigator,
)
from prism_eda.assisted_analysis.providers._protocol import (
    parse_decision,
    render_prompt,
)
from prism_eda.assisted_analysis.providers.base import (
    DecisionRequest,
    DraftFinding,
    ProviderDecision,
    ToolInvocation,
    ToolSpec,
)
from prism_eda.privacy import ColumnPolicy, PrivacyPolicy


def _sample() -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "plan": ["free", "pro", "free", "pro", "free", "pro", "free", "pro"],
            "leak": [0, 1, 0, 1, 0, 1, 0, 1],
            "source": ["web"] * 8,  # constant column -> a baseline profile finding
            "churned": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    return {"customers": customers}


def test_default_investigation_cites_real_evidence() -> None:
    dataset = pe.load(_sample())
    original = dataset.table("customers").copy(deep=True)

    investigator = Investigator(dataset, provider=FakeProvider(), max_steps=5)
    result = investigator.start(
        goal="classification", context={"target": "churned"}
    ).run()

    # Non-mutation.
    pd.testing.assert_frame_equal(dataset.table("customers"), original)

    assert result.status in {
        pe.AnalysisStatus.COMPLETED,
        pe.AnalysisStatus.COMPLETED_WITH_WARNINGS,
    }
    assert result.findings
    evidence_ids = {item.id for item in result.evidence}
    for finding in result.findings:
        assert finding.evidence_ids
        assert set(finding.evidence_ids) <= evidence_ids
    assert result.metadata["engine"] == "assisted"
    assert result.metadata["provider"] == "fake"
    # The AI disclosure is always present in assumptions.
    assert any("AI investigator" in a for a in result.assumptions)


def test_uncited_findings_are_dropped() -> None:
    dataset = pe.load(_sample())

    def finish(request: DecisionRequest) -> ProviderDecision:
        real = request.available_evidence_ids[0]
        return ProviderDecision(
            kind="finish",
            summary="One real, one fabricated.",
            findings=(
                DraftFinding(
                    title="Backed by evidence",
                    summary="cites a real id",
                    severity="high",
                    evidence_ids=(real,),
                ),
                DraftFinding(
                    title="Hallucinated",
                    summary="cites a fake id",
                    severity="critical",
                    evidence_ids=("ev_does_not_exist",),
                ),
            ),
        )

    script = [
        ProviderDecision(
            kind="call_tool",
            tool_calls=(ToolInvocation(tool="profile_dataset"),),
        ),
        finish,
    ]
    result = (
        Investigator(dataset, provider=FakeProvider(script=script), max_steps=5)
        .start(goal="profile")
        .run()
    )

    titles = {f.title for f in result.findings}
    assert "Backed by evidence" in titles
    assert "Hallucinated" not in titles  # dropped for citing a non-existent id
    assert result.metadata["uncited_findings_dropped"] == 1


def test_insufficient_evidence_when_model_finds_nothing() -> None:
    dataset = pe.load(_sample())
    script = [
        ProviderDecision(
            kind="insufficient",
            summary="Nothing conclusive.",
            status="insufficient_evidence",
        )
    ]
    result = (
        Investigator(dataset, provider=FakeProvider(script=script))
        .start(goal="profile")
        .run()
    )

    assert result.status == pe.AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert result.findings == ()


def test_unknown_tool_is_recoverable() -> None:
    dataset = pe.load(_sample())
    script = [
        ProviderDecision(
            kind="call_tool",
            tool_calls=(ToolInvocation(tool="not_a_real_tool"),),
        ),
        ProviderDecision(
            kind="insufficient", status="insufficient_evidence", summary="stop"
        ),
    ]
    # Must not raise; the unknown tool is reported back, run continues.
    result = (
        Investigator(dataset, provider=FakeProvider(script=script))
        .start(goal="profile")
        .run()
    )
    assert result.status == pe.AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_not_converged_falls_back_to_deterministic_findings() -> None:
    dataset = pe.load(_sample())
    # Always wants another tool call; never finishes -> exhausts the budget.
    keep_digging = ProviderDecision(
        kind="call_tool",
        tool_calls=(ToolInvocation(tool="profile_dataset"),),
    )
    result = (
        Investigator(
            dataset, provider=FakeProvider(script=[keep_digging] * 6), max_steps=2
        )
        .start(goal="profile")
        .run()
    )

    assert result.status == pe.AnalysisStatus.COMPLETED_WITH_WARNINGS
    assert any(w.code == "investigator_not_converged" for w in result.warnings)
    assert result.metadata["converged"] is False


def test_exports_work_on_assisted_result() -> None:
    dataset = pe.load(_sample())
    result = (
        Investigator(dataset, provider=FakeProvider())
        .start(goal="classification", context={"target": "churned"})
        .run()
    )

    html = result.render_html()
    assert html.startswith("<!doctype html>")
    assert "AI-assisted" in html  # provenance shown in the report footer
    payload = result.to_dict()
    assert payload["metadata"]["engine"] == "assisted"


# --- protocol -------------------------------------------------------------


def test_protocol_parses_tool_call_and_finish() -> None:
    call = parse_decision(
        '{"action": "call_tool", "tool": "profile_dataset", "arguments": {}}'
    )
    assert call.kind == "call_tool"
    assert call.tool_calls[0].tool == "profile_dataset"

    fenced = parse_decision(
        "```json\n"
        '{"action": "finish", "status": "completed", "summary": "ok", '
        '"findings": [{"title": "t", "summary": "s", "severity": "high", '
        '"confidence": 0.9, "evidence_ids": ["ev_1"]}]}\n'
        "```"
    )
    assert fenced.kind == "finish"
    assert fenced.findings[0].evidence_ids == ("ev_1",)
    assert fenced.findings[0].severity == "high"

    insufficient = parse_decision(
        '{"action": "finish", "status": "insufficient_evidence", "findings": []}'
    )
    assert insufficient.kind == "insufficient"


def test_protocol_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_decision("I am a helpful assistant and cannot output JSON.")


def test_render_prompt_surfaces_target_and_tools() -> None:
    request = DecisionRequest(
        goal="classification",
        dataset_overview="1 table",
        tools=(ToolSpec(name="profile_dataset", description="d"),),
        target="churned",
        remaining_steps=3,
    )
    prompt = render_prompt(request)
    assert "TARGET COLUMN: churned" in prompt
    assert "profile_dataset" in prompt
    assert "EXACTLY ONE JSON object" in prompt


# --- Gemini provider (no network) ----------------------------------------


class _FakeModels:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config=None):  # noqa: ANN001
        self.calls.append({"model": model, "contents": contents})

        class _Response:
            text = self._text

        return _Response()


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.models = _FakeModels(text)


def test_gemini_provider_with_mocked_client() -> None:
    client = _FakeClient(
        '{"action": "call_tool", "tool": "list_tables", "arguments": {}}'
    )
    provider = GeminiProvider(client=client, model="gemma-4-31b-it")
    decision = provider.decide(
        DecisionRequest(goal="profile", dataset_overview="x", tools=())
    )
    assert decision.kind == "call_tool"
    assert decision.tool_calls[0].tool == "list_tables"
    assert client.models.calls[0]["model"] == "gemma-4-31b-it"


def test_gemini_from_env_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="No API key"):
        GeminiProvider.from_env()


# --- privacy --------------------------------------------------------------


def test_privacy_excludes_columns_from_model_context() -> None:
    from prism_eda.assisted_analysis.tools import dataset_overview

    dataset = pe.load(_sample())
    policy = PrivacyPolicy(columns={"leak": ColumnPolicy("exclude")})
    overview = dataset_overview(dataset, policy)
    assert "leak" not in overview
    assert "churned" in overview  # other columns still present


# --- isolation ------------------------------------------------------------


def test_core_import_does_not_pull_in_llm_dependencies() -> None:
    code = (
        "import sys, prism_eda; "
        "assert 'langgraph' not in sys.modules, 'core imported langgraph'; "
        "assert 'google.genai' not in sys.modules, 'core imported google.genai'; "
        "print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
