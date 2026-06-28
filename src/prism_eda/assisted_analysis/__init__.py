"""Optional AI-assisted investigation layer (the ``ai-gemini`` extra).

This subpackage plans and explains analyses with a language model, but only over
Prism's deterministic tools — the model never sees raw data, never runs code, and
every finding it reports must cite real evidence. The deterministic core never
imports this package.

Install the extra::

    pip install "prism-eda[ai-gemini]"

Then::

    from prism_eda.assisted_analysis import Investigator, GeminiProvider

    investigator = Investigator(dataset, provider=GeminiProvider.from_env())
    result = investigator.start(goal="classification").run()
    result.to_html("investigation.html")
"""

from __future__ import annotations

try:
    import langgraph  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "AI-assisted analysis requires the 'ai-gemini' extra. Install it with: "
        "pip install 'prism-eda[ai-gemini]'"
    ) from error

from prism_eda.assisted_analysis.investigator import (
    InvestigationSession,
    Investigator,
)
from prism_eda.assisted_analysis.providers import (
    DecisionRequest,
    DraftFinding,
    FakeProvider,
    GeminiProvider,
    LLMProvider,
    ProviderDecision,
    ProviderError,
    ToolInvocation,
    ToolSpec,
)

__all__ = [
    "DecisionRequest",
    "DraftFinding",
    "FakeProvider",
    "GeminiProvider",
    "InvestigationSession",
    "Investigator",
    "LLMProvider",
    "ProviderDecision",
    "ProviderError",
    "ToolInvocation",
    "ToolSpec",
]
