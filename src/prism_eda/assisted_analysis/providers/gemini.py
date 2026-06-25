"""Google Gemini / Gemma provider.

Uses the ``google-genai`` SDK and the portable prompted-JSON protocol (see
``_protocol.py``), so it works with both Gemini models and the smaller Gemma
models that don't support native function-calling or system instructions.

Install with the extra::

    pip install "prism-eda[ai-gemini]"
"""

from __future__ import annotations

import os
from typing import Any

from prism_eda.assisted_analysis.providers._protocol import (
    parse_decision,
    render_prompt,
)
from prism_eda.assisted_analysis.providers.base import (
    DecisionRequest,
    LLMProvider,
    ProviderDecision,
)

#: Default model. Gemma is small and fast with a generous rate limit, which makes
#: it a good fit for iterating on the investigation flow. Override per instance.
DEFAULT_MODEL = "gemma-4-31b-it"

#: Environment variables checked, in order, by :meth:`GeminiProvider.from_env`.
_API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")


class GeminiProvider(LLMProvider):
    """Drive an investigation with a Google Gemini or Gemma model.

    Parameters:
        api_key: Google AI Studio API key. Kept only on the SDK client; never
            written to investigation state, events, or reports.
        model: Model id (e.g. ``"gemma-4-31b-it"``, ``"gemini-2.5-flash"``).
        temperature: Sampling temperature. Low by default for steadier tool use.
        max_output_tokens: Cap on a single reply.
        client: An optional pre-built ``google.genai.Client`` (used in tests).
    """

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        if client is not None:
            self._client = client
        else:
            genai = _import_genai()
            if not api_key:
                raise ValueError(
                    "GeminiProvider needs an api_key. Pass api_key=... or use "
                    "GeminiProvider.from_env() with GEMINI_API_KEY set."
                )
            self._client = genai.Client(api_key=api_key)

    @classmethod
    def from_env(
        cls,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
    ) -> GeminiProvider:
        """Build a provider, reading the API key from the environment.

        Checks ``GEMINI_API_KEY``, then ``GOOGLE_API_KEY``, then
        ``GOOGLE_GENAI_API_KEY``.
        """
        api_key = next(
            (os.environ[name] for name in _API_KEY_ENV_VARS if os.environ.get(name)),
            None,
        )
        if not api_key:
            raise ValueError(
                "No API key found. Set one of: " + ", ".join(_API_KEY_ENV_VARS)
            )
        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def available_models(self) -> list[str]:
        """Return model ids visible to this account (handy for confirming a name)."""
        return [model.name for model in self._client.models.list()]

    def decide(self, request: DecisionRequest) -> ProviderDecision:
        prompt = render_prompt(request)
        text = self._generate(prompt)
        return parse_decision(text)

    def _generate(self, prompt: str) -> str:
        genai_types = _import_genai_types()
        config = genai_types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            response_mime_type="application/json",
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except Exception:
            # Some Gemma models reject response_mime_type; retry without it.
            config = genai_types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty response.")
        return text


def _import_genai() -> Any:
    try:
        from google import genai
    except ImportError as error:  # pragma: no cover - exercised via extras guard
        raise ImportError(
            "GeminiProvider requires the 'ai-gemini' extra. Install it with: "
            "pip install 'prism-eda[ai-gemini]'"
        ) from error
    return genai


def _import_genai_types() -> Any:
    try:
        from google.genai import types
    except ImportError as error:  # pragma: no cover - exercised via extras guard
        raise ImportError(
            "GeminiProvider requires the 'ai-gemini' extra. Install it with: "
            "pip install 'prism-eda[ai-gemini]'"
        ) from error
    return types
