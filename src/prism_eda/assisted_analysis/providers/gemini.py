"""Google Gemini / Gemma provider.

Uses the ``google-genai`` SDK and the portable prompted-JSON protocol (see
``_protocol.py``), so it works with both Gemini models and the smaller Gemma
models that don't support native function-calling or system instructions.

Install with the extra::

    pip install "prism-eda[ai-gemini]"
"""

from __future__ import annotations

import os
import time
from typing import Any

from prism_eda.assisted_analysis.providers._protocol import (
    parse_decision,
    render_prompt,
)
from prism_eda.assisted_analysis.providers.base import (
    DecisionRequest,
    LLMProvider,
    ProviderDecision,
    ProviderError,
)

#: HTTP-ish status codes worth retrying (rate limit + transient server errors).
_TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}


def _is_transient(error: Exception) -> bool:
    """Heuristically decide whether an SDK error is worth retrying."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if code in _TRANSIENT_CODES:
        return True
    name = type(error).__name__.lower()
    return any(
        token in name
        for token in ("timeout", "connection", "servererror", "unavailable")
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
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._max_retries = max(1, max_retries)
        self._retry_backoff = retry_backoff
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
        max_retries: int = 3,
        retry_backoff: float = 2.0,
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
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    def available_models(self) -> list[str]:
        """Return model ids visible to this account (handy for confirming a name)."""
        return [model.name for model in self._client.models.list()]

    def decide(self, request: DecisionRequest) -> ProviderDecision:
        prompt = render_prompt(request)
        text = self._generate(prompt)
        return parse_decision(text)

    def _generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._generate_once(prompt)
            except Exception as error:  # noqa: BLE001 - re-raised below as clean error
                last_error = error
                if not _is_transient(error) or attempt == self._max_retries - 1:
                    break
                time.sleep(self._retry_backoff * (2**attempt))
        raise ProviderError(
            f"Gemini request to model {self.model!r} failed after "
            f"{self._max_retries} attempt(s): {last_error}. "
            "If this is a 500/503 it is usually transient — retry, or try a "
            "different model (e.g. model='gemini-2.5-flash')."
        ) from last_error

    def _generate_once(self, prompt: str) -> str:
        genai_types = _import_genai_types()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                    response_mime_type="application/json",
                ),
            )
        except Exception as error:
            # Some models reject response_mime_type with a 4xx; for that case
            # only, retry once without it. Transient errors propagate to the
            # retry loop in _generate.
            if _is_transient(error):
                raise
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                ),
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
