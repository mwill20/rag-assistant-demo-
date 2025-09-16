# src/rag_assistant/llm_providers.py

from __future__ import annotations

from typing import Protocol

from .config import settings


class LLMClient(Protocol):
    """Minimal protocol each provider client should satisfy."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - thin wrapper
        ...


class _NullProvider:
    """Offline/no-network provider used in CI/tests when LLM_PROVIDER='none'."""

    name = "none"

    def complete(self, prompt: str) -> str:  # pragma: no cover - deterministic stub
        # Keep behavior deterministic for tests; downstream code should
        # rely on extractive answers / no-answer fallback when this is active.
        return ""


def _init_openai() -> LLMClient:
    """Create a tiny OpenAI client wrapper; requires OPENAI_API_KEY."""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set but LLM_PROVIDER='openai'")
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"OpenAI client import failed: {exc}") from exc

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    model = settings.OPENAI_MODEL

    class _OpenAIClient:
        name = "openai"

        def complete(self, prompt: str) -> str:  # pragma: no cover
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""

    return _OpenAIClient()


def _init_groq() -> LLMClient:
    """Create a tiny Groq client wrapper; requires GROQ_API_KEY."""
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set but LLM_PROVIDER='groq'")
    try:
        from groq import Groq  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Groq client import failed: {exc}") from exc

    client = Groq(api_key=settings.GROQ_API_KEY)
    model = settings.GROQ_MODEL

    class _GroqClient:
        name = "groq"

        def complete(self, prompt: str) -> str:  # pragma: no cover
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""

    return _GroqClient()


def _init_gemini() -> LLMClient:
    """Create a tiny Gemini client wrapper; requires GEMINI_API_KEY."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set but LLM_PROVIDER='gemini'")
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Gemini client import failed: {exc}") from exc

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model_name = settings.GEMINI_MODEL
    model = genai.GenerativeModel(model_name)

    class _GeminiClient:
        name = "gemini"

        def complete(self, prompt: str) -> str:  # pragma: no cover
            resp = model.generate_content(prompt)
            return getattr(resp, "text", "") or ""

    return _GeminiClient()


def get_provider() -> LLMClient | None:
    """
    Return an initialized provider client or None if LLM_PROVIDER is 'none'.

    Defaults to 'none' in CI/tests to avoid network calls.
    """
    prov = (settings.LLM_PROVIDER or "none").lower()
    if prov == "none":
        return _NullProvider()
    if prov == "openai":
        return _init_openai()
    if prov == "groq":
        return _init_groq()
    if prov == "gemini":
        return _init_gemini()
    raise ValueError(f"Unknown LLM_PROVIDER: {prov}")


def safe_complete(prompt: str) -> str:
    """
    Convenience helper: run a completion if a real provider is configured,
    otherwise return an empty string (offline mode).
    """
    client = get_provider()
    if not client or getattr(client, "name", "none") == "none":
        return ""
    return client.complete(prompt)
