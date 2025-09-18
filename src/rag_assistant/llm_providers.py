# src/rag_assistant/llm_providers.py
from __future__ import annotations

import logging
from typing import Protocol

from .config import settings

logger = logging.getLogger("rag-assistant")


class LLMClient(Protocol):
    """Minimal protocol each provider client should satisfy."""
    name: str
    def complete(self, prompt: str) -> str:  # pragma: no cover
        ...


# ------------------------- Null / offline -------------------------
class _NullProvider:
    """Offline/no-network provider used when LLM_PROVIDER='none'."""
    name = "none"

    def complete(self, prompt: str) -> str:  # pragma: no cover
        # Deterministic empty output; pipeline will fall back to extractive mode.
        return ""


# ------------------------- OpenAI -------------------------
def _init_openai() -> LLMClient:
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
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"OpenAI completion failed: {e}")
                return ""

    return _OpenAIClient()


# ------------------------- Groq -------------------------
def _init_groq() -> LLMClient:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set but LLM_PROVIDER='groq'")
    try:
        from groq import Groq  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Groq client import failed: {exc}") from exc

    client = Groq(api_key=settings.GROQ_API_KEY)
    model = settings.GROQ_MODEL  # e.g., "llama-3.1-8b-instant"

    class _GroqClient:
        name = "groq"

        def complete(self, prompt: str) -> str:  # pragma: no cover
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Groq completion failed: {e}")
                return ""

    return _GroqClient()


# ------------------------- Gemini -------------------------
def _init_gemini() -> LLMClient:
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
            try:
                resp = model.generate_content(prompt)
                return getattr(resp, "text", "") or ""
            except Exception as e:
                logger.warning(f"Gemini completion failed: {e}")
                return ""

    return _GeminiClient()


# ------------------------- Provider factory -------------------------
def get_provider() -> LLMClient:
    """
    Return an initialized provider client; returns _NullProvider for 'none'.
    Defaults to 'none' to avoid network calls in CI/tests.
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


# ------------------------- Unified entrypoint -------------------------
def safe_complete(prompt: str) -> str:
    """
    Route to the configured provider. On any failure, return "" so the caller
    can fall back to extractive answers — never raise from here.
    Logs one unmissable line per call.
    """
    try:
        provider = get_provider()
        # determine model name string for logging
        if provider.name == "groq":
            model = settings.GROQ_MODEL
        elif provider.name == "openai":
            model = settings.OPENAI_MODEL
        elif provider.name == "gemini":
            model = settings.GEMINI_MODEL
        else:
            model = "-"

        logger.info(
            "LLM call -> provider=%s model=%s prompt_chars=%d",
            provider.name, model, len(prompt)
        )
        return provider.complete(prompt) or ""
    except Exception as e:  # extreme fallback
        logger.warning(f"safe_complete failed; falling back to extractive. error={e}")
        return ""
