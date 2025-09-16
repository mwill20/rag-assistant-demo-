# src/rag_assistant/llm.py

from __future__ import annotations

from .config import settings

# Expose module-level constants for backward compatibility with code/tests
LLM_PROVIDER: str = settings.LLM_PROVIDER  # "none" | "openai" | "groq" | "gemini"
OPENAI_MODEL: str = settings.OPENAI_MODEL
GROQ_MODEL: str = settings.GROQ_MODEL
GEMINI_MODEL: str = settings.GEMINI_MODEL


def get_active_provider() -> str:
    """
    Return the normalized active provider.
    Tests/CI default to 'none' to avoid network calls.
    """
    return (settings.LLM_PROVIDER or "none").lower()


def get_default_model() -> str:
    """
    Return a default model name for the active provider.
    """
    prov = get_active_provider()
    if prov == "openai":
        return settings.OPENAI_MODEL
    if prov == "groq":
        return settings.GROQ_MODEL
    if prov == "gemini":
        return settings.GEMINI_MODEL
    return "offline-none"
