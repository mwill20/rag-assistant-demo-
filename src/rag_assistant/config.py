# src/rag_assistant/config.py

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _to_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings(BaseSettings):
    # Load from environment and optional .env; ignore unknown keys to stay flexible
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Core data/DB paths ---
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "storage")

    # --- Embeddings / Retrieval ---
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "2"))
    USE_MMR: bool = _to_bool(os.getenv("USE_MMR"), default=False)

    # --- System prompt file (used by pipeline/tests) ---
    SYSTEM_PROMPT_PATH: str = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")

    # --- LLM provider knobs (tests default to offline: provider="none") ---
    LLM_PROVIDER: str = os.getenv(
        "LLM_PROVIDER", "none"
    )  # none | openai | groq | gemini

    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # API keys are optional; only required when corresponding provider is enabled
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")


settings = Settings()
