from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(key: str, default: str) -> str:
    """Return a non-None env var (falling back to default)."""
    val = os.getenv(key)
    return val if val else default


@dataclass(frozen=True)
class Settings:
    # Paths
    DATA_DIR: str = _env("DATA_DIR", str(Path("data")))
    CHROMA_DIR: str = _env("CHROMA_DIR", str(Path("storage")))

    # Embeddings
    EMBEDDING_MODEL: str = _env(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # LLM provider knobs (used by llm.py)
    LLM_PROVIDER: str = _env("LLM_PROVIDER", "none")  # "openai" | "groq" | "none"
    OPENAI_MODEL: str = _env("OPENAI_MODEL", "gpt-4o-mini")
    GROQ_MODEL: str = _env("GROQ_MODEL", "llama-3.1-8b-instant")


settings = Settings()


