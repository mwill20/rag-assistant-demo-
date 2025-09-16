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
    DATA_DIR: str = _env("DATA_DIR", str(Path("data").resolve()))
    CHROMA_DIR: str = _env("CHROMA_DIR", str(Path("storage").resolve()))

    # Search settings
    SEARCH_TYPE: str = _env("SEARCH_TYPE", "similarity")

    # System prompt configuration
    SYSTEM_PROMPT_PATH: str = _env(
        "SYSTEM_PROMPT_PATH", 
        str(Path("system_prompt.txt").resolve())
    )

    # LLM provider settings
    LLM_PROVIDER: str = _env("LLM_PROVIDER", "groq")
    OPENAI_MODEL: str = _env("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()
