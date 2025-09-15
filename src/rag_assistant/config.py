from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default

@dataclass(frozen=True)
class Settings:
    # Project root
    ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # Paths
    DATA_DIR: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "")))
    CHROMA_DIR: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", "")))
    SYSTEM_PROMPT_PATH: Path = field(default_factory=lambda: Path(_env("SYSTEM_PROMPT_PATH", "")))

    # Retrieval
    EMBEDDING_MODEL: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    RETRIEVAL_MODE: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "knn").lower())  # "knn" or "mmr"
    K: int = field(default_factory=lambda: int(_env("K", "3")))

    # LLM selection (preferences are parsed from this string)
    LLM_PREFERENCE_STR: str = field(default_factory=lambda: _env("LLM_PREFERENCE", "openai,groq,gemini"))
    OPENAI_MODEL: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    GROQ_MODEL: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama3-70b-8192"))
    GEMINI_MODEL: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-1.5-pro"))

    # API keys (optional; presence enables the provider)
    OPENAI_API_KEY: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    GROQ_API_KEY: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    GEMINI_API_KEY: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))

    # Output
    DEFAULT_RETURN_FORMAT: str = field(default_factory=lambda: _env("DEFAULT_RETURN_FORMAT", "text").lower())  # "text" or "json"

    def __post_init__(self):
        # Fill default paths if envs unset
        if not str(self.DATA_DIR):
            object.__setattr__(self, "DATA_DIR", self.ROOT / "data")
        if not str(self.CHROMA_DIR):
            object.__setattr__(self, "CHROMA_DIR", self.ROOT / "storage")
        if not str(self.SYSTEM_PROMPT_PATH):
            object.__setattr__(self, "SYSTEM_PROMPT_PATH", self.ROOT / "src" / "rag_assistant" / "prompts" / "system_prompt.txt")

        # Normalize retrieval mode
        mode = self.RETRIEVAL_MODE.lower()
        if mode not in {"knn", "mmr"}:
            object.__setattr__(self, "RETRIEVAL_MODE", "knn")

        # Normalize return format
        fmt = self.DEFAULT_RETURN_FORMAT.lower()
        if fmt not in {"text", "json"}:
            object.__setattr__(self, "DEFAULT_RETURN_FORMAT", "text")

    @property
    def LLM_PREFERENCE(self) -> list[str]:
        """Parsed provider preference list, filtered to known providers."""
        allowed = {"openai", "groq", "gemini"}
        parts = [p.strip().lower() for p in self.LLM_PREFERENCE_STR.split(",")]
        return [p for p in parts if p in allowed]

settings = Settings()

