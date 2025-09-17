# src/rag_assistant/pipeline.py

from __future__ import annotations

import os
import time
import logging
from pathlib import Path

from langchain.schema import Document
from pydantic import BaseModel

from .config import settings
from .llm_providers import safe_complete
from .providers import get_retriever

logger = logging.getLogger("rag-assistant")


class PipelineResult(BaseModel):
    answer: str
    sources: list[str]


def _load_system_prompt() -> str:
    """
    Load system prompt from settings.SYSTEM_PROMPT_PATH.
    Falls back to a safe, minimal prompt if the file is missing.
    """
    path = Path(settings.SYSTEM_PROMPT_PATH)
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            pass  # fall through to default
    return "You are a concise RAG assistant. Cite sources when possible."


# ---------------- Reliability knobs (env-configurable) ----------------
_SAFE_RETRIES = int(os.getenv("SAFE_COMPLETE_MAX_RETRIES", "2"))
_SAFE_TIMEOUT = float(os.getenv("SAFE_COMPLETE_TIMEOUT", "10.0"))  # seconds
_SAFE_RETRY_DELAY = float(os.getenv("SAFE_COMPLETE_RETRY_DELAY", "0.75"))  # seconds

# Stop-condition knobs
_MIN_CHARS = int(os.getenv("NO_ANSWER_MIN_CHARS", "80"))


def _robust_complete(prompt: str) -> str:
    """
    Wrap the existing safe_complete with retry + basic timeout.
    Timeout is enforced as a soft wall-clock budget per attempt.
    """
    last_err: Exception | None = None
    # retries=N means up to N+1 attempts total
    for attempt in range(1, _SAFE_RETRIES + 2):
        start = time.time()
        try:
            result = safe_complete(prompt)
            return result
        except Exception as e:
            last_err = e
        elapsed = time.time() - start
        if elapsed > _SAFE_TIMEOUT:
            last_err = TimeoutError(
                f"safe_complete exceeded {_SAFE_TIMEOUT:.1f}s on attempt {attempt}"
            )
        if attempt <= _SAFE_RETRIES:
            time.sleep(_SAFE_RETRY_DELAY)
        else:
            break

    # Final fallback: never crash the pipeline; provide a safe response
    return (
        "I couldn’t complete the generation step reliably. "
        "Based on the provided documents, I don’t have a supported answer."
    )


def _prepare_context(question: str, k: int) -> tuple[str, list[str]]:
    """
    Retrieve top-k documents and return a stitched context plus their source paths.
    Deterministic, no network calls required.
    """
    retriever = get_retriever(persist_directory=settings.CHROMA_DIR, k=k)
    results = retriever.vectorstore.similarity_search_with_score(question, k=k)
    docs = [doc for (doc, _score) in results]
    stitched = "\n".join(d.page_content for d in docs if d.page_content)
    sources = [str(d.metadata.get("source", "?")) for d in docs]
    return stitched.strip(), sources


def _answer_offline(stitched_context: str) -> str:
    """
    Offline/extractive fallback when no LLM provider is configured.
    """
    if stitched_context:
        return stitched_context
    return "No answer found in the indexed documents."


def _answer_with_llm(system_prompt: str, question: str, context: str, sources: list[str]) -> str:
    """
    Use the configured LLM provider if available (safe_complete handles 'none').
    Includes stop-condition checks to avoid wasteful or hallucinatory calls.
    """
    if not sources:
        logger.warning("Stop-condition triggered: no documents retrieved.")
        return "I don’t know based on the provided documents."

    if len(context) < _MIN_CHARS:
        logger.warning(
            f"Stop-condition triggered: context too short ({len(context)} chars < {_MIN_CHARS})."
        )
        return "I don’t know based on the provided documents."

    prompt = (
        f"{system_prompt}\n\n"
        "Given the CONTEXT below, answer the QUESTION concisely.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )
    completion = _robust_complete(prompt).strip()
    return completion or _answer_offline(context)


def run_pipeline(question: str, *, return_format: str = "json") -> dict[str, object]:
    """
    Main entry used by the API and tests.
    - Retrieves top-k docs
    - Produces an answer via offline stitching or LLM (if configured)
    """
    system_prompt = _load_system_prompt()
    context, sources = _prepare_context(question, k=settings.RETRIEVAL_K)

    answer = _answer_with_llm(system_prompt, question, context, sources)
    result = PipelineResult(answer=answer, sources=sources)

    if return_format.lower() == "json":
        return result.model_dump()
    return result.model_dump()
