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
    path = Path(settings.SYSTEM_PROMPT_PATH)
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return "You are a concise RAG assistant. Cite sources when possible."


# ---------------- Reliability knobs (env-configurable) ----------------
_SAFE_RETRIES = int(os.getenv("SAFE_COMPLETE_MAX_RETRIES", "2"))
_SAFE_TIMEOUT = float(os.getenv("SAFE_COMPLETE_TIMEOUT", "10.0"))  # seconds
_SAFE_RETRY_DELAY = float(os.getenv("SAFE_COMPLETE_RETRY_DELAY", "0.75"))  # seconds

# Stop-condition knobs
_MIN_CHARS = int(os.getenv("NO_ANSWER_MIN_CHARS", "80"))


def _robust_complete(prompt: str) -> str:
    last_err: Exception | None = None
    for attempt in range(1, _SAFE_RETRIES + 2):
        start = time.time()
        try:
            return safe_complete(prompt)
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
    return (
        "I couldn’t complete the generation step reliably. "
        "Based on the provided documents, I don’t have a supported answer."
    )


def _prepare_context(question: str, k: int) -> tuple[str, list[str]]:
    retriever = get_retriever(persist_directory=settings.CHROMA_DIR, k=k)
    results = retriever.vectorstore.similarity_search_with_score(question, k=k)
    docs = [doc for (doc, _score) in results]
    stitched = "\n".join(d.page_content for d in docs if d.page_content)
    sources = [str(d.metadata.get("source", "?")) for d in docs]
    return stitched.strip(), sources


def _answer_offline(effective_context: str) -> str:
    """
    Offline/extractive fallback when no LLM provider is configured.
    We now return the EFFECTIVE context (session memory + retrieved text).
    """
    if effective_context:
        return effective_context
    return "No answer found in the indexed documents."


def _answer_with_llm(
    system_prompt: str,
    question: str,
    context: str,
    sources: list[str],
    session_context: str | None = None,
) -> str:
    mem = (session_context or "").strip()
    mem_len = len(mem)

    if not sources and mem_len == 0:
        logger.warning("Stop-condition: no documents and no session memory.")
        return "I don’t know based on the provided documents."

    if (len(context) + mem_len) < _MIN_CHARS:
        logger.warning(
            f"Stop-condition: combined context too short ({len(context)+mem_len} < {_MIN_CHARS})."
        )
        return "I don’t know based on the provided documents."

    # Build the prompt, including session memory if present
    session_block = f"SESSION MEMORY:\n{mem}\n\n" if mem else ""
    prompt = (
        f"{system_prompt}\n\n"
        "Given the (optional) SESSION MEMORY and the retrieved CONTEXT below, answer the QUESTION concisely.\n\n"
        f"{session_block}"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )
    completion = _robust_complete(prompt).strip()

    # Offline fallback should include memory + retrieval so users see the effect
    effective_context = (session_block + f"CONTEXT:\n{context}").strip()
    return completion or _answer_offline(effective_context)


def run_pipeline(
    question: str,
    *,
    return_format: str = "json",
    session_context: str | None = None,
) -> dict[str, object]:
    system_prompt = _load_system_prompt()
    context, sources = _prepare_context(question, k=settings.RETRIEVAL_K)

    answer = _answer_with_llm(
        system_prompt,
        question,
        context,
        sources,
        session_context=session_context,
    )
    result = PipelineResult(answer=answer, sources=sources)
    return result.model_dump()

