# src/rag_assistant/pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel
from langchain.schema import Document

from .config import settings
from .providers import get_retriever
from .llm_providers import safe_complete


class PipelineResult(BaseModel):
    answer: str
    sources: List[str]


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


def _prepare_context(question: str, k: int) -> Tuple[str, List[str]]:
    """
    Retrieve top-k documents and return a stitched context plus their source paths.
    Deterministic, no network calls required.
    """
    retriever = get_retriever(persist_directory=settings.CHROMA_DIR, k=k)
    docs: List[Document] = retriever.get_relevant_documents(question)

    stitched = "\n".join(d.page_content for d in docs if d.page_content)
    sources = [str(d.metadata.get("source", "?")) for d in docs]
    return stitched.strip(), sources


def _answer_offline(stitched_context: str) -> str:
    """
    Offline/extractive fallback when no LLM provider is configured.
    Returns the stitched context directly or a safe no-answer message.
    """
    if stitched_context:
        return stitched_context
    return "No answer found in the indexed documents."


def _answer_with_llm(system_prompt: str, question: str, context: str) -> str:
    """
    Use the configured LLM provider if available (safe_complete handles 'none').
    """
    prompt = (
        f"{system_prompt}\n\n"
        "Given the CONTEXT below, answer the QUESTION concisely.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )
    completion = safe_complete(prompt).strip()
    return completion or _answer_offline(context)


def run_pipeline(question: str, *, return_format: str = "json") -> Dict[str, Any]:
    """
    Main entry used by the API and tests.
    - Retrieves top-k docs
    - Produces an answer via offline stitching or LLM (if configured)
    - Returns {'answer': str, 'sources': List[str]}
    """
    system_prompt = _load_system_prompt()
    context, sources = _prepare_context(question, k=settings.RETRIEVAL_K)

    # Try LLM first; safe_complete returns "" when provider is 'none'
    answer = _answer_with_llm(system_prompt, question, context)

    result = PipelineResult(answer=answer, sources=sources)
    if return_format.lower() == "json":
        return result.model_dump()
    return result.model_dump()
