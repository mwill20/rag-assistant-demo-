# src/rag_assistant/pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from pydantic import BaseModel

from .config import settings
from .llm_providers import safe_complete

# Build retriever locally so we can force MMR (diversified results)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


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


def _build_retriever(k: int):
    """
    Create a Chroma retriever with MMR (maximal marginal relevance) to reduce duplicates.
    - k: number of final results returned to the pipeline
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vs = Chroma(embedding_function=embeddings, persist_directory=settings.CHROMA_DIR)
    # MMR diversifies results while staying relevant
    fetch_k = max(8, k * 8)
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.7},
    )


def _prepare_context(question: str, k: int) -> Tuple[str, List[str]]:
    """
    Retrieve top-k documents and return a stitched context plus their source paths.
    Deterministic, no network calls required.
    """
    retriever = _build_retriever(k)
    docs: List[Document] = retriever.invoke(question)  # modern API

    stitched = "\n".join(d.page_content for d in docs if d.page_content)
    # Normalize paths to forward slashes for consistency with API/static links
    sources = [str(d.metadata.get("source", "?")).replace("\\", "/") for d in docs]
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


def run_pipeline(question: str, *, return_format: str = "json") -> dict[str, object]:
    """
    Main entry used by the API and tests.
    - Retrieves top-k docs (MMR retriever)
    - Produces an answer via offline stitching or LLM (if configured)
    - Returns {'answer': str, 'sources': List[str]}
    """
    system_prompt = _load_system_prompt()
    context, sources = _prepare_context(question, k=settings.RETRIEVAL_K)

    # Try LLM first; safe_complete returns "" when provider is 'none'
    answer = _answer_with_llm(system_prompt, question, context)

    result = PipelineResult(answer=answer, sources=sources)
    return result.model_dump()
