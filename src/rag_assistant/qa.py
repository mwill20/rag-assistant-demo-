# src/rag_assistant/qa.py
from __future__ import annotations

import json
import os
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings  # expects EMBEDDING_MODEL, CHROMA_DIR


def _build_retriever(vs, k: int):
    """
    Retrieval mode via env:
      SEARCH_TYPE / RETRIEVAL_MODE = "similarity" (default) or "mmr"
      FETCH_K (MMR candidates, default 20)
      MMR_LAMBDA (diversity vs relevance, default 0.5)
    """
    search_type = (
        os.getenv("SEARCH_TYPE") or os.getenv("RETRIEVAL_MODE") or "similarity"
    ).lower()
    if search_type == "mmr":
        fetch_k = int(os.getenv("FETCH_K", "20"))
        mmr_lambda = float(os.getenv("MMR_LAMBDA", "0.5"))
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": mmr_lambda},
        )
    return vs.as_retriever(search_kwargs={"k": k})


def run_qa(question: str, k: int = 2) -> tuple[str, list[str]]:
    """
    Retrieve top-k chunks and return (answer_text, sources).
    Extractive by default (stitches retrieved chunks).
    Safer 'no-answer' heuristic: only refuse when we both
    (a) retrieve nothing useful AND (b) the top distance is very poor.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vs = Chroma(embedding_function=embeddings, persist_directory=settings.CHROMA_DIR)

    # Build retriever (MMR toggle supported)
    retriever = _build_retriever(vs, k=k)
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant context found.", []

    stitched = "\n".join(d.page_content for d in docs).strip()
    sources = [d.metadata.get("source", "?") for d in docs]

    # Tunables (generous defaults so tests/demos don’t trip the guard)
    max_dist = float(
        os.getenv("NO_ANSWER_MAX_DIST", "1.25")
    )  # cosine distance can be up to ~2
    min_chars = int(os.getenv("NO_ANSWER_MIN_CHARS", "80"))

    # Try to get a top distance; if unavailable, don’t block
    topd = None
    try:
        pairs = vs.similarity_search_with_score(question, k=min(max(1, k), len(docs)))
        if pairs:
            topd = pairs[0][1]
    except Exception:
        topd = None

    # Refuse only when BOTH conditions hold: (very poor distance) AND (very little content)
    if (topd is not None and topd > max_dist) and len(stitched) < min_chars:
        return "No relevant context found.", []

    return stitched or "No relevant context found.", sources


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this project?"
    answer, sources = run_qa(query)
    print(json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False))


if __name__ == "__main__":
    main()
