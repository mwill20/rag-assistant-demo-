# src/rag_assistant/api.py

import os
from typing import Optional, Literal

from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

app = FastAPI(title="RAG Assistant API", version="0.1.0")


# ----- Schemas -----
class AskRequest(BaseModel):
    question: str


class SourceScored(BaseModel):
    path: str
    page: Optional[int | str] = None
    metric: Literal["cosine"] = "cosine"
    score_type: Literal["distance"] = "distance"  # lower is better
    score: float


class AskResponse(BaseModel):
    answer: str
    # Legacy field kept for backward compatibility and tests:
    sources: list[str]
    # New field with raw retrieval scores (cosine distance):
    sources_scored: Optional[list[SourceScored]] = None


# ----- Endpoints -----
@app.get("/readyz")
def readyz():
    """
    Liveness/readiness with a useful signal: how many chunks are in the vector store.
    Returns: {"status": "ok" | "empty", "chunks": int}
    """
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    persist = os.getenv("CHROMA_DIR", "./storage")

    embeddings = HuggingFaceEmbeddings(model_name=model)
    vs = Chroma(embedding_function=embeddings, persist_directory=persist)

    # Prefer fast count; fall back to a safe method if needed.
    count = 0
    try:
        count = vs._collection.count()  # internal but quick
    except Exception:
        try:
            # Fallback: count ids (bounded; fine for small local corpora)
            count = len(vs.get(limit=1_000_000, include=[])["ids"])
        except Exception:
            count = -1

    return {"status": "ok" if count > 0 else "empty", "chunks": count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Use QA path with scores; keep legacy sources list[str] too
    from .qa import run_qa

    answer, sources_scored = run_qa(req.question, include_scores=True)

    # Derive legacy sources (paths-only) from scored items
    sources_paths = (
        [item.get("path", "?") for item in sources_scored] if isinstance(sources_scored, list) else []
    )

    payload = {
        "answer": answer,
        "sources": sources_paths,
        "sources_scored": sources_scored,
    }
    return AskResponse(**payload)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

