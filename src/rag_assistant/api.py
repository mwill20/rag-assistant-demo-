# src/rag_assistant/api.py

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from fastapi import FastAPI
from pydantic import BaseModel
from .config import settings  # kept light; no heavy imports at module import time

app = FastAPI(title="RAG Assistant API", version="0.1.0")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

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
    # Lazy import so server starts instantly; heavy deps load only when /ask is called
    from .pipeline import run_pipeline
    result = run_pipeline(req.question, return_format="json")
    return AskResponse(**result)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
