# src/rag_assistant/api.py

import os
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Literal

from fastapi import FastAPI, Request
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from .qa import run_qa
from .pipeline import run_pipeline
from .memory import SessionMemory

app = FastAPI(title="RAG Assistant API", version="0.1.0")

# -------------------- Logging (console + rotating file) --------------------
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("rag-assistant")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False

_ch = logging.StreamHandler()
_ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
_fh.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

if not logger.handlers:
    logger.addHandler(_ch)
    logger.addHandler(_fh)

# -------------------- Session Memory --------------------
memory = SessionMemory(max_turns=5)

# -------------------- Schemas --------------------
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class SourceScored(BaseModel):
    path: str
    page: Optional[int | str] = None
    metric: Literal["cosine"] = "cosine"
    score_type: Literal["distance"] = "distance"
    score: float
    href: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    sources_scored: Optional[list[SourceScored]] = None
    session_id: str

# -------------------- Endpoints --------------------
@app.get("/readyz")
def readyz():
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    persist = os.getenv("CHROMA_DIR", "./storage")

    embeddings = HuggingFaceEmbeddings(model_name=model)
    vs = Chroma(embedding_function=embeddings, persist_directory=persist)

    count = 0
    try:
        count = vs._collection.count()
    except Exception:
        try:
            count = len(vs.get(limit=1_000_000, include=[])["ids"])
        except Exception:
            count = -1

    return {"status": "ok" if count > 0 else "empty", "chunks": count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    request_id = str(uuid.uuid4())
    start = time.time()

    # Session handling
    session_id = req.session_id or str(uuid.uuid4())
    session_context = memory.get_context(session_id)

    # Run pipeline with session context + retrieval
    result = run_pipeline(req.question, return_format="json")
    answer = result["answer"]
    sources = result["sources"]

    # Also call run_qa to get structured scores for API response
    _, sources_scored = run_qa(req.question, include_scores=True)

    # Save Q&A back into memory
    memory.add_turn(session_id, req.question, answer)

    elapsed = time.time() - start
    logger.info(
        f"[{request_id}] session={session_id} "
        f"question='{req.question}' retrieved={len(sources)} "
        f"answer_chars={len(answer)} elapsed={elapsed:.2f}s"
    )

    if not sources:
        logger.warning(f"[{request_id}] No relevant context retrieved.")

    return AskResponse(
        answer=answer,
        sources=sources,
        sources_scored=sources_scored,
        session_id=session_id,
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
