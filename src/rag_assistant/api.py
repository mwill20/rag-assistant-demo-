# src/rag_assistant/api.py
from __future__ import annotations

import os
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Literal
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from .qa import run_qa
from .pipeline import run_pipeline
from .memory import SessionMemory

# -------------------- App & Static --------------------
app = FastAPI(title="RAG Assistant API", version="0.1.0")

# Serve DATA_DIR at /static for clickable links in the UI
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")

# -------------------- Logging (console + rotating file) --------------------
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("rag-assistant")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    _fh.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
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
    href: Optional[str] = None  # HTTP deep-link (/static/...)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]  # legacy plain paths (repo-relative when possible)
    sources_scored: Optional[list[SourceScored]] = None
    session_id: str


# -------------------- Helpers --------------------
def _rel_or_plain(path_str: str) -> str:
    """
    Prefer repo-relative path under DATA_DIR when possible.
    Falls back to original string if path is outside DATA_DIR or invalid.
    """
    try:
        p = Path(path_str).resolve()
    except Exception:
        return path_str.replace("\\", "/")
    try:
        rel = p.relative_to(DATA_DIR)
        return str(rel).replace("\\", "/")
    except Exception:
        # Not under DATA_DIR — return normalized original
        return str(p).replace("\\", "/")


def _http_href_for(path_str: str, page: Optional[int | str], request: Request) -> str:
    """
    Build a browser-safe HTTP link to the file served at /static.
    - If file is under DATA_DIR, use /static/<relative>
    - Else, try to treat the given path as relative text (best-effort)
    - Add #page=N for PDFs when page is an int-like value
    """
    # Compute relative segment
    rel = _rel_or_plain(path_str)
    # If rel still looks absolute (outside DATA_DIR), strip drive and keep basename dir
    if Path(rel).is_absolute():
        rel = Path(rel).name

    # Base URL (scheme + host + prefix)
    base = str(request.base_url).rstrip("/")
    # Quote each segment to be URL-safe
    rel_parts = [quote(part) for part in rel.split("/")]
    url = f"{base}/static/" + "/".join(rel_parts)

    # Page anchor for PDFs (if provided)
    if page is not None:
        try:
            p = int(page)
            if p > 0:
                url += f"#page={p}"
        except Exception:
            pass
    return url


# -------------------- Endpoints --------------------
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

    # Main pipeline (uses system prompt + retrieval + memory context)
    result = run_pipeline(req.question, return_format="json", session_context=session_context)
    answer = result["answer"]
    raw_sources = result["sources"]

    # Normalize legacy sources to repo-relative where possible
    legacy_sources = [_rel_or_plain(s) for s in raw_sources]

    # Get scored results (k consistent with pipeline settings)
    _, sources_scored = run_qa(req.question, include_scores=True)

    # Enrich with HTTP hrefs for browser-friendly links
    enriched_scored: list[SourceScored] = []
    for item in sources_scored or []:
        path = item.get("path", "?")
        page = item.get("page")
        href = _http_href_for(path, page, request)
        enriched_scored.append(
            SourceScored(
                path=_rel_or_plain(path),
                page=page,
                metric=item.get("metric", "cosine"),
                score_type=item.get("score_type", "distance"),
                score=float(item.get("score", 0.0)),
                href=href,
            )
        )

    # Save Q/A to session memory
    memory.add_turn(session_id, req.question, answer)

    elapsed = time.time() - start
    logger.info(
        f"[{request_id}] session={session_id} question='{req.question}' "
        f"retrieved={len(legacy_sources)} answer_chars={len(answer)} elapsed={elapsed:.2f}s"
    )
    if not legacy_sources:
        logger.warning(f"[{request_id}] No relevant context retrieved.")

    return AskResponse(
        answer=answer,
        sources=legacy_sources,
        sources_scored=enriched_scored,
        session_id=session_id,
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
