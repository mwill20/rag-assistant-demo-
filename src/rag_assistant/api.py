# src/rag_assistant/api.py

import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from urllib.parse import quote
from pathlib import PurePosixPath

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---- App & static files ------------------------------------------------------

DATA_DIR = os.getenv("DATA_DIR", "./data")      # set to repo root externally to serve /docs too
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="RAG Assistant API", version="0.1.0")

# Serve your document corpus so the UI can click through to sources.
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")


# ---- Schemas -----------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ScoredSource(BaseModel):
    name: str                       # e.g., "docs/file.pdf" (relative path)
    href: Optional[str] = None      # absolute link for UI
    page: Optional[int] = None      # PDF page, if available
    distance: Optional[float] = None  # cosine distance (lower = closer)
    label: Optional[str] = None     # friendly basename for display


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    sources_scored: List[ScoredSource] = []


# ---- Utilities ---------------------------------------------------------------

def _norm_rel_path(path: str) -> str:
    """Normalize to a forward-slash relative path under DATA_DIR."""
    try:
        rel = os.path.relpath(path, start=DATA_DIR)
    except Exception:
        rel = path
    return rel.replace("\\", "/").lstrip("./")


def _href_for(rel_path: str, page: Optional[int], base_url: str) -> str:
    """Build a clickable /static URL; URL-encode path; add #page for PDFs."""
    encoded = quote(rel_path, safe="/:._-")
    url = f"{base_url.rstrip('/')}/static/{encoded}"
    return f"{url}#page={page}" if page is not None else url


def _label_for(rel_path: str, page: Optional[int]) -> str:
    """Human-friendly label: basename + optional (page N)."""
    base = PurePosixPath(rel_path).name or "Source"
    return f"{base} (page {int(page)})" if page is not None else base


def _build_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)


# ---- Health/ready endpoints ---------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Report vector store chunk count."""
    vs = _build_vectorstore()
    count = 0
    try:
        count = vs._collection.count()  # type: ignore[attr-defined]
    except Exception:
        try:
            count = len(vs.get(limit=1_000_000, include=[])["ids"])
        except Exception:
            count = -1
    return {"status": "ok" if count > 0 else "empty", "chunks": count}


# ---- Ask endpoint ------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    """
    Returns:
      - answer (LLM/extractive from pipeline)
      - sources (legacy list of names)
      - sources_scored (clickable links + cosine distance + label)
    """
    # 1) Final answer via your pipeline (keeps behavior intact)
    from .pipeline import run_pipeline  # lazy import
    pipeline_result = run_pipeline(req.question, return_format="json")

    answer = pipeline_result.get("answer", "")
    sources_legacy = [s.replace("\\", "/") for s in pipeline_result.get("sources", [])]

    # 2) Retrieve a diverse top-k with distances; dedupe (source,page)
    vs = _build_vectorstore()
    base = str(request.base_url).rstrip("/")
    scored: List[ScoredSource] = []

    try:
        pool_k = 40   # sample large pool
        final_k = 5   # keep top unique hits
        pool: List[Tuple[object, float]] = vs.similarity_search_with_score(req.question, k=pool_k)  # type: ignore[assignment]
        seen = set()

        for doc, score in pool:
            meta = getattr(doc, "metadata", {}) or {}
            raw_source = meta.get("source") or ""
            page = meta.get("page")

            key = (str(raw_source), int(page) if isinstance(page, (int, float)) else page)
            if key in seen:
                continue
            seen.add(key)

            rel = _norm_rel_path(raw_source)
            href = _href_for(rel, page, base)
            distance = float(score) if score is not None else None
            label = _label_for(rel, page)

            scored.append(ScoredSource(name=rel, href=href, page=page, distance=distance, label=label))
            if len(scored) >= final_k:
                break

    except Exception:
        scored = []

    # 3) Fallback to legacy names if scoring failed
    if not scored and sources_legacy:
        for s in sources_legacy:
            rel = _norm_rel_path(s)
            scored.append(
                ScoredSource(
                    name=rel,
                    href=_href_for(rel, None, base),
                    page=None,
                    distance=None,
                    label=_label_for(rel, None),
                )
            )

    return AskResponse(
        answer=answer,
        sources=[s.name for s in scored] if scored else sources_legacy,
        sources_scored=scored,
    )

