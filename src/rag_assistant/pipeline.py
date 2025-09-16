from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .config import settings
from .providers import get_provider_stack, get_retriever

class QAResponse(BaseModel):
    answer: str
    sources: list[str]

_DEF_PROMPT = (
    "You are a helpful RAG assistant. Use only the provided context. "
    "Cite file paths you used in a 'sources' list."
)


def _load_system_prompt() -> str:
    sp = Path(settings.SYSTEM_PROMPT_PATH)
    try:
        return sp.read_text(encoding="utf-8").strip()
    except Exception:
        return _DEF_PROMPT


def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s.strip("`")
    if s.endswith("```"):
        s = s.rsplit("\n", 1)[0]
    return s.strip()


def _unique(seq: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_sources(docs) -> list[str]:
    srcs = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        s = (
            meta.get("source")
            or meta.get("file_path")
            or meta.get("path")
            or meta.get("pdf_path")
        )
        if s:
            srcs.append(s)
    return _unique(srcs)


def _render_text(answer: str, sources: list[str]) -> str:
    answer = (answer or "").strip()
    cites = "\n".join(f"- {s}" for s in sources) if sources else "None"
    return f"{answer}\n\nSources:\n{cites}"


def _coerce_json(raw: str, context: str, sources: list[str]) -> dict:
    raw = _strip_code_fences(raw)
    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    ans = (data.get("answer") or "").strip()
    srcs = data.get("sources") or []

    # If a provider parrots our prompt back, fall back to context snippet.
    if ans.startswith("Answer using ONLY the CONTEXT below") or (
        "RETURN_FORMAT" in ans and "CONTEXT:" in ans
    ):
        ans = context[:600] or "No context found."

    if not srcs:
        srcs = sources

    resp = QAResponse(answer=ans, sources=_unique(srcs))
    return resp.model_dump()


def _count_words(s: str) -> int:
    tokens = re.findall(r"\b\w+\b", s or "")
    return len(tokens)


def _normalize_no_punct(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", s or "")
    return " ".join(s.split())


def _maybe_force_exact_words(question: str, answer: str) -> str:
    """
    If the question asks for 'exactly N words', try to enforce it:
    1) Ask the LLM to rewrite to exactly N words, no punctuation.
    2) If still off, trim to N if too long; otherwise return normalized answer.
    """
    m = re.search(r"exactly\s+(\d+)\s+words", question, flags=re.I)
    if not m:
        return answer
    n = int(m.group(1))
    norm = _normalize_no_punct(answer)
    if _count_words(norm) == n:
        return norm

    # Retry via a short formatting LLM call
    fix_user = (
        f"Rewrite the following into EXACTLY {n} words, no punctuation, plain lowercase words.\n"
        f"Output ONLY the {n} words, nothing else.\n\n"
        f"TEXT:\n{norm}"
    )
    for prov in get_provider_stack(settings.LLM_PREFERENCE):
        try:
            fixed = prov.generate("You strictly follow formatting.", fix_user)
            fixed_norm = _normalize_no_punct(fixed or "")
            if _count_words(fixed_norm) == n:
                return fixed_norm
        except Exception:
            continue

    # Final fallback: trim if too long; else return normalized
    parts = norm.split()
    if len(parts) > n:
        return " ".join(parts[:n])
    return norm


def run_pipeline(question: str, return_format: str | None = None):
    # 1) Retrieve
    persist = os.getenv("CHROMA_DIR", "./storage")
    top_k = int(os.getenv("TOP_K", "2"))
    retriever = get_retriever(persist_directory=persist, k=top_k)
    try:
        # Newer retrievers expose .invoke
        docs = retriever.invoke(question)
    except Exception:
        # Older retrievers expose .get_relevant_documents
        docs = retriever.get_relevant_documents(question) or []

    sources = _extract_sources(docs)
    context = "\n\n".join(getattr(d, "page_content", "") for d in docs).strip()

    # 2) Prompts — include QUESTION explicitly
    system_prompt = _load_system_prompt()
    rf = (return_format or settings.DEFAULT_RETURN_FORMAT).lower()
    user_prompt = (
        "Answer using ONLY the CONTEXT below. If context is insufficient, say so and do not fabricate sources.\n\n"
        f"QUESTION: {question}\n\n"
        f"RETURN_FORMAT: {rf.upper()}  # 'TEXT' for plain language, 'JSON' for a strict object\n\n"
        "JSON contract when RETURN_FORMAT == JSON:\n"
        '  { "answer": str, "sources": [str] }\n\n'
        f"CONTEXT:\n{context}\n\n"
        f"Sources you may cite (paths/ids): {sources}"
    )

    # 3) LLM with fallbacks
    raw = None
    for prov in get_provider_stack(settings.LLM_PREFERENCE):
        try:
            raw = prov.generate(system_prompt, user_prompt)
            if raw:
                break
        except Exception:
            continue
    if not raw:
        raw = json.dumps(
            {"answer": (context[:600] or "No context found."), "sources": sources}
        )

    # 4) Format handling
    if rf == "json":
        return _coerce_json(raw, context, sources)

    # TEXT mode
    raw_str = _strip_code_fences(raw)
    try:
        data = json.loads(raw_str)
        answer = (data.get("answer") or "").strip()
        if answer.startswith("Answer using ONLY the CONTEXT below") or (
            "RETURN_FORMAT:" in answer and "CONTEXT:" in answer
        ):
            answer = context[:600] or "No context found."
        answer = _maybe_force_exact_words(question, answer)
        return _render_text(answer, data.get("sources") or sources)
    except Exception:
        if raw_str.startswith("Answer using ONLY the CONTEXT below") or (
            "RETURN_FORMAT:" in raw_str and "CONTEXT:" in raw_str
        ):
            answer = context[:600] or "No context found."
        else:
            answer = raw_str
        answer = _maybe_force_exact_words(question, answer)
        return _render_text(answer, sources)
