# ui.py
# Streamlit UI for RAG Assistant Demo
# - Minimal, modern layout
# - Talks to FastAPI /ask so session memory + logging work
# - Shows sources with deep links + scores
# - Optional: upload docs -> saves to ./data and re-runs ingest

from __future__ import annotations

import os
import json
import time
import pathlib
import subprocess
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd


# -------------------- Config --------------------
DEFAULT_API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "./data")).resolve()


# -------------------- UI Styling --------------------
st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #e5e7eb; }
    .block-container { padding-top: 2rem; }
    .rag-card {
        background: #111827; border: 1px solid #1f2937;
        padding: 1rem 1.25rem; border-radius: 14px; box-shadow: 0 0 16px rgba(0,0,0,0.2);
    }
    .small { font-size: 0.9rem; color: #9ca3af; }
    a, a:visited { color: #60a5fa; text-decoration: none; }
    a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- Helpers --------------------
def post_ask(api_base: str, question: str, session_id: Optional[str]) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/ask"
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def run_ingest() -> str:
    """Run `python -m rag_assistant.ingest` and return a short status string."""
    try:
        start = time.time()
        proc = subprocess.run(
            ["python", "-m", "rag_assistant.ingest"],
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.time() - start
        return f"Ingest complete in {elapsed:.2f}s.\n\n{proc.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Ingest failed:\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"


def save_uploaded_files(files: List[Any], dst_dir: pathlib.Path) -> List[pathlib.Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files or []:
        # Streamlit returns an UploadedFile; write to disk
        out = dst_dir / f.name
        with open(out, "wb") as w:
            w.write(f.read())
        saved.append(out)
    return saved


def build_sources_table(sources_scored: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in sources_scored or []:
        rows.append(
            {
                "path": item.get("path"),
                "page": item.get("page"),
                "metric": item.get("metric"),
                "score_type": item.get("score_type"),
                "distance": item.get("score"),
                "href": item.get("href"),
            }
        )
    return pd.DataFrame(rows)


# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    cols = st.columns(2)
    with cols[0]:
        if st.button("➕ New Session"):
            st.session_state.session_id = None
    with cols[1]:
        st.write(f"**Session:** {st.session_state.session_id or '—'}")

    st.markdown("---")
    st.markdown("### 📚 Add Documents (optional)")
    uploads = st.file_uploader(
        "Drop PDFs / MD / TXT to include in the knowledge base",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Add to Knowledge Base"):
        saved = save_uploaded_files(uploads, DATA_DIR)
        if saved:
            with st.spinner("Rebuilding vector index…"):
                msg = run_ingest()
            st.success(f"Saved {len(saved)} file(s).\n\n" + msg)
        else:
            st.info("No files selected.")

    st.markdown("---")
    st.markdown(
        """
        **Hints**
        - The API returns `sources_scored` with cosine **distance** (lower = more similar).
        - Deep links use `file://` URIs and `#page=N` for PDFs when available.
        - Memory persists per session (in-memory only; resets on server restart).
        """
    )


# -------------------- Main --------------------
st.markdown("<h1>🔎 RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='small'>Ask a question about the indexed documents. "
    "Responses cite sources and show retrieval scores.</p>",
    unsafe_allow_html=True,
)

with st.container():
    q = st.text_area("Your question", height=120, placeholder="e.g., What is this project about?")
    go = st.button("Ask", type="primary")

    if go and q.strip():
        try:
            res = post_ask(api_base, q.strip(), st.session_state.session_id)
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

        # Persist / get session_id
        st.session_state.session_id = res.get("session_id", st.session_state.session_id)

        # Answer card
        st.markdown("<div class='rag-card'>", unsafe_allow_html=True)
        st.markdown("#### Answer")
        st.markdown(res.get("answer", "").strip())

        # Sources
        st.markdown("#### Sources")
        sources = res.get("sources_scored") or []
        if sources:
            for s in sources:
                path = s.get("path")
                href = s.get("href")
                dist = s.get("score")
                page = s.get("page")
                label = f"{path}" + (f" (page {page})" if page else "")
                if href:
                    st.markdown(f"- [{label}]({href})  — distance: `{dist:.3f}`")
                else:
                    st.markdown(f"- {label}  — distance: `{dist:.3f}`")
        else:
            # Fallback to legacy sources
            for p in res.get("sources", []):
                st.markdown(f"- {p}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Debug panel
        with st.expander("🔧 Debug: Top-k chunks & scores"):
            df = build_sources_table(res.get("sources_scored") or [])
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            st.code(json.dumps(res, indent=2), language="json")


# -------------------- Footer --------------------
st.markdown(
    "<p class='small'>Cosine distance is shown (lower = closer). "
    "Powered by Chroma + SentenceTransformers + FastAPI + Streamlit.</p>",
    unsafe_allow_html=True,
)
