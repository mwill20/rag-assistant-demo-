# ui.py — Streamlit UI (wide, scaled, humanized answer rendering)
from __future__ import annotations
import os, json, time, pathlib, subprocess, re
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd

DEFAULT_API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "./data")).resolve()

st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="wide")

# ---------- Sidebar (settings) ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    ui_scale = st.slider(
        "UI Scale", min_value=100, max_value=190, value=160, step=5,
        help="Increase overall font & element sizes"
    )
    st.write(f"**Session:** {st.session_state.session_id or '—'}")
    show_raw = st.checkbox("Show raw context + memory (debug)", value=False)
    st.markdown("---")

    st.markdown("### 📚 Add Documents (optional)")
    uploads = st.file_uploader(
        "Drop PDFs / MD / TXT", type=["pdf", "md", "txt"], accept_multiple_files=True
    )
    if st.button("Add to Knowledge Base"):
        saved: list[pathlib.Path] = []
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads or []:
            out = DATA_DIR / f.name
            with open(out, "wb") as w:
                w.write(f.read())
            saved.append(out)

        if saved:
            with st.spinner("Rebuilding vector index…"):
                try:
                    start = time.time()
                    proc = subprocess.run(
                        ["python", "-m", "rag_assistant.ingest"],
                        capture_output=True, text=True, check=True
                    )
                    st.success(f"Ingest complete in {time.time()-start:.2f}s.\n\n{proc.stdout}")
                except subprocess.CalledProcessError as e:
                    st.error(f"Ingest failed:\n\n{e.stdout}\n\n{e.stderr}")
        else:
            st.info("No files selected.")

    st.markdown("---")
    st.markdown(
        """
        **Hints**
        - API returns `sources_scored` with cosine **distance** (lower = closer).
        - Deep links use `/static` HTTP URLs (and `#page=N` for PDFs when available).
        - Session memory is in-memory only (resets on server restart).
        """
    )

# ---------- Dynamic CSS (scaled) ----------
base_px = int(16 * ui_scale / 100)
small_px = max(13, int(13 * ui_scale / 100))
button_px = int(16 * ui_scale / 100)
textarea_min_h = int(260 * ui_scale / 120)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0f172a; color: #e5e7eb; font-size: {base_px}px; }}
    .block-container {{ padding-top: 2rem; }}
    h1, .stMarkdown h1 {{ font-size: {int(base_px*2.0)}px; }}
    h2, .stMarkdown h2 {{ font-size: {int(base_px*1.6)}px; }}
    h3, .stMarkdown h3 {{ font-size: {int(base_px*1.35)}px; }}
    .small {{ font-size: {small_px}px; color: #9ca3af; }}

    .rag-card {{
        background: #111827;
        border: 1px solid #1f2937;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 0 16px rgba(0,0,0,0.25);
        line-height: 1.65;
        font-size: {int(base_px*1.02)}px;
    }}

    textarea {{ min-height: {textarea_min_h}px !important; font-size: {int(base_px*1.0)}px !important; }}

    .stButton>button {{
        padding: 0.65rem 1.2rem;
        font-size: {button_px}px;
        border-radius: 12px;
        background: #ef4444;
        color: #fff;
        border: 1px solid #b91c1c;
    }}
    .stButton>button:hover {{ filter: brightness(1.05); }}

    a, a:visited {{ color: #60a5fa; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Text formatting helpers ----------
def reflow_pdf_text(text: str) -> str:
    """Unwrap column newlines, de-hyphenate, preserve paragraph breaks."""
    if not text:
        return text
    t = text.replace("\r\n", "\n")
    t = re.sub(r"-\n([a-z])", r"\1", t, flags=re.IGNORECASE)  # de-hyphenate
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)                   # single newline -> space
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def strip_header_sections(text: str) -> str:
    """
    In offline/extractive mode the answer may start with
    'SESSION MEMORY: ... CONTEXT: ...'. Show only the content after CONTEXT:.
    """
    if not text:
        return text
    t = text
    # Prefer the last CONTEXT:, in case it appears earlier in memory text.
    last_ctx = t.rfind("CONTEXT:")
    if last_ctx != -1:
        t = t[last_ctx + len("CONTEXT:") :]
    # Remove any leading label residue
    t = re.sub(r"^\s*(Answer:|ANSWER:)\s*", "", t)
    return t.strip()

_BULLET_CHARS = r"•·▪●○◦"
def bulletize(text: str) -> str:
    """
    Convert inline bullets into Markdown list items.
    Also handles ' o ' style bullets (common in PDF extracts).
    """
    if not text:
        return text
    t = text

    # Convert explicit bullet glyphs into list items
    t = re.sub(fr"\s*[{_BULLET_CHARS}]\s*", "\n- ", t)

    # Convert ' o ' separators to bullets when between words
    t = re.sub(r"\s[oO]\s(?=[A-Za-z])", "\n- ", t)

    # Normalize: ensure '- ' starts at line-begin
    t = re.sub(r"[ \t]*\n[ \t]*-\s*", "\n- ", t)

    # If we now have a dense line with many bullets, add paragraph breaks before headers-like chunks.
    return t.strip()

def humanize_answer(text: str) -> str:
    t = strip_header_sections(text)
    t = reflow_pdf_text(t)
    t = bulletize(t)
    return t.strip()

# ---------- API helpers ----------
def post_ask(api_base: str, question: str, session_id: Optional[str]) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/ask"
    payload: Dict[str, Any] = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def build_sources_df(sources_scored: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s in sources_scored or []:
        rows.append({
            "path": s.get("path"),
            "page": s.get("page"),
            "metric": s.get("metric"),
            "score_type": s.get("score_type"),
            "distance": s.get("score"),
            "href": s.get("href"),
        })
    return pd.DataFrame(rows)

# ---------- Main ----------
st.markdown("<h1>🔎 RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='small'>Ask a question about the indexed documents. "
    "Responses cite sources and show retrieval scores.</p>",
    unsafe_allow_html=True,
)

q = st.text_area("Your question", placeholder="e.g., What is this project all about?")
go = st.button("Ask")

if go and q.strip():
    try:
        res = post_ask(api_base, q.strip(), st.session_state.session_id)
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    st.session_state.session_id = res.get("session_id", st.session_state.session_id)

    st.markdown("<div class='rag-card'>", unsafe_allow_html=True)
    st.markdown("### Answer")

    raw_answer = (res.get("answer") or "").strip()
    if show_raw:
        st.markdown("##### Raw context + memory")
        st.code(raw_answer)
        st.markdown("##### Rendered")
    st.markdown(humanize_answer(raw_answer))

    st.markdown("### Sources")
    sources = res.get("sources_scored") or []
    if sources:
        for s in sources:
            label = f"{s.get('path')}" + (f" (page {s.get('page')})" if s.get("page") else "")
            dist = s.get("distance")
            href = s.get("href")
            suffix = f" — distance: `{float(dist):.3f}`" if isinstance(dist, (int, float)) else ""
            if href:
                st.markdown(f"- [{label}]({href}){suffix}")
            else:
                st.markdown(f"- {label}{suffix}")

    else:
        for p in res.get("sources", []):
            st.markdown(f"- {p}")

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔧 Debug: Top-k chunks & scores"):
        df = build_sources_df(res.get("sources_scored") or [])
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.code(json.dumps(res, indent=2), language="json")

st.markdown(
    "<p class='small'>Cosine distance is shown (lower = closer). "
    "Powered by Chroma + SentenceTransformers + FastAPI + Streamlit.</p>",
    unsafe_allow_html=True,
)
