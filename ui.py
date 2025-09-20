# ui.py — Streamlit UI (wide, scaled, humanized answer rendering)
from __future__ import annotations
import os, sys, json, time, pathlib, subprocess, re
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd
from pathlib import Path

DEFAULT_API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "./data")).resolve()

st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="wide")

# ---------- Sidebar (settings) ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
    provider = st.selectbox("LLM Provider", ["none", "openai", "groq"], index=0)
    api_key = st.text_input("API Key", type="password", placeholder="Paste provider key (kept in-session)")
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    ui_scale = st.slider(
        "UI Scale", min_value=100, max_value=190, value=160, step=5,
        help="Increase overall font & element sizes"
    )
    st.write(f"**Session:** {st.session_state.session_id or '—'}")
    show_raw = st.checkbox("Show raw response JSON (debug)", value=False)
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
                        [sys.executable, "-m", "rag_assistant.ingest"],  # venv’s Python
                        cwd=str(pathlib.Path(__file__).resolve().parent),
                        env={
                            **os.environ,
                            "DATA_DIR": str(DATA_DIR),
                            "CHROMA_DIR": os.getenv("CHROMA_DIR", "./storage"),
                        },
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

    .rag-card {{
        background: #111827;
        border: 1px solid #1f2937;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 0 16px rgba(0,0,0,0.25);
        /* readability tweaks */
        max-width: 980px;
        margin: 0 auto;
        line-height: 1.75;
        font-size: {int(base_px*1.02)}px;
    }}

    /* Paragraph + list spacing */
    .rag-card p {{ margin: 0 0 0.9rem 0; }}
    .rag-card ul {{ margin: 0.25rem 0 0.75rem 1.25rem; }}
    .rag-card li {{ margin: 0.15rem 0; }}

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

    /* Badges for page and score */
    .badge {{
        display: inline-block;
        padding: 0.08rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.85em;
        border: 1px solid #374151;
        background: #111827;
        margin-left: 0.35rem;
    }}
    .badge.score {{ background: #0b1220; }}
    .badge.page  {{ background: #1b2437; }}
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
    """If offline/extractive shows CONTEXT blocks, hide labels before answer."""
    if not text:
        return text
    t = text
    last_ctx = t.rfind("CONTEXT:")
    if last_ctx != -1:
        t = t[last_ctx + len("CONTEXT:") :]
    t = re.sub(r"^\s*(Answer:|ANSWER:)\s*", "", t)
    return t.strip()

_BULLET_CHARS = r"•·▪●○◦"
def bulletize(text: str) -> str:
    """Convert inline bullets into Markdown list items (incl. ' o ' pattern)."""
    if not text:
        return text
    t = text
    t = re.sub(fr"\s*[{_BULLET_CHARS}]\s*", "\n- ", t)
    t = re.sub(r"\s[oO]\s(?=[A-Za-z])", "\n- ", t)
    t = re.sub(r"[ \t]*\n[ \t]*-\s*", "\n- ", t)
    return t.strip()

def humanize_answer(text: str) -> str:
    t = strip_header_sections(text)
    t = reflow_pdf_text(t)
    t = bulletize(t)
    return t.strip()

def _fmt_score(x) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "0.000"

# ---------- API helpers ----------
def post_ask(api_base: str, question: str, session_id: Optional[str],
             provider: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/ask"
    payload: Dict[str, Any] = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    # per-request LLM selection
    provider = (provider or "").strip().lower()
    if provider and provider != "none" and api_key:
        payload["llm_provider"] = provider
        payload["api_key"] = api_key

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def build_sources_df(sources_scored: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s in sources_scored or []:
        name = s.get("name")
        label = s.get("label") or (Path(name).name if name else None)
        rows.append({
            "name": name,
            "label": label,
            "page": s.get("page"),
            "distance": s.get("distance"),
            "href": s.get("href"),
        })
    return pd.DataFrame(rows)

def render_answer(resp: Dict[str, Any]):
    st.markdown("### Answer")
    raw_answer = (resp.get("answer") or "").strip()
    st.markdown(f"<div class='rag-card'><p>{humanize_answer(raw_answer)}</p></div>", unsafe_allow_html=True)

    # --- Sources (legacy list: basenames only) ---
    sources = resp.get("sources") or []
    if sources:
        st.markdown("### Sources")
        st.markdown("<div class='rag-card'>", unsafe_allow_html=True)
        for s in sources:
            st.markdown(f"- {Path(str(s)).name}")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Evidence (clickable links + cosine distance) ---
    scored = resp.get("sources_scored") or []
    if scored:
        st.markdown("### Evidence (links + distance)")
        st.markdown("<div class='rag-card'>", unsafe_allow_html=True)
        for item in scored:
            name = item.get("label") or Path(str(item.get("name") or "")).name or "Source"
            href = item.get("href") or ""
            score = _fmt_score(item.get("distance"))
            page = item.get("page")
            page_badge = f"<span class='badge page'>page {int(page)}</span>" if isinstance(page, (int, float)) else ""
            score_badge = f"<span class='badge score'>dist {score}</span>"
            if href:
                st.markdown(
                    f"- <a href=\"{href}\">{name}</a> {page_badge} {score_badge}",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"- {name} {page_badge} {score_badge}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Main ----------
st.markdown("<h1>🔎 RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='small'>Ask a question about the indexed documents. "
    "Responses cite sources and show retrieval scores.</p>",
    unsafe_allow_html=True,
)

q = st.text_area("Your question", placeholder="e.g., Summarize this doc in 10 words.")
go = st.button("Ask")

if go and q.strip():
    try:
        res = post_ask(api_base, q.strip(), st.session_state.session_id, provider, api_key)
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    st.session_state.session_id = res.get("session_id", st.session_state.session_id)

    if show_raw:
        with st.expander("🔧 Debug JSON"):
            st.code(json.dumps(res, indent=2), language="json")

    render_answer(res)

    with st.expander("🔧 Debug: Top-k chunks & scores"):
        df = build_sources_df(res.get("sources_scored") or [])
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No scored sources returned.")

st.markdown(
    "<p class='small'>Cosine distance is shown (lower = closer). "
    "Powered by Chroma + SentenceTransformers + FastAPI + Streamlit.</p>",
    unsafe_allow_html=True,
)
