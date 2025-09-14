# src/rag_assistant/qa.py

import sys
import json
from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import settings

# Ensure Windows consoles can emit UTF-8 (avoids cp1252 UnicodeEncodeError)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _strip_bom(s: str) -> str:
    # Remove any UTF-8 BOM characters that can sneak into docs
    return s.replace("\ufeff", "") if s else s


def run_qa(question: str) -> Tuple[str, List[str]]:
    """Return (answer_text, sources_list) using extractive fallback over top-k chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vs = Chroma(embedding_function=embeddings, persist_directory=settings.CHROMA_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(question) or []

    cleaned = [_strip_bom(d.page_content or "").strip() for d in docs]
    cleaned = [c for c in cleaned if c]

    stitched = "\n---\n".join(cleaned)
    sources = [d.metadata.get("source", "?") for d in docs if d is not None]

    if not stitched:
        stitched = (
            "No relevant content retrieved. Ensure documents exist in DATA_DIR and run "
            "`make ingest` before asking questions."
        )

    return stitched, sources


def main():
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "What is this project?"
    answer, sources = run_qa(query)
    result = {"answer": answer, "sources": sources}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
