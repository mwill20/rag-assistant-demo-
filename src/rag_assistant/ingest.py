from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import settings
from .utils import ensure_dirs


def load_docs(data_dir: str) -> List:
    """Load PDFs, .md, and .txt from DATA_DIR (top-level)."""
    docs = []
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ingest] DATA_DIR not found: {data_dir}")
        return docs

    for p in data_path.glob("*"):
        # Skip a readme in the data folder; it's usually instructions, not content.
        if p.name.lower() == "readme.md":
            continue
        if p.is_dir():
            continue

        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in {".md", ".txt"}:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())

    return docs


def main() -> None:
    # Make sure the storage directory exists
    ensure_dirs(settings.CHROMA_DIR)

    print(f"[ingest] Loading documents from: {settings.DATA_DIR}")
    docs = load_docs(settings.DATA_DIR)
    if not docs:
        print("[ingest] No documents found. Put files in the data/ folder and retry.")
        return

    # Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Created {len(chunks)} chunks")

    # Build embeddings (HuggingFace sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # Create/persist Chroma DB. With langchain-chroma, persistence is automatic
    # when persist_directory is provided.
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DIR,
    )

    print(f"Ingested {len(chunks)} chunks into {settings.CHROMA_DIR}")


if __name__ == "__main__":
    main()

