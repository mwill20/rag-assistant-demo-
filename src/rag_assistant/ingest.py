# src/rag_assistant/ingest.py

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import settings


def _load_documents(data_dir: Path) -> List:
    """Load MD/TXT/PDF from data_dir with encoding autodetect for text files."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[ingest] Data dir does not exist: {data_dir}")
        return []

    loaders = [
        DirectoryLoader(
            str(data_dir),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        ),
        DirectoryLoader(
            str(data_dir),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        ),
        DirectoryLoader(
            str(data_dir),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        ),
    ]

    docs: List = []
    for ld in loaders:
        try:
            docs.extend(ld.load())
        except Exception as e:
            print(f"[ingest] Loader error ({ld}): {e}")

    # Strip stray UTF-8 BOM chars to avoid Windows console/test failures.
    for d in docs:
        if getattr(d, "page_content", None):
            d.page_content = d.page_content.replace("\ufeff", "")

    return docs


def main():
    print(f"[ingest] DATA_DIR={settings.DATA_DIR}")
    print(f"[ingest] CHROMA_DIR={settings.CHROMA_DIR}")

    docs = _load_documents(Path(settings.DATA_DIR))

    if not docs:
        print("[ingest] No documents found. Put files in ./data and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vs = Chroma(embedding_function=embeddings, persist_directory=str(settings.CHROMA_DIR))

    # Add & persist
    if chunks:
        vs.add_documents(chunks)
        try:
            vs.persist()
        except Exception:
            pass
        try:
            print(f"[ingest] Collection size now: {vs._collection.count()}")
        except Exception:
            pass
        print(f"[ingest] Ingested {len(chunks)} chunks into {settings.CHROMA_DIR}")
    else:
        print("[ingest] No chunks created; nothing ingested.")


if __name__ == "__main__":
    main()
