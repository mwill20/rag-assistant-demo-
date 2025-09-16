import os
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _build_vectorstore(persist_directory: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_MODEL)
    return Chroma(embedding_function=embeddings, persist_directory=persist_directory)


@lru_cache(maxsize=16)
def _bm25_from_storage(persist_directory: str, k: int) -> BM25Retriever:
    """
    Build an in-memory BM25 index from all docs persisted in Chroma storage.
    Cached so subsequent calls are instant.
    """
    vs = _build_vectorstore(persist_directory)
    # Pull all docs (safe for small local corpora). If your corpus grows large,
    # replace with a lightweight disk-backed text store.
    data = vs.get(limit=100000, include=["documents", "metadatas"])
    docs: list[Document] = []
    for text, meta in zip(
        data.get("documents", []), data.get("metadatas", []), strict=False
    ):
        docs.append(Document(page_content=text or "", metadata=meta or {}))
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = max(1, k)
    return retriever


def build_retriever(persist_directory: str, k: int = 2):
    """
    Central place to build a retriever. Toggle behavior via RETRIEVAL_MODE.
    Modes: knn (default), mmr, bm25
    """
    mode = os.getenv("RETRIEVAL_MODE", "knn").lower()

    if mode == "bm25":
        return _bm25_from_storage(persist_directory, k)

    vs = _build_vectorstore(persist_directory)

    if mode == "mmr":
        # Maximal Marginal Relevance for diversity
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(8, k * 4), "lambda_mult": 0.5},
        )

    # Default: vanilla vector similarity (kNN)
    return vs.as_retriever(search_kwargs={"k": k})


# Back-compat shim for existing imports
def get_retriever(*, persist_directory: str, k: int = 2):
    return build_retriever(persist_directory=persist_directory, k=k)
