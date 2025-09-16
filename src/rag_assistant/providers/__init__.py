# src/rag_assistant/providers/__init__.py

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import settings


def _build_vector_retriever(persist_directory: str, k: int) -> Any:
    """
    Create a Chroma retriever with either similarity or MMR search,
    controlled by settings.USE_MMR. This path is deterministic and
    does not rely on any external network calls.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vectordb = Chroma(
        embedding_function=embeddings, persist_directory=persist_directory
    )

    search_kwargs = {"k": k}
    search_type = "mmr" if settings.USE_MMR else "similarity"

    return vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def get_retriever(persist_directory: str, k: int | None = None) -> Any:
    """
    Public factory used by the pipeline/tests.

    Note:
    - Previously this referenced a `SEARCH_TYPE` flag for hybrid retrieval.
      To keep CI deterministic and avoid extra dependencies, we default to
      vector-only retrieval here. If you want hybrid later, wire it up in a
      separate change (BM25 + EnsembleRetriever) and keep this interface stable.
    """
    top_k = int(k or settings.RETRIEVAL_K)
    return _build_vector_retriever(persist_directory=persist_directory, k=top_k)
