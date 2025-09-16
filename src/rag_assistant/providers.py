from __future__ import annotations

from typing import Any, Iterator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from .config import settings


def get_retriever(persist_directory: str, k: int = 2) -> Any:
    """Get a document retriever configured based on settings."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    if settings.SEARCH_TYPE == "hybrid":
        bm25_retriever = BM25Retriever.from_documents(
            vectordb.get(), k=k
        )
        vector_retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
    
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_provider_stack(preference: str | None = None) -> Iterator[Any]:
    """Get LLM providers in order of preference."""
    return iter([])  # Stub for now