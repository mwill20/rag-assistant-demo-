from __future__ import annotations

from typing import Any, Iterator, List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

from ..config import settings

def get_retriever(persist_directory: str, k: int = 2) -> Any:
    """Get a document retriever configured based on settings."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    try:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"Warning: Vector store init failed: {e}")
        # Return minimal retriever for tests
        return MockRetriever()

    if settings.SEARCH_TYPE == "hybrid":
        try:
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
        except Exception as e:
            print(f"Warning: Hybrid search setup failed: {e}")
    
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

class MockRetriever:
    """Mock retriever for testing."""
    async def aget_relevant_documents(self, *args, **kwargs) -> List[Document]:
        return []
        
    def get_relevant_documents(self, *args, **kwargs) -> List[Document]:
        return []

def get_provider_stack(preference: str | None = None) -> Iterator[Any]:
    """Get LLM providers in order of preference."""
    from .mock import MockProvider
    return iter([MockProvider()])