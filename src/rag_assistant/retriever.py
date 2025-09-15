# src/rag_assistant/retriever.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import settings

def get_retriever():
    """
    Build a retriever backed by Chroma with k-NN (default) or MMR,
    driven by env: RETRIEVAL_MODE and K in config.settings.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vs = Chroma(embedding_function=embeddings, persist_directory=str(settings.CHROMA_DIR))

    if settings.RETRIEVAL_MODE == "mmr":
        return vs.as_retriever(search_type="mmr", search_kwargs={"k": settings.K})
    return vs.as_retriever(search_kwargs={"k": settings.K})
