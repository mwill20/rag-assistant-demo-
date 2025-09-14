from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import settings

app = FastAPI(title="RAG Assistant API", version="0.1.0")

class AskRequest(BaseModel):
    """Incoming request schema for /ask."""
    question: str
    k: int = 4
    max_chars: int = 1500

class AskResponse(BaseModel):
    """Outgoing response schema for /ask."""
    answer: str
    sources: list[str]

def _retriever():
    """Build a retriever backed by Chroma (persisted in ./storage)."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    return Chroma(embedding_function=embeddings, persist_directory=settings.CHROMA_DIR)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Take a question, fetch top-k chunks, stitch a concise extractive answer,
    and return answer + unique sources.
    """
    vs = _retriever()
    docs = vs.similarity_search(req.question, k=req.k)

    if not docs:
        return AskResponse(answer="No results found. Did you run ingestion?", sources=[])

    stitched = "\n---\n".join(d.page_content for d in docs)
    answer = stitched[:req.max_chars]
    sources = sorted({d.metadata.get("source", "") for d in docs if d.metadata.get("source")})
    return AskResponse(answer=answer, sources=sources)

@app.get("/healthz")
def healthz():
    """Lightweight health check for uptime probes and quick diagnostics."""
    return {"status": "ok"}
