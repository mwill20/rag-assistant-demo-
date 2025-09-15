# src/rag_assistant/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from .config import settings  # kept light; no heavy imports at module import time

app = FastAPI(title="RAG Assistant API", version="0.1.0")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Lazy import so server starts instantly; heavy deps load only when /ask is called
    from .pipeline import run_pipeline
    result = run_pipeline(req.question, return_format="json")
    return AskResponse(**result)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
