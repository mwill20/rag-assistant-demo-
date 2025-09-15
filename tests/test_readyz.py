from fastapi.testclient import TestClient
from rag_assistant.api import app

def test_readyz_reports_chunks():
    client = TestClient(app)
    r = client.get("/readyz")
    assert r.status_code == 200
    data = r.json()
    assert "chunks" in data and isinstance(data["chunks"], int)
    assert data["chunks"] > 0, "Vector store is empty — run ingestion first."
