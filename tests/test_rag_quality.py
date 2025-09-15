import json
from pathlib import Path
from fastapi.testclient import TestClient
from rag_assistant.api import app

KNOWN = {"project_overview.md", "usage_notes.md", "dummy.md"}

def _load_golden():
    items = []
    p = Path("tests/fixtures/golden.jsonl")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def test_answers_cite_known_sources():
    client = TestClient(app)
    golden = _load_golden()
    assert golden, "No golden questions loaded."

    for item in golden:
        r = client.post("/ask", json={"question": item["question"]})
        assert r.status_code == 200, f"POST /ask failed for: {item['question']}"
        payload = r.json()
        sources = payload.get("sources") or []
        assert sources, f"No sources returned for: {item['question']}"
        basenames = {Path(s).name for s in sources}
        unknown = basenames - KNOWN
        assert not unknown, f"Unknown source(s) cited: {unknown}"
