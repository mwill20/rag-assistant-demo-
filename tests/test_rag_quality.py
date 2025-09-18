# tests/test_rag_quality.py
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from rag_assistant.api import app

# Golden set bundled with the repo
KNOWN = {"project_overview.md", "usage_notes.md", "dummy.md"}

# Optional allow-list for user-added docs (comma-separated env or extend the set below)
ALLOWED_EXTRA = {
    s.strip() for s in os.getenv("TEST_ALLOWED_EXTRA_SOURCES", "").split(",") if s.strip()
} | {
    # add any local/demo files you might ingest during development here:
    "Week 1 Assignment.pdf",
    "gemini-for-google-workspace-prompting-guide-101.pdf",
}


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

        # Require that at least one golden source is cited
        assert basenames & KNOWN, (
            f"No golden sources cited for: {item['question']} (got: {sorted(basenames)})"
        )

        # Disallow unknowns unless they are explicitly allowed
        unknown = {b for b in basenames if b not in KNOWN and b not in ALLOWED_EXTRA}
        assert not unknown, f"Unknown source(s) cited: {sorted(unknown)}"
