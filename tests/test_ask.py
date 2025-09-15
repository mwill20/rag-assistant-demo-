import json, sys, subprocess
def test_ask_cli():
    out = subprocess.check_output([sys.executable, "-m", "rag_assistant.qa", "What documents are in this corpus?"])
    data = json.loads(out.decode("utf-8"))
    assert "answer" in data and isinstance(data["sources"], list)
