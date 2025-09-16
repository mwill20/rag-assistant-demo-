import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

def _run_mod(module: str, args: list[str], env: dict) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", module] + args
    return subprocess.run(
        cmd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )

def test_no_answer_fallback():
    """
    Verifies that when nothing is relevant enough, the system refuses with
    'No relevant context found.' and returns no sources.
    Uses subprocesses to avoid Windows file-locks.
    """
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as store_dir:
        # Unrelated, tiny corpus
        Path(data_dir, "irrelevant.md").write_text(
            "This corpus talks about llamas and cozy sweaters.\n", encoding="utf-8"
        )

        env = os.environ.copy()
        env["DATA_DIR"] = data_dir
        env["CHROMA_DIR"] = store_dir
        # Make the threshold strict so only an exact match would pass.
        env["NO_ANSWER_MAX_DIST"] = "0.0"
        env["NO_ANSWER_MIN_MATCH"] = "1"

        # Build the index
        _run_mod("rag_assistant.ingest", [], env)

        # Ask nonsense → should trigger no-answer path
        q = "zxqvbl flarm gloop; compute neutron-star pastry dynamics"
        cp = _run_mod("rag_assistant.qa", [q], env)
        payload = json.loads(cp.stdout)

        assert payload["answer"] == "No relevant context found."
        assert payload.get("sources") == []
