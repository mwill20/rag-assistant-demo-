import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

def _run_mod(module: str, args: list[str], env: dict) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", module] + args
    return subprocess.run(cmd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def test_golden_minicorpus():
    """
    Windows-friendly golden test:
    - Use separate processes for ingest and QA to avoid file locks.
    - Assert known phrase and citation of mini.md.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as data_dir, \
         tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as store_dir:

        # Create tiny corpus
        mini = Path(data_dir) / "mini.md"
        mini.write_text(
            "Project Overview:\n"
            "This repository contains a minimal Retrieval-Augmented Generation (RAG) assistant.\n"
            "It ingests Markdown, embeds chunks, stores them in Chroma, and answers with citations.\n",
            encoding="utf-8",
        )

        # Child env (do NOT mutate parent env)
        child_env = os.environ.copy()
        child_env["DATA_DIR"] = data_dir
        child_env["CHROMA_DIR"] = store_dir
        # <- allow answers (avoid accidental no-answer fallback)
        child_env["NO_ANSWER_MAX_DIST"] = "0.99"
        child_env["NO_ANSWER_MIN_MATCH"] = "1"

        # Ingest in a subprocess (releases file handles on exit)
        ing = _run_mod("rag_assistant.ingest", [], child_env)
        assert ing.returncode == 0, ing.stderr

        # Ask via CLI in a subprocess; parse JSON
        q = "What does this repository contain and how does it answer?"
        qa = _run_mod("rag_assistant.qa", [q], child_env)
        payload = json.loads(qa.stdout)

        answer = payload.get("answer", "")
        sources = payload.get("sources", [])

        assert "minimal Retrieval-Augmented Generation (RAG) assistant" in answer
        assert any(Path(s).name == "mini.md" for s in sources)
        assert answer.strip()
        assert len(sources) >= 1


