import subprocess, json, sys, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src")

def run(cmd):
    return subprocess.check_output(cmd, shell=True, env=ENV, cwd=ROOT, text=True)

def test_ingest_and_qa():
    run("python -m rag_assistant.ingest")
    out = run('python -m rag_assistant.qa "What is this project?"')
    j = json.loads(out)
    assert "answer" in j and "sources" in j
    assert len(j["sources"]) >= 1
