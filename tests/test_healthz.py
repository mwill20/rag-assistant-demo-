import subprocess, sys, json, os, time
import requests
from threading import Thread

def test_healthz():
    # spin server
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "rag_assistant.api:app", "--port", "8001"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.2)
    try:
        r = requests.get("http://127.0.0.1:8001/healthz", timeout=3)
        assert r.json() == {"status": "ok"}
    finally:
        proc.terminate()
