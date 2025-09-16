import subprocess
import sys
import time

import requests


def _stop(proc, grace=5):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except Exception:
            proc.kill()


def test_healthz():
    port = 8001
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "rag_assistant.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    # Start server
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    url = f"http://127.0.0.1:{port}/healthz"
    deadline = time.time() + 15.0  # give Windows a bit more time
    ok = False
    last_exc = None

    try:
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=1.5)
                if r.status_code == 200 and r.json().get("status") in {
                    "ok",
                    "ready",
                    "healthy",
                    None,
                }:
                    ok = True
                    break
            except Exception as e:
                last_exc = e
            time.sleep(0.5)
        if not ok:
            # Try to capture logs to help debugging
            _stop(proc, grace=3)
            logs = ""
            if proc.stdout:
                try:
                    logs = proc.stdout.read() or ""
                except Exception:
                    logs = ""
            raise AssertionError(
                f"Uvicorn failed to report healthy in time. Last error: {last_exc}\n--- logs ---\n{logs}"
            )
    finally:
        _stop(proc, grace=3)
