import os, sys, time, subprocess, requests

def test_healthz():
    # spin server on a transient port
    port = 8001
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "rag_assistant.api:app", "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Poll /healthz up to ~6 seconds
        url = f"http://127.0.0.1:{port}/healthz"
        deadline = time.time() + 6.0
        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=1.0)
                if r.status_code == 200:
                    assert r.json().get("status") in {"ok", "ready", "healthy", None}
                    return
            except Exception as e:
                last_err = e
            time.sleep(0.25)
        # If we get here, it never came up—surface server logs to help debug
        out, err = proc.communicate(timeout=1) if proc.poll() is not None else proc.communicate(timeout=1)
        raise AssertionError(f"/healthz never responded. last_err={last_err}\n--- uvicorn stdout ---\n{out.decode(errors='ignore')}\n--- uvicorn stderr ---\n{err.decode(errors='ignore')}")
    finally:
        if proc.poll() is None:
            proc.terminate()
