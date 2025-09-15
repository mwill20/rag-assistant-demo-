# src/rag_assistant/qa.py

import sys, json

# Ensure Windows consoles can emit UTF-8 (avoids cp1252 UnicodeEncodeError)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .pipeline import run_pipeline

def main():
    args = sys.argv[1:]
    use_text = any(a.strip() == "--text" for a in args)
    question = " ".join(a for a in args if not a.startswith("--")).strip() or "What is this project?"
    fmt = "text" if use_text else "json"
    result = run_pipeline(question, return_format=fmt)

    if fmt == "json":
        print(json.dumps(result, ensure_ascii=False))
    else:
        # result is a plain string from pipeline TEXT mode
        if isinstance(result, dict):
            # defensive: if provider returned JSON anyway
            ans = (result.get("answer") or "").strip()
            srcs = result.get("sources") or []
            print(ans + ("\n\nSources:\n" + "\n".join(f"- {s}" for s in srcs) if srcs else ""))
        else:
            print(result)

if __name__ == "__main__":
    main()
