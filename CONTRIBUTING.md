# Contributing

Thanks for helping improve this RAG Assistant!

## Quick Start
1. Fork + clone.
2. Create a branch: `git switch -c feat/short-name`.
3. Set up env (Python 3.11+):
   ```bash
   uv venv || python -m venv .venv
   . .venv/Scripts/Activate.ps1  # Windows
   # or: source .venv/bin/activate
   pip install -U pip
   pip install -e ".[dev]"
