# Makefile — Cross-platform (Windows/macOS/Linux) and tab-safe
# Paste this whole file at repo root.

.RECIPEPREFIX := >
PY ?= python
UVICORN ?= uvicorn

.PHONY: help setup seed ingest qa ask api test mmr reindex clean

help:
> @echo Targets:
> @echo   setup   - install/update deps (editable with dev extras)
> @echo   seed    - create data/dummy.md
> @echo   ingest  - build/persist vector store
> @echo   qa      - quick CLI query
> @echo   ask     - CLI query; pass Q="your question"
> @echo   api     - start FastAPI on localhost:8000
> @echo   test    - run pytest
> @echo   mmr     - re-ingest with RETRIEVAL_MODE=mmr (if supported)
> @echo   reindex - wipe and recreate storage/
> @echo   clean   - remove caches, build artifacts

setup:
> $(PY) -c "import os,sys,subprocess; \
> subprocess.check_call([sys.executable,'-m','pip','install','-U','pip','setuptools','wheel']); \
> subprocess.check_call([sys.executable,'-m','pip','install','-e','.[dev]']); \
> os.path.exists('requirements.txt') and subprocess.check_call([sys.executable,'-m','pip','install','-r','requirements.txt'])"

seed:
> $(PY) -c "from pathlib import Path; p=Path('data'); p.mkdir(parents=True,exist_ok=True); \
> (p/'dummy.md').write_text('RAG Assistant demo corpus. Replace with your real docs.\\n',encoding='utf-8'); \
> print('Seeded data/dummy.md')"

ingest:
> $(PY) -m rag_assistant.ingest

qa:
> $(PY) -m rag_assistant.qa "What is this project?"

# Usage: make ask Q="What documents are in this corpus?"
ask:
> $(PY) -c "import os,sys,subprocess; q=os.environ.get('Q') or 'What is this project?'; \
> sys.exit(subprocess.call([sys.executable,'-m','rag_assistant.qa',q]))"

api:
> $(PY) -m $(UVICORN) rag_assistant.api:app --reload

test:
> pytest -q

# Re-ingest using MMR retrieval (if your code reads RETRIEVAL_MODE)
mmr:
> $(PY) -c "import os,sys,subprocess; os.environ['RETRIEVAL_MODE']='mmr'; \
> sys.exit(subprocess.call([sys.executable,'-m','rag_assistant.ingest']))"

reindex:
> $(PY) -c "import shutil,os; shutil.rmtree('storage',ignore_errors=True); os.makedirs('storage',exist_ok=True)"

clean:
> $(PY) -c "import shutil,glob; \
> [shutil.rmtree(p,ignore_errors=True) for p in ['storage','.pytest_cache','build','dist']]; \
> [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('src/*.egg-info')]"

