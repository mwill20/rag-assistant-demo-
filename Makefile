# Makefile — Python-first; works on Windows/macOS/Linux
PY ?= python
UVICORN ?= uvicorn

.PHONY: setup seed ingest qa ask api test mmr

setup:
	$(PY) -m pip install -U pip setuptools wheel
	@if [ -f requirements.txt ]; then \
		pip install -r requirements.txt -e .; \
	else \
		pip install -e .; \
	fi

seed:
	$(PY) -c "from pathlib import Path;p=Path('data');p.mkdir(parents=True,exist_ok=True);(p/'dummy.md').write_text('RAG Assistant demo corpus. Replace with your real docs.\n',encoding='utf-8');print('Seeded data/dummy.md')"

ingest:
	$(PY) -m rag_assistant.ingest

qa:
	$(PY) -m rag_assistant.qa "What is this project?"

# Usage: make ask Q="What documents are in this corpus?"
ask:
	$(PY) -c "import os,sys,subprocess;q=os.environ.get('Q') or 'What is this project?'; sys.exit(subprocess.call([sys.executable,'-m','rag_assistant.qa',q]))"

api:
	$(PY) -m $(UVICORN) rag_assistant.api:app --reload

test:
	pytest -q

# Re-ingest using MMR retrieval (cross-platform; sets env inside Python)
mmr:
	$(PY) -c "import os,sys,subprocess; os.environ['RETRIEVAL_MODE']='mmr'; sys.exit(subprocess.call([sys.executable,'-m','rag_assistant.ingest']))"

reindex:
    python -c "import shutil,os; d='storage'; shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)"
    python -m rag_assistant.ingest

