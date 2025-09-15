# RAG Assistant — Ready Tensor AAIDC Project 1
[![CI](https://github.com/mwill20/rag-assistant-demo-/actions/workflows/ci.yml/badge.svg)](https://github.com/mwill20/rag-assistant-demo-/actions/workflows/ci.yml)


Minimal, reproducible Retrieval-Augmented Generation (RAG) assistant built with LangChain + Chroma, offering both a CLI and a FastAPI endpoint. Designed to meet Ready Tensor Project 1 requirements and the Elite repository standards: clear docs, clean structure, pinned environments (at release), tests, and legal hygiene.

---

## Table of Contents
- [Overview](#overview)  
- [Target Audience](#target-audience)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Environment Setup](#environment-setup)  
- [Usage](#usage)  
- [Data Requirements](#data-requirements)  
- [Testing](#testing)  
- [Configuration](#configuration)  
- [Methodology](#methodology)  
- [Performance](#performance)  
- [Project Structure](#project-structure)  
- [Troubleshooting](#troubleshooting)  
- [License](#license)  
- [Contributing](#contributing)  
- [Changelog](#changelog)  
- [Citation](#citation)  
- [Contact](#contact)  
- [Quality Checklist](#quality-checklist)  

---

## Overview
This repo demonstrates a compact RAG pipeline: documents are ingested → chunked → embedded → stored in Chroma → retrieved by similarity → answered with citations. It includes:

- A CLI for quick Q&A.  
- A FastAPI service with `/healthz` and `/ask`.  
- A deterministic mini-corpus to keep tests and demos reproducible until you swap in your real corpus.  

Meets AAIDC Project 1 requirements (vector DB, embeddings, retrieval, LangChain) and Ready Tensor publication expectations.

---

## Target Audience
- Learners completing Ready Tensor AAIDC Project 1.  
- Reviewers/graders verifying functionality, reproducibility, and code quality.  
- Practitioners seeking a minimal, production-lean RAG scaffold.  

---

## Prerequisites
- **OS:** Windows 10/11, macOS, or Linux  
- **Python:** 3.11+  
- **VS Code:** recommended (with Python extension)  
- **No API keys required** (extractive fallback works). Optional OpenAI/Groq keys supported.  

---

## Installation

### Windows PowerShell
```powershell
git clone <your-repo>
cd rag-assistant
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt -e .

### MacOS/Linux
git clone <your-repo>
cd rag-assistant
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt -e .

## Environment Setup
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./storage
DATA_DIR=./data
LLM_PROVIDER=none   # none | openai | groq
OPENAI_API_KEY=
GROQ_API_KEY=

## Usage
mkdir data 2>$null
"Ready Tensor teaches RAG. Phase 5 polishes tests." | Out-File -FilePath data/dummy.md -Encoding utf8

### Run pipeline:
make ingest
make qa
make api   # starts http://127.0.0.1:8000

## API examples (PowerShell)
Invoke-RestMethod http://127.0.0.1:8000/healthz | ConvertTo-Json

$r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" `
  -Method POST -ContentType "application/json" `
  -Body '{ "question": "What is this project?" }'
$r | ConvertTo-Json -Depth 5

## API sample outputs
**Health check**
```json
{ "status": "ok" }

## /ask (truncated)
{
  "answer": "Ready Tensor teaches RAG. Phase 5 polishes tests.\n---\nReady Tensor teaches RAG. Phase 5 polishes tests.",
  "sources": ["data/dummy.md"]
}

## Testing
make test

Includes:
- test_ingest.py → Chroma persistence exists after ingest.
- test_api.py → /healthz returns {"status":"ok"}.
- test_smoke.py → end-to-end sanity.

## Configuration
| Variable         | Description                   | Default                                |
| ---------------- | ----------------------------- | -------------------------------------- |
| DATA\_DIR        | Path to source documents      | ./data                                 |
| CHROMA\_DIR      | Chroma persistence directory  | ./storage                              |
| EMBEDDING\_MODEL | SentenceTransformers model id | sentence-transformers/all-MiniLM-L6-v2 |
| LLM\_PROVIDER    | none \| openai \| groq        | none                                   |
| OPENAI\_API\_KEY | OpenAI key (if used)          | empty                                  |
| GROQ\_API\_KEY   | Groq key (if used)            | empty                                  |

## Methodology
- Loaders: LangChain loaders for MD/TXT/PDF.
- Chunking: RecursiveCharacterTextSplitter (~1k chars, 200 overlap).
- Embeddings: SentenceTransformers → vectors in Chroma.
- Retrieval: k-NN similarity search.

    Answering:
    - Default (no keys): stitched extractive answer from top-k chunks with citations.
    - With keys: wrap retriever in LangChain RetrievalQA and prompt the model to cite sources.

## Performance
Scope is educational and minimal; performance depends on corpus size and retriever k. For larger corpora, consider:
- Reranking (e.g., CrossEncoder)
- Better chunking strategies
- Persisted embeddings cache per commit

## Project Structure
rag-assistant/
├─ src/rag_assistant/          # package
│  ├─ api.py                   # FastAPI endpoints
│  ├─ ingest.py                # data → chunks → embeddings → Chroma
│  ├─ qa.py                    # retrieval QA (CLI)
│  ├─ config.py                # env settings
│  └─ utils.py
├─ tests/                      # pytest suite
├─ data/                       # docs (not committed, except dummy.md for demo)
├─ storage/                    # Chroma DB (generated, gitignored)
├─ .env_example
├─ requirements.txt
├─ Makefile
├─ README.md
└─ LICENSE

## Troubleshooting
- ModuleNotFoundError: rag_assistant → ensure pip install -e . ran.
- API empty answers → run make ingest first; confirm files exist in data/.
- Windows encoding quirks → use Invoke-RestMethod and ConvertTo-Json.
- Pytest flakes from pyproject.toml encoding → ensure UTF-8 without BOM.

### Windows/UTF-8 note
If you hit `UnicodeEncodeError` in the CLI, your console code page is likely cp1252. This repo mitigates it by:
- Loading text with encoding autodetect and stripping BOMs during ingest.
- Reconfiguring CLI stdout to UTF-8.
To re-seed the demo doc *without a BOM*:
```powershell
[IO.File]::WriteAllText("data\dummy.md",
  "Ready Tensor teaches RAG. Phase 5 polishes tests.",
  (New-Object System.Text.UTF8Encoding($false))
)


## License
Default recommendation: MIT License. Add a LICENSE file at the repo root.
If your data/model licenses impose constraints, reflect that in LICENSE and this README.

## Contributing
External contributions are welcome once the project is published.
See CONTRIBUTING.md and CODE_OF_CONDUCT.md (to be added) for guidelines, environment setup, and commit style.

## Changelog
See CHANGELOG.md (to be added).
We follow Semantic Versioning (MAJOR.MINOR.PATCH).

## Citation
If you use this project in academic or professional work, please cite it. Add a CITATION.cff file or cite as:
@software{rag_assistant,
  title        = {RAG Assistant (AAIDC Project 1)},
  author       = {Your Name},
  year         = {2025},
  url          = {https://github.com/<you>/rag-assistant}
}

## Contact
Maintainer: 
Issues → GitHub Issues tab

## Quality Checklist
Documentation
- README covers Overview → Install → Usage → Data → Tests → Config → Methodology → License.
- Example CLI and API outputs included (paste snippets before submission).

## Repository Structure
- src/ package layout, clear entry points (ingest.py, qa.py, api.py).
- .gitignore excludes .venv/, storage/, .env, __pycache__/.

## Environment & Dependencies
- requirements.txt installs cleanly in fresh venv.
- For release tags, pin versions and attach a lockfile (optional: pip-tools/uv).

## License & Legal
- LICENSE present and compatible with dependencies.
- Data/model usage rights documented (if distributing datasets/models).

## Code Quality
- pytest green locally; minimal, focused tests.
- Type hints and docstrings for public functions (as time permits).
- Optional: add pre-commit with black, ruff, mypy, pytest hooks.

### Retrieval modes
Choose with `RETRIEVAL_MODE`:

- `knn` (default): vector similarity over MiniLM embeddings
- `mmr`: diversity-aware retrieval (MMR) on the same vectors
- `bm25`: token-based lexical retrieval (BM25) built in-memory from persisted chunks

Change mode:
  Windows (PowerShell):  $env:RETRIEVAL_MODE='bm25'
  macOS/Linux (bash):    export RETRIEVAL_MODE=bm25
