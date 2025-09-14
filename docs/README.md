# 1) Create & activate venv
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Install deps
python -m pip install -U pip
pip install -r requirements.txt

# 3) Tell Python where your package lives (temp for this terminal)
$env:PYTHONPATH = (Resolve-Path .\src).Path

# 4) Add your docs to ./data/
#    (Markdown, TXT, PDFs are supported)

# 5) Build vectors (Chroma DB)
make ingest

# 6) Ask from the CLI
make qa
make ask Q="What is this project?"

# 7) Run the API
make api
