.PHONY: setup ingest qa ask api test

setup:
	python -m pip install -U pip
	pip install -r requirements.txt -e .

ingest:
	python -m rag_assistant.ingest

qa:
	python -m rag_assistant.qa "What documents are in this corpus?"

ask:
	python -m rag_assistant.qa "$(Q)"

api:
	uvicorn rag_assistant.api:app --reload

test:
	pytest -q

