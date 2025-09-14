.PHONY: ingest qa api test

ingest:
	python -m rag_assistant.ingest

qa:
	python -m rag_assistant.qa "What documents are in this corpus?"

api:
	uvicorn rag_assistant.api:app --reload

test:
	pytest -q
