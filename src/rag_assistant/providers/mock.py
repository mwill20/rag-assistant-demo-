import json


class MockProvider:
    """Mock LLM provider for testing."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = {
            "answer": "This is a RAG assistant demo project that demonstrates question answering.",
            "sources": ["data/project_overview.md"],
        }
        return json.dumps(response)
