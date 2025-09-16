from pathlib import Path

class MockProvider:
    """Mock LLM provider for testing."""
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Return a response that matches test expectations
        sample_source = str(Path("data/test.txt").resolve())
        return f'''{{
            "answer": "This is a test response that cites sources properly",
            "sources": ["{sample_source}"]
        }}'''