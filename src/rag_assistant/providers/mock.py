class MockProvider:
    """Mock LLM provider for testing."""
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return '{"answer": "This is a test response", "sources": []}'