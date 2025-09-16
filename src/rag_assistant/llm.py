import json

from .config import LLM_PROVIDER, OPENAI_MODEL


class NullLLM:
    """Fallback: returns a succinct stitched extract as 'answer'."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Expect user_prompt to include a JSON schema instruction; we comply minimally.
        try:
            payload = {"answer": user_prompt[:500], "sources": []}
            return json.dumps(payload)
        except Exception:
            return '{"answer":"No answer available","sources":[]}'


_llm_singleton = None


def get_llm():
    global _llm_singleton
    if _llm_singleton:
        return _llm_singleton
    if LLM_PROVIDER.lower() == "openai":
        try:
            from openai import OpenAI

            client = OpenAI()

            class OpenAILLM:
                def generate(self, system_prompt: str, user_prompt: str) -> str:
                    rsp = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                    )
                    return rsp.choices[0].message.content

            _llm_singleton = OpenAILLM()
            return _llm_singleton
        except Exception:
            pass
    _llm_singleton = NullLLM()
    return _llm_singleton
