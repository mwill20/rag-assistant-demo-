# src/rag_assistant/llm_providers.py  (Groq tweaks)

from __future__ import annotations

from dataclasses import dataclass

from .config import settings


class BaseProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        raise NotImplementedError


@dataclass
class OpenAIProvider(BaseProvider):
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 512

    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            r = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            return None


@dataclass
class GroqProvider(BaseProvider):
    api_key: str
    # Use a more instruction-following model + stricter decoding
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 32

    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            r = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            return None


@dataclass
class GeminiProvider(BaseProvider):
    api_key: str
    model: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_tokens: int = 512

    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model, system_instruction=system_prompt)
            r = model.generate_content(user_prompt)
            text = getattr(r, "text", None)
            return (text or "").strip() or None
        except Exception:
            return None


class NullProvider(BaseProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        return None


def get_provider_stack(preference: list[str]) -> list[BaseProvider]:
    pref = [p.strip().lower() for p in (preference or []) if p]
    out: list[BaseProvider] = []
    if "openai" in pref and settings.OPENAI_API_KEY:
        out.append(OpenAIProvider(api_key=settings.OPENAI_API_KEY))
    if "groq" in pref and settings.GROQ_API_KEY:
        out.append(GroqProvider(api_key=settings.GROQ_API_KEY))
    if "gemini" in pref and settings.GEMINI_API_KEY:
        out.append(GeminiProvider(api_key=settings.GEMINI_API_KEY))
    out.append(NullProvider())
    return out
