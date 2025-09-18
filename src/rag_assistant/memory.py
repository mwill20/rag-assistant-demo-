# src/rag_assistant/memory.py
from collections import deque
import uuid

class SessionMemory:
    def __init__(self, max_turns: int = 5):
        self._sessions = {}
        self.max_turns = max_turns

    def _ensure_session(self, session_id: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "turns": deque(),   # store last N turns
                "summary": ""       # rolling summary
            }

    def add_turn(self, session_id: str, question: str, answer: str):
        self._ensure_session(session_id)
        sess = self._sessions[session_id]
        sess["turns"].append({"q": question, "a": answer})
        if len(sess["turns"]) > self.max_turns:
            # Summarize oldest turn into summary
            old = sess["turns"].popleft()
            sess["summary"] += f"Q: {old['q']} A: {old['a']}\n"

    def get_context(self, session_id: str) -> str:
        self._ensure_session(session_id)
        sess = self._sessions[session_id]
        context = ""
        if sess["summary"]:
            context += f"Conversation summary:\n{sess['summary']}\n"
        if sess["turns"]:
            context += "Recent turns:\n"
            for t in list(sess["turns"]):
                context += f"Q: {t['q']}\nA: {t['a']}\n"
        return context.strip()
