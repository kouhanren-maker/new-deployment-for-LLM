# agent/runtime/memory.py
from typing import Dict, Any

class Memory:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get(self, user_id: str) -> Dict[str, Any]:
        return self.sessions.get(user_id, {})

    def update(self, user_id: str, key: str, value: Any):
        self.sessions.setdefault(user_id, {})[key] = value

    def recent_query(self, user_id: str) -> str:
        return self.get(user_id).get("last_query", "")

mem = Memory()
