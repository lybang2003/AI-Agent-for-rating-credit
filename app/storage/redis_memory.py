from __future__ import annotations

from typing import List, Dict, Any

try:
    import redis
except Exception:  # optional
    redis = None  # type: ignore

from app.config import settings


class ChatMemory:
    def __init__(self) -> None:
        self.client = None
        if redis and settings.redis_url:
            try:
                self.client = redis.from_url(settings.redis_url)
            except Exception:
                self.client = None

    def add_message(self, session_id: str, role: str, content: str) -> None:
        key = f"chat:{session_id}"
        entry = {"role": role, "content": content}
        if not self.client:
            return
        self.client.rpush(key, str(entry))
        self.client.ltrim(key, -50, -1)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        key = f"chat:{session_id}"
        if not self.client:
            return []
        items = self.client.lrange(key, 0, -1)
        out: List[Dict[str, Any]] = []
        for i in items:
            try:
                s = i.decode("utf-8") if isinstance(i, (bytes, bytearray)) else str(i)
                # naive eval-safe parse
                d: Dict[str, Any] = eval(s, {"__builtins__": {}}, {})
                out.append(d)
            except Exception:
                pass
        return out


chat_memory = ChatMemory()

