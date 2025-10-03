from __future__ import annotations

from typing import List, Dict, Any, Optional

try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None  # type: ignore

from app.config import settings


class PineconeStore:
    def __init__(self) -> None:
        self.pc = None
        self.index = None
        if Pinecone and settings.pinecone_api_key and settings.pinecone_index:
            try:
                self.pc = Pinecone(api_key=settings.pinecone_api_key)
                self.index = self.pc.Index(settings.pinecone_index)
            except Exception:
                self.pc = None
                self.index = None

    def upsert(self, items: List[Dict[str, Any]]) -> None:
        if not self.index:
            return
        vectors = []
        for it in items:
            vectors.append({
                "id": it.get("id"),
                "values": it.get("vector"),
                "metadata": it.get("metadata", {}),
            })
        self.index.upsert(vectors)

    def query(self, vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.index:
            return []
        res = self.index.query(vector=vector, top_k=top_k, filter=filter or {}, include_metadata=True)
        out: List[Dict[str, Any]] = []
        for m in res.get("matches", []):
            out.append({"id": m.get("id"), "score": m.get("score"), "metadata": m.get("metadata")})
        return out


pinecone_store = PineconeStore()

