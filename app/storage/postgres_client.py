from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import settings


class PostgresClient:
    def __init__(self) -> None:
        self.engine: Optional[Engine] = None
        if settings.postgres_url:
            try:
                self.engine = create_engine(settings.postgres_url, pool_pre_ping=True)
                self._ensure_schema()
            except Exception:
                self.engine = None

    def _ensure_schema(self) -> None:
        if not self.engine:
            return
        with self.engine.begin() as conn:
            # features: wide numeric table by (company_id, feature_key, feature_value)
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS features (
                  company_id TEXT NOT NULL,
                  feature_key TEXT NOT NULL,
                  feature_value DOUBLE PRECISION,
                  PRIMARY KEY (company_id, feature_key)
                );
                """
            ))

    def upsert_features(self, company_id: str, features: Dict[str, float]) -> None:
        if not self.engine:
            return
        with self.engine.begin() as conn:
            for k, v in features.items():
                conn.execute(
                    text(
                        """
                        INSERT INTO features (company_id, feature_key, feature_value)
                        VALUES (:cid, :fkey, :fval)
                        ON CONFLICT (company_id, feature_key)
                        DO UPDATE SET feature_value = EXCLUDED.feature_value
                        """
                    ),
                    {"cid": company_id, "fkey": k, "fval": float(v)},
                )

    def get_features(self, company_id: str) -> Dict[str, float]:
        if not self.engine:
            return {}
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT feature_key, feature_value FROM features WHERE company_id = :cid"),
                {"cid": company_id},
            ).all()
        return {r[0]: float(r[1]) for r in rows}


pg = PostgresClient()

