from __future__ import annotations

from typing import Dict, Any, List, Optional

import pandas as pd

from app.storage.postgres_client import pg


class InputNormalizer:
    def normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        company = payload.get("company") or payload.get("input", {}).get("company")
        as_of_date = payload.get("as_of_date") or payload.get("input", {}).get("as_of_date")
        features = payload.get("features") or payload.get("input", {}).get("features") or {}
        return {"company": company, "as_of_date": as_of_date, "features": features}


class DataCatalog:
    def plan(self, company: str, required: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "sources": [
                {"name": "features", "type": "internal"},
                {"name": "docs", "type": "internal"},
                {"name": "web", "type": "external"},
            ],
            "required": required or [],
            "company": company,
        }


class InternalRetrieval:
    def get_features(self, company: str) -> Dict[str, float]:
        # PostgreSQL first
        try:
            feats_pg = pg.get_features(company)
            if feats_pg:
                return feats_pg
        except Exception:
            pass

        # Fallback: read CSV if present
        try:
            df = pd.read_csv("corporateCreditRatingWithFinancialRatios.csv")
            # naive match on company name column variants
            for col in ["CompanyName", "company", "name"]:
                if col in df.columns:
                    rows = df[df[col] == company]
                    if len(rows) > 0:
                        row = rows.iloc[0].to_dict()
                        return {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        except Exception:
            pass

        return {}


class WebGather:
    def search_and_extract(self, company: str) -> Dict[str, float]:
        # Stub: có thể tích hợp Tavily/Bing + crawling sau
        return {}


class FeatureMerger:
    def merge(self, provided: Dict[str, float], internal: Dict[str, float], external: Dict[str, float]) -> Dict[str, float]:
        merged = dict(internal)
        merged.update(external)
        merged.update(provided)
        return merged

