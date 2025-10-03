from __future__ import annotations

from typing import Dict, Any, List
from tavily import TavilyClient


class WebFetcher:
    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key

    def search_metric(self, company: str, metric: str, until_year: int = 2025, max_results: int = 5) -> Dict[str, Any]:
        if not self.api_key:
            return {"company": company, "metric": metric, "results": [], "error": "missing_api_key"}
        client = TavilyClient(api_key=self.api_key)
        q = f"{company} {metric} by year until {until_year} financial ratio"
        res = client.search(q, search_depth="advanced", max_results=max_results)
        items: List[Dict[str, Any]] = []
        if isinstance(res, dict):
            for r in res.get("results", [])[:max_results]:
                items.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content") or r.get("snippet"),
                })
        return {"company": company, "metric": metric, "results": items}


