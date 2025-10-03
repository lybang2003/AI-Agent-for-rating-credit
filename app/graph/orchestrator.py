from __future__ import annotations

from typing import Dict, Any

try:
    from langgraph.graph import StateGraph
except Exception:
    StateGraph = None  # optional

from app.agents.core import InputNormalizer, DataCatalog, InternalRetrieval, WebGather, FeatureMerger
from app.agents.ml_report import PredictorAgent, ExplainerAgent, ReporterAgent


class Orchestrator:
    def __init__(self) -> None:
        self.normalizer = InputNormalizer()
        self.catalog = DataCatalog()
        self.internal = InternalRetrieval()
        self.web = WebGather()
        self.merger = FeatureMerger()
        self.predictor = PredictorAgent()
        self.explainer = ExplainerAgent()
        self.reporter = ReporterAgent()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        norm = self.normalizer.normalize(payload)
        plan = self.catalog.plan(norm["company"], list((norm.get("features") or {}).keys()))
        internal_feats = self.internal.get_features(norm["company"]) if norm.get("company") else {}
        external_feats = {} if internal_feats else self.web.search_and_extract(norm.get("company", ""))
        merged = self.merger.merge(norm.get("features", {}), internal_feats, external_feats)

        pred = self.predictor.predict(merged)
        exp = self.explainer.explain(merged)
        return {"normalized": norm, "plan": plan, "features": merged, "prediction": pred, "explanation": exp}


orchestrator = Orchestrator()

