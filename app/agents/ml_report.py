from __future__ import annotations

from typing import Dict, Any, Optional

import plotly.graph_objects as go

from app.ml.predictor import CreditRatingPredictor, DualCreditRatingPredictor


class PredictorAgent:
    def __init__(self) -> None:
        # Lazy-load để tránh crash khi môi trường lib chưa khớp
        self.predictor: Optional[CreditRatingPredictor] = None
        self.dual: Optional[DualCreditRatingPredictor] = None

    def _ensure_loaded(self) -> None:
        if self.predictor is None:
            # Trì hoãn nạp model đến khi thật sự cần
            self.predictor = CreditRatingPredictor()
        if self.dual is None:
            self.dual = DualCreditRatingPredictor()

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        try:
            self._ensure_loaded()
            if self.predictor is None:
                raise RuntimeError("Predictor not loaded")
            return self.predictor.predict(features)
        except Exception as e:
            # Fallback heuristic đơn giản để demo khi model không nạp được
            rating = self._heuristic_rating(features)
            return {
                "rating": rating,
                "probDist": None,
                "modelVersion": "heuristic",
                "error": f"Predictor load/execute error: {e}"
            }

    def predict_both(self, features: Dict[str, float]) -> Dict[str, Any]:
        try:
            self._ensure_loaded()
            if self.dual is None:
                raise RuntimeError("Dual predictor not loaded")
            return self.dual.predict_both(features)
        except Exception as e:
            # Trả về heuristic như dự phòng
            return {
                "results": [
                    {
                        "rating": self._heuristic_rating(features),
                        "probDist": None,
                        "modelVersion": "heuristic",
                        "modelPath": "heuristic",
                        "error": f"Dual predictor error: {e}",
                    }
                ]
            }

    def _heuristic_rating(self, feats: Dict[str, float]) -> str:
        # Điểm dựa trên một số chỉ số phổ biến
        score = 0.0
        gm = float(feats.get("grossMargin", feats.get("GrossMargin", 0.0)))
        om = float(feats.get("operatingMargin", feats.get("OperatingMargin", 0.0)))
        em = float(feats.get("ebitMargin", feats.get("EbitMargin", 0.0)))
        ebitda = float(feats.get("ebitdaMargin", feats.get("EbitdaMargin", 0.0)))
        d2e = float(feats.get("debtEquityRatio", feats.get("DebtToEquity", 0.0)))
        cr = float(feats.get("currentRatio", feats.get("CurrentRatio", 0.0)))
        roe = float(feats.get("returnOnEquity", feats.get("ROE", 0.0)))

        score += gm * 0.05
        score += om * 0.08
        score += em * 0.05
        score += ebitda * 0.08
        score += roe * 0.5
        score += max(min((cr - 1.0), 2.0), -1.0) * 2.0
        score -= max(d2e - 1.0, 0.0) * 5.0

        # Ánh xạ điểm sang hạng (chỉ để minh hoạ/demo)
        if score >= 25:
            return "AAA"
        if score >= 20:
            return "AA"
        if score >= 15:
            return "A"
        if score >= 10:
            return "BBB"
        if score >= 6:
            return "BB"
        if score >= 3:
            return "B"
        return "CCC"


class ExplainerAgent:
    def explain(self, features: Dict[str, float]) -> Dict[str, Any]:
        # Placeholder: có thể tích hợp SHAP sau
        top = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        return {
            "method": "heuristic-top-abs",
            "top_features": [{"name": k, "value": v} for k, v in top],
        }


class ReporterAgent:
    def chart_radar(self, company: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        if not metrics:
            return {"html": "<p>No metrics</p>"}
        names = list(metrics.keys())[:10]
        values = [metrics[k] for k in names]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=names + [names[0]], fill="toself", name=company))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        html = fig.to_html(include_plotlyjs="cdn")
        return {"html": html}

    def chart_compare(self, companies_metrics: Dict[str, Dict[str, float]], top_k: int = 8) -> Dict[str, Any]:
        # Bar grouped theo metric so sánh nhiều công ty
        if not companies_metrics:
            return {"html": "<p>No data</p>"}
        # Chọn danh sách metric chung
        all_metrics = set()
        for m in companies_metrics.values():
            all_metrics.update(m.keys())
        metrics = list(all_metrics)[:top_k]
        fig = go.Figure()
        for company, m in companies_metrics.items():
            vals = [float(m.get(k, 0.0)) for k in metrics]
            fig.add_trace(go.Bar(name=company, x=metrics, y=vals))
        fig.update_layout(barmode='group')
        return {"html": fig.to_html(include_plotlyjs="cdn")}

    def table_list(self, items: Dict[str, float], title: str = "Danh sách") -> Dict[str, Any]:
        # Trả về HTML bảng liệt kê
        if not items:
            return {"html": "<p>No items</p>"}
        rows = ''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in items.items()])
        html = f"""
        <h4>{title}</h4>
        <table border='1' cellpadding='6' cellspacing='0'>
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """
        return {"html": html}

