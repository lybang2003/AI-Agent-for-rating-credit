from __future__ import annotations

import json
from typing import Dict, List, Optional, Any

import joblib
import numpy as np

from app.config import settings


class CreditRatingPredictor:
    def __init__(self, model_path: Optional[str] = None, feature_order: Optional[List[str]] = None) -> None:
        self.model_path = model_path or settings.model_path
        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict) and "model" in bundle:
            # Bundle từ trainer nhanh
            self.model = bundle["model"]
            self.label_encoder = bundle.get("label_encoder")
            self.feature_order = bundle.get("feature_order", feature_order or [])
            self.model_version = getattr(self.model, "__class__", type("M", (), {})).__name__
        else:
            self.model = bundle
            self.label_encoder = None
            self.model_version = getattr(self.model, "__class__", type("M", (), {})).__name__
        # Classes mapping
        self.class_labels: Optional[List[str]] = None
        if settings.model_classes:
            try:
                self.class_labels = json.loads(settings.model_classes)
            except Exception:
                self.class_labels = None
        elif hasattr(self.model, "classes_"):
            try:
                # If classes_ are encoded ints, skip; if strings, use directly
                classes = list(getattr(self.model, "classes_"))
                if classes and isinstance(classes[0], str):
                    self.class_labels = classes
            except Exception:
                pass
        if feature_order is not None:
            self.feature_order = feature_order
        else:
            if settings.model_feature_order:
                try:
                    self.feature_order = json.loads(settings.model_feature_order)
                except Exception:
                    self.feature_order = []
            else:
                # best-effort: try from model if available
                self.feature_order = getattr(self.model, "feature_names_in_", []).tolist() if hasattr(self.model, "feature_names_in_") else []

    def ensure_feature_vector(self, feature_row: Dict[str, float]) -> np.ndarray:
        if self.feature_order:
            values = [float(feature_row.get(k, 0.0)) for k in self.feature_order]
        else:
            # fall back to sorted keys for deterministic order
            keys = sorted(feature_row.keys())
            self.feature_order = keys
            values = [float(feature_row[k]) for k in keys]
        return np.array(values, dtype=float).reshape(1, -1)

    def predict(self, feature_row: Dict[str, float]) -> Dict[str, Any]:
        x = self.ensure_feature_vector(feature_row)

        # 1. Predict
        y_pred = self.model.predict(x)

        # 2. Decode về nhãn gốc
        if getattr(self, "label_encoder", None) is not None:
            try:
                rating = self.label_encoder.inverse_transform(y_pred)[0]
            except Exception:
                rating = str(y_pred[0])
        elif self.class_labels is not None:
            try:
                idx = int(y_pred[0])
                if 0 <= idx < len(self.class_labels):
                    rating = self.class_labels[idx]
                else:
                    rating = str(y_pred[0])
            except Exception:
                rating = str(y_pred[0])
        else:
            rating = str(y_pred[0])

        # 3. Probability distribution (nếu có)
        prob_dist = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(x)
                prob_dist = proba[0].tolist()
            except Exception:
                pass

        return {
            "rating": rating,
            "probDist": prob_dist,
            "modelVersion": self.model_version,
        }


class DualCreditRatingPredictor:
    """
    Bộ dự đoán chạy nhiều model lần lượt và trả về kết quả của từng model.
    Dùng cho trường hợp cần so sánh 2+ model (ví dụ train1.pkl và train2.pkl).
    """

    def __init__(self, model_paths: list[str] | None = None, feature_order: Optional[List[str]] = None) -> None:  # type: ignore[name-defined]
        from typing import List  # local import to avoid affecting module top

        self.model_paths = model_paths or getattr(settings, "model_paths", [settings.model_path])
        self.predictors: List[CreditRatingPredictor] = []
        for p in self.model_paths:
            try:
                self.predictors.append(CreditRatingPredictor(model_path=p, feature_order=feature_order))
            except Exception:
                # Bỏ qua model lỗi để không chặn toàn bộ
                pass

    def predict_both(self, feature_row: Dict[str, float]) -> Dict[str, Any]:  # type: ignore[name-defined]
        results = []
        for idx, pred in enumerate(self.predictors):
            try:
                r = pred.predict(feature_row)
                r["modelPath"] = getattr(pred, "model_path", f"model_{idx}")
                results.append(r)
            except Exception as e:
                results.append({
                    "rating": None,
                    "probDist": None,
                    "modelVersion": None,
                    "modelPath": getattr(pred, "model_path", f"model_{idx}"),
                    "error": str(e),
                })
        return {"results": results}
