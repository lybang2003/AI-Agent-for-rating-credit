from __future__ import annotations

from typing import Dict, Any, Optional
import os
import pickle

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


class InvestmentPredictorAgent:
    def __init__(self, model_path: str = "binary_model_92_percent_accuracy.pkl") -> None:
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.feature_columns: Optional[list] = None

    def _ensure_loaded(self) -> None:
        """Lazy load model để tránh crash khi khởi tạo"""
        if self.model is None:
            try:
                # Tìm đường dẫn tuyệt đối đến model
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
                full_model_path = os.path.join(project_root, self.model_path)
                
                if not os.path.exists(full_model_path):
                    raise FileNotFoundError(f"Model file not found: {full_model_path}")
                
                with open(full_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Giả định model được lưu dưới dạng dict với keys: 'model', 'feature_columns'
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.feature_columns = model_data.get('feature_columns', [])
                else:
                    # Nếu model được lưu trực tiếp
                    self.model = model_data
                    self.feature_columns = []
                    
            except Exception as e:
                print(f"Error loading investment model: {e}")
                self.model = None
                self.feature_columns = None

    def predict_investment(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Dự đoán có nên đầu tư hay không (binary classification)
        
        Args:
            features: Dictionary chứa các chỉ số tài chính
            
        Returns:
            Dict chứa kết quả dự đoán: invest (True/False), confidence, probability
        """
        try:
            self._ensure_loaded()
            
            if self.model is None:
                # Fallback heuristic khi model không load được
                return self._heuristic_investment_decision(features)
            
            # Chuẩn bị features cho model
            if self.feature_columns:
                # Sử dụng feature columns đã được định nghĩa
                model_features = []
                for col in self.feature_columns:
                    value = features.get(col, 0.0)
                    model_features.append(float(value))
            else:
                # Sử dụng tất cả features có sẵn
                model_features = list(features.values())
            
            # Kiểm tra số features
            expected_features = 16  # Số features mà model binary cần (LightGBM)
            if len(model_features) != expected_features:
                # Bổ sung features thiếu với giá trị mặc định
                while len(model_features) < expected_features:
                    model_features.append(0.0)
                # Hoặc cắt bớt nếu quá nhiều
                model_features = model_features[:expected_features]
            
            # Dự đoán
            if hasattr(self.model, 'predict_proba'):
                # Model có predict_proba (sklearn)
                probabilities = self.model.predict_proba([model_features])[0]
                prediction = self.model.predict([model_features])[0]
                
                # Giả định class 1 = invest, class 0 = not invest
                invest_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                confidence = max(probabilities)
                
            elif hasattr(self.model, 'predict'):
                # Model chỉ có predict (LightGBM)
                prediction_raw = self.model.predict([model_features])[0]
                
                # LightGBM trả về raw score, cần chuyển thành probability
                # Sử dụng sigmoid để chuyển score thành probability
                import math
                probability = 1 / (1 + math.exp(-prediction_raw))
                
                # Quyết định đầu tư dựa trên threshold 0.5
                prediction = 1 if probability > 0.5 else 0
                invest_prob = probability
                confidence = abs(probability - 0.5) * 2  # Confidence dựa trên khoảng cách từ 0.5
                
            else:
                raise ValueError("Model không có method predict hoặc predict_proba")
            
            invest = bool(prediction)
            
            return {
                "invest": invest,
                "confidence": float(confidence),
                "probability": float(invest_prob),
                "model_version": "binary_model_92_percent_accuracy",
                "features_used": len(model_features),
                "features_provided": len(features),
                "features_padded": len(model_features) > len(features),
                "error": None
            }
            
        except Exception as e:
            # Fallback heuristic khi có lỗi
            return self._heuristic_investment_decision(features, str(e))

    def _heuristic_investment_decision(self, features: Dict[str, float], error: str = None) -> Dict[str, Any]:
        """
        Heuristic fallback để quyết định đầu tư dựa trên các chỉ số tài chính
        """
        # Lấy các chỉ số quan trọng
        roe = float(features.get("returnOnEquity", features.get("ROE", 0.0)))
        current_ratio = float(features.get("currentRatio", features.get("CurrentRatio", 0.0)))
        debt_equity = float(features.get("debtEquityRatio", features.get("DebtToEquity", 0.0)))
        net_margin = float(features.get("netProfitMargin", features.get("NetProfitMargin", 0.0)))
        gross_margin = float(features.get("grossMargin", features.get("GrossMargin", 0.0)))
        
        # Tính điểm đầu tư
        score = 0.0
        
        # ROE > 15% là tốt
        if roe > 0.15:
            score += 3
        elif roe > 0.10:
            score += 2
        elif roe > 0.05:
            score += 1
        
        # Current Ratio > 1.5 là tốt
        if current_ratio > 2.0:
            score += 2
        elif current_ratio > 1.5:
            score += 1
        elif current_ratio < 1.0:
            score -= 2
        
        # Debt/Equity < 1.0 là tốt
        if debt_equity < 0.5:
            score += 2
        elif debt_equity < 1.0:
            score += 1
        elif debt_equity > 2.0:
            score -= 2
        
        # Net Margin > 10% là tốt
        if net_margin > 0.15:
            score += 2
        elif net_margin > 0.10:
            score += 1
        elif net_margin < 0.05:
            score -= 1
        
        # Gross Margin > 30% là tốt
        if gross_margin > 0.40:
            score += 1
        elif gross_margin < 0.20:
            score -= 1
        
        # Quyết định đầu tư
        invest = score >= 4
        confidence = min(0.9, max(0.3, abs(score) / 8.0))
        probability = 0.5 + (score / 10.0)
        probability = max(0.1, min(0.9, probability))
        
        return {
            "invest": invest,
            "confidence": confidence,
            "probability": probability,
            "model_version": "heuristic",
            "score": score,
            "error": error
        }

    def get_investment_recommendation(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Tạo khuyến nghị đầu tư chi tiết
        """
        result = self.predict_investment(features)
        
        # Tạo khuyến nghị dựa trên kết quả
        if result["invest"]:
            if result["confidence"] > 0.8:
                recommendation = "MẠNH MẼ KHUYẾN NGHỊ ĐẦU TƯ"
                risk_level = "Thấp"
            elif result["confidence"] > 0.6:
                recommendation = "KHUYẾN NGHỊ ĐẦU TƯ"
                risk_level = "Trung bình"
            else:
                recommendation = "CÓ THỂ ĐẦU TƯ"
                risk_level = "Trung bình-Cao"
        else:
            if result["confidence"] > 0.8:
                recommendation = "KHÔNG NÊN ĐẦU TƯ"
                risk_level = "Cao"
            elif result["confidence"] > 0.6:
                recommendation = "THẬN TRỌNG KHI ĐẦU TƯ"
                risk_level = "Trung bình-Cao"
            else:
                recommendation = "CẦN XEM XÉT THÊM"
                risk_level = "Không xác định"
        
        return {
            **result,
            "recommendation": recommendation,
            "risk_level": risk_level,
            "reasoning": self._generate_reasoning(features, result),
            "ai_explanation": self.get_ai_explanation(features, result)
        }
    
    def _generate_reasoning(self, features: Dict[str, float], result: Dict[str, Any]) -> str:
        """Tạo lý do cho khuyến nghị đầu tư"""
        reasons = []
        
        roe = float(features.get("returnOnEquity", features.get("ROE", 0.0)))
        current_ratio = float(features.get("currentRatio", features.get("CurrentRatio", 0.0)))
        debt_equity = float(features.get("debtEquityRatio", features.get("DebtToEquity", 0.0)))
        
        if roe > 0.15:
            reasons.append(f"- ROE cao ({roe:.1%}) cho thấy hiệu quả sử dụng vốn tốt")
        elif roe < 0.05:
            reasons.append(f"- ROE thấp ({roe:.1%}) cho thấy hiệu quả sử dụng vốn kém")
        
        if current_ratio > 2.0:
            reasons.append(f"- Thanh khoản tốt (Current Ratio: {current_ratio:.2f})")
        elif current_ratio < 1.0:
            reasons.append(f"- Thanh khoản yếu (Current Ratio: {current_ratio:.2f})")
        
        if debt_equity < 0.5:
            reasons.append(f"- Cơ cấu nợ an toàn (D/E: {debt_equity:.2f})")
        elif debt_equity > 2.0:
            reasons.append(f"- Đòn bẩy cao (D/E: {debt_equity:.2f})")
        
        if not reasons:
            reasons.append("Cần phân tích thêm các chỉ số tài chính khác")
        
        return "\n".join(reasons)

    def get_ai_explanation(self, features: Dict[str, float], result: Dict[str, Any]) -> str:
        """Sử dụng AI để giải thích khuyến nghị đầu tư"""
        try:
            import google.generativeai as genai
            from app.config import settings
            import time
            
            if not settings.gemini_api_key:
                return "Không có API key để tạo giải thích AI"
            
            # Kiểm tra API key format
            if len(settings.gemini_api_key) < 20:
                return "API key không hợp lệ"
            
            # Chuẩn bị dữ liệu cho AI
            invest_decision = "NÊN ĐẦU TƯ" if result['invest'] else "KHÔNG NÊN ĐẦU TƯ"
            probability = result['probability']
            
            # Tạo prompt cho AI
            prompt = f"""
Bạn là chuyên gia tài chính. Hãy giải thích ngắn gọn (2-3 câu) tại sao với các chỉ số tài chính sau lại {invest_decision}:

CHỈ SỐ TÀI CHÍNH:
- ROE (Return on Equity): {features.get('returnOnEquity', 0):.1%}
- Current Ratio: {features.get('currentRatio', 0):.2f}
- Debt/Equity Ratio: {features.get('debtEquityRatio', 0):.2f}
- Gross Margin: {features.get('grossMargin', 0):.1%}
- Net Profit Margin: {features.get('netProfitMargin', 0):.1%}
- Operating Margin: {features.get('operatingMargin', 0):.1%}

KẾT QUẢ DỰ ĐOÁN: {invest_decision} (Xác suất: {probability:.1%})

Yêu cầu:
1. Giải thích khái niệm của các chỉ số quan trọng
2. Phân tích tại sao kết quả là {invest_decision}
3. Ngắn gọn, dễ hiểu, phù hợp với nhà đầu tư

Trả lời bằng tiếng Việt, tối đa 150 từ.
"""
            
            genai.configure(api_key=settings.gemini_api_key)

            # Thử phát hiện model khả dụng theo API key trước (tránh 404 do không có quyền)
            model_names = []
            try:
                available = genai.list_models()
                dynamic = []
                for m in available:
                    # Lọc model có hỗ trợ generateContent
                    methods = getattr(m, 'supported_generation_methods', []) or []
                    name = getattr(m, 'name', '')
                    if any('generate' in str(x).lower() for x in methods) and name:
                        # genai.GenerativeModel nhận tên không cần prefix 'models/'
                        dynamic.append(name.split('/')[-1])
                # Ưu tiên các model flash/pro mới
                preferred_order = [
                    'gemini-2.5-flash',
                    'gemini-1.5-flash',
                    'gemini-1.5-pro',
                    'gemini-1.5-flash-8b',
                    'gemini-1.5-pro-latest',
                ]
                # Sắp xếp dynamic theo preferred_order trước, sau đó giữ nguyên phần còn lại
                preferred_set = set(preferred_order)
                ordered = [m for m in preferred_order if m in dynamic]
                ordered += [m for m in dynamic if m not in preferred_set]
                model_names = ordered[:10]  # giới hạn tối đa 10 model thử
            except Exception as _:
                # Nếu list_models thất bại, dùng danh sách fallback tĩnh
                model_names = [
                    'gemini-2.5-flash',
                    'gemini-1.5-flash',
                    'gemini-1.5-pro',
                ]
            max_retries = 2
            result_text = None

            def _extract_text(resp: Any) -> str:
                if not resp:
                    return ""
                # SDK có .text nhanh nếu hợp lệ
                if hasattr(resp, 'text') and resp.text:
                    try:
                        return resp.text
                    except Exception:
                        pass
                # Thử qua parts
                try:
                    parts = getattr(resp, 'parts', None)
                    if parts:
                        texts = []
                        for p in parts:
                            if hasattr(p, 'text') and p.text:
                                texts.append(p.text)
                        if texts:
                            return "".join(texts)
                except Exception:
                    pass
                # Thử qua candidates
                try:
                    candidates = getattr(resp, 'candidates', None)
                    if candidates:
                        texts = []
                        for c in candidates:
                            content = getattr(c, 'content', None)
                            if content and getattr(content, 'parts', None):
                                for p in content.parts:
                                    if hasattr(p, 'text') and p.text:
                                        texts.append(p.text)
                        if texts:
                            return "".join(texts)
                except Exception:
                    pass
                return ""

            for model_name in model_names:
                for retry in range(max_retries):
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                max_output_tokens=300,
                                temperature=0.6,
                            )
                        )
                        # Log finish_reason nếu có
                        try:
                            if response and getattr(response, 'candidates', None):
                                fr = getattr(response.candidates[0], 'finish_reason', None)
                                print(f"Finish reason: {fr}")
                        except Exception:
                            pass

                        text = _extract_text(response)
                        if text:
                            result_text = text
                            print(f"Success with model: {model_name}")
                            break

                        # Nếu MAX_TOKENS (2) hoặc nội dung rỗng, thử prompt ngắn
                        short_prompt = (
                            f"Giải thích ngắn gọn tại sao {invest_decision} với ROE {features.get('returnOnEquity', 0):.1%}, "
                            f"Current Ratio {features.get('currentRatio', 0):.2f}. Trả lời bằng tiếng Việt, tối đa 50 từ."
                        )
                        short_resp = model.generate_content(short_prompt)
                        text_short = _extract_text(short_resp)
                        if text_short:
                            result_text = text_short
                            print(f"Success with model: {model_name} (short prompt)")
                            break
                    except Exception as model_error:
                        print(f"Model {model_name} attempt {retry+1} failed: {model_error}")
                        if retry < max_retries - 1:
                            time.sleep(1)
                        continue

                if result_text:
                    break

            if not result_text:
                print("All models failed, using fallback")
                return self._generate_fallback_explanation(features, result)

            return result_text
            
        except Exception as e:
            error_msg = str(e)
            print(f"AI Explanation Error: {error_msg}")
            
            # Xử lý các lỗi phổ biến
            if any(keyword in error_msg.lower() for keyword in ["quota", "429", "rate", "limit", "exceeded"]):
                print("Quota/Rate limit exceeded, using fallback")
                return self._generate_fallback_explanation(features, result)
            elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                print("API key issue, using fallback")
                return self._generate_fallback_explanation(features, result)
            elif "model" in error_msg.lower() or "not found" in error_msg.lower():
                print("Model not found, using fallback")
                return self._generate_fallback_explanation(features, result)
            else:
                print(f"Unknown error, using fallback: {error_msg}")
                return self._generate_fallback_explanation(features, result)
    
    def _generate_fallback_explanation(self, features: Dict[str, float], result: Dict[str, Any]) -> str:
        """Tạo giải thích fallback khi AI API không khả dụng"""
        invest_decision = "NÊN ĐẦU TƯ" if result['invest'] else "KHÔNG NÊN ĐẦU TƯ"
        probability = result['probability']
        
        # Phân tích các chỉ số chính
        roe = features.get('returnOnEquity', 0)
        current_ratio = features.get('currentRatio', 0)
        debt_equity = features.get('debtEquityRatio', 0)
        gross_margin = features.get('grossMargin', 0)
        
        explanations = []
        
        # ROE analysis
        if roe > 15:
            explanations.append(f"ROE cao ({roe:.1%}) cho thấy hiệu quả sử dụng vốn tốt")
        elif roe > 10:
            explanations.append(f"ROE trung bình ({roe:.1%}) cho thấy khả năng sinh lời ổn định")
        else:
            explanations.append(f"ROE thấp ({roe:.1%}) cho thấy hiệu quả sử dụng vốn cần cải thiện")
        
        # Current Ratio analysis
        if current_ratio > 2:
            explanations.append(f"Current Ratio cao ({current_ratio:.2f}) cho thấy khả năng thanh toán tốt")
        elif current_ratio > 1:
            explanations.append(f"Current Ratio ổn định ({current_ratio:.2f}) cho thấy khả năng thanh toán đủ")
        else:
            explanations.append(f"Current Ratio thấp ({current_ratio:.2f}) cho thấy rủi ro thanh khoản")
        
        # Debt/Equity analysis
        if debt_equity < 0.5:
            explanations.append(f"Tỷ lệ nợ/vốn thấp ({debt_equity:.2f}) cho thấy cấu trúc tài chính an toàn")
        elif debt_equity < 1:
            explanations.append(f"Tỷ lệ nợ/vốn hợp lý ({debt_equity:.2f}) cho thấy cấu trúc tài chính ổn định")
        else:
            explanations.append(f"Tỷ lệ nợ/vốn cao ({debt_equity:.2f}) cho thấy rủi ro tài chính")
        
        # Gross Margin analysis
        if gross_margin > 40:
            explanations.append(f"Gross Margin cao ({gross_margin:.1%}) cho thấy khả năng cạnh tranh tốt")
        elif gross_margin > 20:
            explanations.append(f"Gross Margin ổn định ({gross_margin:.1%}) cho thấy hiệu quả hoạt động")
        else:
            explanations.append(f"Gross Margin thấp ({gross_margin:.1%}) cho thấy áp lực cạnh tranh")
        
        # Kết luận
        if result['invest']:
            conclusion = f"Dựa trên phân tích, {invest_decision} với xác suất {probability:.1%}. "
        else:
            conclusion = f"Dựa trên phân tích, {invest_decision} với xác suất {probability:.1%}. "
        
        return conclusion + " ".join(explanations[:3]) + " (Giải thích tự động - AI API tạm thời không khả dụng)"

