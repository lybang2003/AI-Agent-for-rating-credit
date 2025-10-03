import os
import json
import pandas as pd
import streamlit as st
import google.generativeai as genai
from tavily import TavilyClient

from app.graph.orchestrator import orchestrator
from app.agents.ml_report import PredictorAgent
from app.etl.loader import load_csv_to_storage
from app.agents.ml_report import ReporterAgent
from app.config import settings
from app.agents.chat_prompts import get_prompts
from app.agents.data_query import CSVDataQuery
from app.agents.web_fetch import WebFetcher

st.set_page_config(page_title="Credit Rating Multi-Agent", layout="wide")

# Sidebar for navigation
with st.sidebar:
    st.header("Cấu hình")
    page = st.radio("Chọn trang", ["Main Interface", "Tools Interface"])

    with st.expander("⚙️ ETL", expanded=False):
        if st.button("Nạp ETL local CSV"):
            total = 0
            for path in ["corporateCreditRatingWithFinancialRatios.csv", "corporate_rating.csv"]:
                if os.path.exists(path):
                    try:
                        res = load_csv_to_storage(path)
                        total += res.get("loaded", 0)
                    except Exception as e:
                        st.error(f"ETL lỗi với {path}: {e}")
            st.success(f"ETL xong: {total} records")

if page == "Main Interface":
    st.title("Credit Rating Multi-Agent UI")
    tab_predict, tab_inspect = st.tabs(["Predict", "Inspect"])

    with tab_predict:
        st.subheader("Dự đoán rating")
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Tên công ty", value="ACME Corp")
            as_of = st.text_input("As-of date (YYYY-MM-DD)", value="2025-09-30")

            input_mode = st.radio("Chế độ nhập features", ["Form", "JSON"], horizontal=True)
            features: dict = {}

            if input_mode == "Form":
                st.markdown("#### Chỉ số tài chính")
                currentRatio = st.number_input("currentRatio", value=2.1, step=0.1)
                longTermDebtToCapital = st.number_input("longTermDebtToCapital", value=0.4, step=0.1)
                debtEquityRatio = st.number_input("debtEquityRatio", value=0.8, step=0.1)
                grossMargin = st.number_input("grossMargin", value=45.0, step=0.1)
                operatingMargin = st.number_input("operatingMargin", value=20.0, step=0.1)
                ebitMargin = st.number_input("ebitMargin", value=18.0, step=0.1)
                ebitdaMargin = st.number_input("ebitdaMargin", value=25.0, step=0.1)
                preTaxProfitMargin = st.number_input("preTaxProfitMargin", value=15.0, step=0.1)
                netProfitMargin = st.number_input("netProfitMargin", value=12.0, step=0.1)
                assetTurnover = st.number_input("assetTurnover", value=0.9, step=0.1)
                returnOnEquity = st.number_input("returnOnEquity", value=10.0, step=0.1)
                returnOnTangibleEquity = st.number_input("returnOnTangibleEquity", value=9.0, step=0.1)
                returnOnAssets = st.number_input("returnOnAssets", value=7.0, step=0.1)
                returnOnInvestment = st.number_input("returnOnInvestment", value=8.0, step=0.1)
                operatingCashFlowPerShare = st.number_input("operatingCashFlowPerShare", value=5.0, step=0.1)
                freeCashFlowPerShare = st.number_input("freeCashFlowPerShare", value=3.0, step=0.1)

                features = {
                    "currentRatio": float(currentRatio),
                    "longTermDebtToCapital": float(longTermDebtToCapital),
                    "debtEquityRatio": float(debtEquityRatio),
                    "grossMargin": float(grossMargin),
                    "operatingMargin": float(operatingMargin),
                    "ebitMargin": float(ebitMargin),
                    "ebitdaMargin": float(ebitdaMargin),
                    "preTaxProfitMargin": float(preTaxProfitMargin),
                    "netProfitMargin": float(netProfitMargin),
                    "assetTurnover": float(assetTurnover),
                    "returnOnEquity": float(returnOnEquity),
                    "returnOnTangibleEquity": float(returnOnTangibleEquity),
                    "returnOnAssets": float(returnOnAssets),
                    "returnOnInvestment": float(returnOnInvestment),
                    "operatingCashFlowPerShare": float(operatingCashFlowPerShare),
                    "freeCashFlowPerShare": float(freeCashFlowPerShare),
                }
            else:
                features_text = st.text_area(
                    "Features JSON",
                    value='{"currentRatio": 2.1, "debtEquityRatio": 0.8, "grossMargin": 45.0}',
                )
                try:
                    features = json.loads(features_text) if features_text.strip() else {}
                except Exception:
                    st.error("JSON features không hợp lệ")
                    features = {}

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Predict (một model)"):
                    result = orchestrator.run({"company": company, "as_of_date": as_of, "features": features})
                    pred = result.get("prediction", {})
                    rating_text = pred.get("rating")
                    if rating_text:
                        st.success(f"Mức rating của công ty: {rating_text}")
                    else:
                        st.warning("Không nhận được kết quả rating")
                    with st.expander("Giải thích"):
                        st.json(result.get("explanation", {}))
            with col_btn2:
                if st.button("Predict cả hai model"):
                    agent = PredictorAgent()
                    both = agent.predict_both(features)
                    results = both.get("results", []) if isinstance(both, dict) else []
                    if results:
                        for item in results:
                            rating_text = (item or {}).get("rating")
                            model_path = (item or {}).get("modelPath", "model")
                            if rating_text:
                                st.info(f"{model_path}: {rating_text}")
                            else:
                                st.warning(f"{model_path}: không có rating")
                    else:
                        st.warning("Không nhận được kết quả từ các model")
        with col2:
            st.caption("Feature hợp nhất (nếu có)")
            if st.button("Xem feature hợp nhất"):
                result = orchestrator.run({"company": company, "as_of_date": as_of, "features": features})
                df = pd.DataFrame([result.get("features", {})]).T
                df.columns = ["value"]
                st.dataframe(df)

        st.divider()
        st.subheader("Chat hỏi đáp")
        colc1, colc2 = st.columns([3, 2])
        with colc1:
            user_query = st.text_area("Câu hỏi", value="Dự đoán rating và yếu tố chính của ACME Corp?", height=90)
            use_tavily = st.checkbox("Bật tìm kiếm Tavily làm ngữ cảnh", value=True)
            num_results = st.slider("Số kết quả tìm kiếm", 1, 10, 3) if use_tavily else 0
            company_ctx = st.text_input("Công ty (tùy chọn)", value="ACME Corp")
            if st.button("Hỏi"):
                context_snippets = []
                if use_tavily:
                    if not settings.tavily_api_key:
                        st.warning("Chưa có TAVILY_API_KEY trong môi trường. Bỏ qua tìm kiếm.")
                    else:
                        try:
                            tavily = TavilyClient(api_key=settings.tavily_api_key)
                            q = user_query if not company_ctx else f"{user_query} công ty {company_ctx}"
                            res = tavily.search(q, search_depth="basic", max_results=num_results)
                            if isinstance(res, dict):
                                for item in (res.get("results") or [])[:num_results]:
                                    snippet = item.get("content") or item.get("snippet") or ""
                                    if snippet:
                                        context_snippets.append(snippet)
                        except Exception as e:
                            st.error(f"Tavily lỗi: {e}")

                if not settings.gemini_api_key:
                    st.error("Chưa cấu hình GEMINI_API_KEY. Đặt trong .env hoặc môi trường rồi thử lại.")
                else:
                    try:
                        genai.configure(api_key=settings.gemini_api_key)
                        # Use default model
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        system_prompt = (
                            "Bạn là trợ lý tài chính. Sử dụng ngữ cảnh web nếu có, "
                            "và trả lời ngắn gọn, có cấu trúc. Nếu câu hỏi liên quan dự đoán rating, "
                            "hãy gợi ý dùng phần Predict để có con số từ mô hình nội bộ."
                        )
                        context_text = "\n\n".join(context_snippets) if context_snippets else ""
                        full_prompt = (
                            f"[HỆ THỐNG]\n{system_prompt}\n\n"
                            f"[NGỮ CẢNH WEB]\n{context_text}\n\n"
                            f"[CÂU HỎI]\n{user_query}\n"
                            + (f"\n[CÔNG TY]\n{company_ctx}\n" if company_ctx else "")
                        )
                        resp = model.generate_content(full_prompt)
                        st.markdown(resp.text or "(Không có nội dung)")
                        if context_text:
                            with st.expander("Nguồn ngữ cảnh (Tavily)"):
                                st.write(context_text)
                    except Exception as e:
                        st.error(f"Gemini lỗi: {e}")
        with colc2:
            # Empty column now
            st.write("")

    with tab_inspect:
        st.subheader("Kiểm tra CSV")
        for path in ["corporateCreditRatingWithFinancialRatios.csv", "corporate_rating.csv"]:
            if os.path.exists(path):
                with st.expander(path):
                    try:
                        df = pd.read_csv(path).head(50)
                        st.dataframe(df)
                    except Exception as e:
                        st.error(str(e))

else:  # Tools Interface
    st.title("Tools Interface")
    tab_tools = st.tabs(["Query CSV", "Radar Chart", "Web Fetch", "Gemini Tools"])

    with tab_tools[0]:  # Query CSV
        st.subheader("Truy vấn dữ liệu nội bộ")
        dq = CSVDataQuery()
        all_metrics = [
            "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding", "netProfitMargin",
            "pretaxProfitMargin", "grossProfitMargin", "operatingProfitMargin", "returnOnAssets",
            "returnOnCapitalEmployed", "returnOnEquity", "assetTurnover", "fixedAssetTurnover",
            "debtEquityRatio", "debtRatio", "effectiveTaxRate",
            "freeCashFlowOperatingCashFlowRatio", "freeCashFlowPerShare", "cashPerShare",
            "companyEquityMultiplier", "ebitPerRevenue", "enterpriseValueMultiple",
            "operatingCashFlowPerShare", "operatingCashFlowSalesRatio", "payablesTurnover",
        ]
        sidebar_company = st.text_input("Công ty", value="ACME Corp", key="sb_company")
        sb_metric = st.selectbox("Chỉ số", options=all_metrics, index=0, key="sb_metric")
        if st.button("Query", key="sb_do_query"):
            ts = dq.query_metric_timeseries(sidebar_company or "", sb_metric)
            if ts.get("years"):
                st.success(f"Tìm thấy {sb_metric} của {sidebar_company}: ")
                st.dataframe({"year": ts["years"], "value": ts["values"]})
            else:
                st.info("Không tìm thấy dữ liệu phù hợp trong CSV nội bộ")
                with st.expander("Chi tiết gỡ lỗi"):
                    st.json(ts.get("debug", {}))

    with tab_tools[1]:  # Radar Chart
        st.subheader("Biểu đồ nhanh từ features")
        chart_company = st.text_input("Công ty", value="ACME Corp", key="sb_chart_company")
        sb_metrics_chart = st.text_input("Metrics", value="CurrentRatio,DebtToEquity,ROA,ROE", key="sb_metrics")
        if st.button("Vẽ radar", key="sb_do_chart"):
            try:
                r = orchestrator.run({"company": chart_company})
                feats = r.get("features", {})
                metrics_list = [m.strip() for m in sb_metrics_chart.split(",") if m.strip()]
                metrics_dict = {m: float(feats.get(m, 0.0)) for m in metrics_list}
                reporter = ReporterAgent()
                chart = reporter.chart_radar(chart_company, metrics_dict)
                st.components.v1.html(chart.get("html", "<p>No chart</p>"), height=420, scrolling=True)
            except Exception as e:
                st.error(f"Vẽ biểu đồ lỗi: {e}")

    with tab_tools[2]:  # Web Fetch
        st.subheader("Web fetch")
        web_company = st.text_input("Công ty", value="ACME Corp", key="sb_web_company")
        metric_for_web = st.text_input("Metric", value="Current Ratio", key="web_metric")
        use_tavily = st.checkbox("Bật Tavily", value=True, key="sb_tavily")
        num_results = st.slider("Số kết quả web", 1, 10, 3, key="sb_num") if use_tavily else 0
        until_year = st.number_input("Tới năm", value=2025, step=1)
        if st.button("Tìm", key="sb_do_web"):
            wf = WebFetcher(settings.tavily_api_key if use_tavily else None)
            res = wf.search_metric(web_company or "", metric_for_web, until_year=int(until_year), max_results=int(num_results or 0))
            items = res.get("results", [])
            if items:
                for it in items:
                    st.write(f"- [{it.get('title')}]({it.get('url')})")
            else:
                st.info("Không tìm thấy kết quả phù hợp")

    with tab_tools[3]:  # Gemini Tools
        st.subheader("Gemini Tools")
        st.markdown("#### Gợi ý prompt sẵn có")
        with st.expander("Xem danh sách prompts (>=30)"):
            st.code("\n".join(get_prompts()))
        st.markdown("#### Liệt kê model khả dụng")
        if st.button("Liệt kê model", key="sb_list_models"):
            try:
                if not settings.gemini_api_key:
                    st.error("Chưa cấu hình GEMINI_API_KEY")
                else:
                    genai.configure(api_key=settings.gemini_api_key)
                    models = getattr(genai, "list_models", lambda: [])()
                    names = [getattr(m, "name", None) for m in (models or []) if getattr(m, "name", None)]
                    st.code("\n".join(names) or "(trống)")
            except Exception as e:
                st.error(str(e))