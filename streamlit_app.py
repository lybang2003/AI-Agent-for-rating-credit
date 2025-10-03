import os
import json
import pandas as pd
import streamlit as st
import google.generativeai as genai
import requests
import os
from dotenv import load_dotenv
load_dotenv()
from app.agents.financial_ratios import *
import datetime
from app.agents.financial_ratios import (
    current_ratio, quick_ratio, cash_ratio, days_of_sales_outstanding, net_profit_margin, pretax_profit_margin, gross_profit_margin, operating_profit_margin,
    return_on_assets, return_on_capital_employed, return_on_equity, asset_turnover, fixed_asset_turnover, debt_equity_ratio, debt_ratio, company_equity_multiplier,
    effective_tax_rate, free_cash_flow_operating_cash_flow_ratio, free_cash_flow_per_share, operating_cash_flow_per_share, operating_cash_flow_sales_ratio,
    cash_per_share, ebit_per_revenue, enterprise_value_multiple, payables_turnover, safe_div
)
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
                # as_of = st.text_input("As-of date (YYYY-MM-DD)", value="2025-09-30")

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
                    result = orchestrator.run({"company": company, "features": features})
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
                result = orchestrator.run({"company": company, "features": features})
                df = pd.DataFrame([result.get("features", {})]).T
                df.columns = ["value"]
                st.dataframe(df)

        st.divider()
        st.subheader("Chat hỏi đáp")
        # Thêm nút xóa lịch sử hội thoại
        if st.button("Xóa lịch sử hội thoại"):
            st.session_state["chat_history"] = []
        # Khởi tạo lịch sử hội thoại nếu chưa có
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        st.markdown("#### Chat Bot Tài chính")
        # Hiển thị lịch sử hội thoại
        for chat in st.session_state["chat_history"]:
            if chat["role"] == "user":
                st.markdown(f"<div style='text-align:right; color:blue'><b>Bạn:</b> {chat['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left; color:green'><b>Bot:</b> {chat['content']}</div>", unsafe_allow_html=True)

        user_query = st.text_area("Câu hỏi", value="", height=90)
    use_tavily = True
    num_results = 10
    if st.button("Gửi câu hỏi") and user_query.strip():
            # API keys từ .env
            alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
            finnhub_api_key = os.getenv("FINNHUB_API_KEY")
            financial_api_key = os.getenv("FINANCIAL_API_KEY")

            def get_alphavantage_price(symbol):
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alphavantage_api_key}"
                try:
                    resp = requests.get(url)
                    if resp.status_code == 200:
                        data = resp.json().get("Global Quote", {})
                        return data
                except Exception as e:
                    st.error(f"AlphaVantage API lỗi: {e}")
                return None

            def get_finnhub_news(symbol):
                today = datetime.date.today()
                last_week = today - datetime.timedelta(days=7)
                url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={last_week}&to={today}&token={finnhub_api_key}"
                try:
                    resp = requests.get(url)
                    if resp.status_code == 200:
                        return resp.json()
                except Exception as e:
                    st.error(f"Finnhub API lỗi: {e}")
                return None

            import re
            def get_financial_data(company, api_key, year=None):
                # Ví dụ lấy Balance Sheet, Income Statement, Cash Flow từ financialmodelingprep.com
                base_url = "https://financialmodelingprep.com/api/v3"
                endpoints = {
                    "balance_sheet": f"{base_url}/balance-sheet-statement/{company}?apikey={api_key}",
                    "income_statement": f"{base_url}/income-statement/{company}?apikey={api_key}",
                    "cash_flow": f"{base_url}/cash-flow-statement/{company}?apikey={api_key}"
                }
                data = {}
                try:
                    for key, url in endpoints.items():
                        resp = requests.get(url)
                        if resp.status_code == 200:
                            result = resp.json()
                            # Nếu có năm, lọc theo năm
                            if year and isinstance(result, list):
                                result = [r for r in result if str(r.get("date", "")).startswith(str(year))]
                                if result:
                                    data[key] = result[0]
                                else:
                                    data[key] = {}
                            elif isinstance(result, list) and result:
                                data[key] = result[0]
                            else:
                                data[key] = result
                        else:
                            data[key] = {}
                    return data
                except Exception as e:
                    st.error(f"Financial Modeling Prep API lỗi: {e}")
                return None

                financial_api_key = "UmQvIg3EKUstRE9UuUiYBIroY7fgqJIR"
                # Phân tích intent: tìm mã công ty, năm, các chỉ số cần lấy
                match = re.search(r"(\w+).*?(\d{4})", user_query)
                company_intent = company
                year_intent = None
                if match:
                    company_intent = match.group(1)
                    year_intent = match.group(2)

                # Intent detection: nếu chứa các từ khoá tin tức thì gọi Tavily, ngược lại gọi API số liệu
                # Intent detection nâng cao
                news_keywords = ["tin tức", "news", "báo", "thông tin", "sự kiện", "update", "latest"]
                price_keywords = ["giá", "price", "realtime", "quote", "stock", "cổ phiếu"]
                finnhub_keywords = ["insider", "earnings", "financial news", "tin tức tài chính"]
                ratio_keywords = ["currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding", "netProfitMargin", "pretaxProfitMargin", "grossProfitMargin", "operatingProfitMargin", "returnOnAssets", "returnOnCapitalEmployed", "returnOnEquity", "assetTurnover", "fixedAssetTurnover", "debtEquityRatio", "debtRatio", "effectiveTaxRate", "freeCashFlowOperatingCashFlowRatio", "freeCashFlowPerShare", "cashPerShare", "companyEquityMultiplier", "ebitPerRevenue", "enterpriseValueMultiple", "operatingCashFlowPerShare", "operatingCashFlowSalesRatio", "payablesTurnover"]

                financial_context = ""
                ratio_results = {}
                context_snippets = []
                # Tin tức: Tavily
                if any(kw in user_query.lower() for kw in news_keywords):
                    try:
                        tavily = TavilyClient(api_key=settings.tavily_api_key)
                        q = user_query
                        res = tavily.search(q, search_depth="basic", max_results=num_results)
                        if isinstance(res, dict):
                            for item in (res.get("results") or [])[:num_results]:
                                snippet = item.get("content") or item.get("snippet") or ""
                                if snippet:
                                    context_snippets.append(snippet)
                    except Exception as e:
                        st.error(f"Tavily lỗi: {e}")
                    financial_context += "\n\n[NGỮ CẢNH TIN TỨC]\n" + "\n".join(context_snippets)
                # Giá & realtime: AlphaVantage
                elif any(kw in user_query.lower() for kw in price_keywords):
                    price_data = get_alphavantage_price(company_intent)
                    if price_data:
                        financial_context += "\n\n[GIÁ CỔ PHIẾU - ALPHAVANTAGE]\n" + str(price_data)
                # Tin tức tài chính, insider, earnings: Finnhub
                elif any(kw in user_query.lower() for kw in finnhub_keywords):
                    news_data = get_finnhub_news(company_intent)
                    if news_data:
                        financial_context += "\n\n[FINNHUB NEWS]\n" + str(news_data)
                # Financial ratios & statements: FMP
                elif any(kw.lower() in user_query.lower() for kw in ratio_keywords):
                    requested_ratios = [r for r in ratio_keywords if r.lower() in user_query.lower()]
                    financial_data = get_financial_data(company_intent, financial_api_key, year_intent)
                    bs = financial_data.get("balance_sheet", {}) if financial_data else {}
                    is_ = financial_data.get("income_statement", {}) if financial_data else {}
                    cf = financial_data.get("cash_flow", {}) if financial_data else {}
                    for r in requested_ratios:
                        if r == "currentRatio":
                            ratio_results[r] = current_ratio(bs)
                        elif r == "quickRatio":
                            ratio_results[r] = quick_ratio(bs)
                        elif r == "cashRatio":
                            ratio_results[r] = cash_ratio(bs)
                        elif r == "daysOfSalesOutstanding":
                            ratio_results[r] = days_of_sales_outstanding(bs, is_)
                        elif r == "netProfitMargin":
                            ratio_results[r] = net_profit_margin(is_)
                        elif r == "pretaxProfitMargin":
                            ratio_results[r] = pretax_profit_margin(is_)
                        elif r == "grossProfitMargin":
                            ratio_results[r] = gross_profit_margin(is_)
                        elif r == "operatingProfitMargin":
                            ratio_results[r] = operating_profit_margin(is_)
                        elif r == "returnOnAssets":
                            ratio_results[r] = return_on_assets(is_, bs)
                        elif r == "returnOnCapitalEmployed":
                            ratio_results[r] = return_on_capital_employed(is_, bs)
                        elif r == "returnOnEquity":
                            ratio_results[r] = return_on_equity(is_, bs)
                        elif r == "assetTurnover":
                            ratio_results[r] = asset_turnover(is_, bs)
                        elif r == "fixedAssetTurnover":
                            ratio_results[r] = fixed_asset_turnover(is_, bs)
                        elif r == "debtEquityRatio":
                            ratio_results[r] = debt_equity_ratio(bs)
                        elif r == "debtRatio":
                            ratio_results[r] = debt_ratio(bs)
                        elif r == "companyEquityMultiplier":
                            ratio_results[r] = company_equity_multiplier(bs)
                        elif r == "effectiveTaxRate":
                            ratio_results[r] = effective_tax_rate(is_)
                        elif r == "freeCashFlowOperatingCashFlowRatio":
                            ratio_results[r] = free_cash_flow_operating_cash_flow_ratio(cf)
                        elif r == "freeCashFlowPerShare":
                            ratio_results[r] = free_cash_flow_per_share(cf, bs)
                        elif r == "operatingCashFlowPerShare":
                            ratio_results[r] = operating_cash_flow_per_share(cf, bs)
                        elif r == "operatingCashFlowSalesRatio":
                            ratio_results[r] = operating_cash_flow_sales_ratio(cf, is_)
                        elif r == "cashPerShare":
                            ratio_results[r] = cash_per_share(bs)
                        elif r == "ebitPerRevenue":
                            ratio_results[r] = ebit_per_revenue(is_)
                        elif r == "enterpriseValueMultiple":
                            ratio_results[r] = enterprise_value_multiple(bs, is_)
                        elif r == "payablesTurnover":
                            ratio_results[r] = payables_turnover(bs, is_)
                    if ratio_results:
                        import pandas as pd
                        df_ratios = pd.DataFrame(list(ratio_results.items()), columns=["Chỉ số", "Giá trị"])
                        financial_context += "Các chỉ số tài chính yêu cầu:\n" + df_ratios.to_markdown(index=False) + "\n"
                    financial_context += "\nBáo cáo tài chính gốc:\n" + str(financial_data)
            # Lưu câu hỏi vào lịch sử
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            context_snippets = []
            if use_tavily:
                if not settings.tavily_api_key:
                    st.warning("Chưa có TAVILY_API_KEY trong môi trường. Bỏ qua tìm kiếm.")
                else:
                    try:
                        tavily = TavilyClient(api_key=settings.tavily_api_key)
                        q = user_query
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
                    # Đảm bảo financial_context luôn tồn tại
                    if 'financial_context' not in locals():
                        financial_context = ""
                    genai.configure(api_key=settings.gemini_api_key)
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    system_prompt = (
                        "Bạn là trợ lý tài chính. Sử dụng ngữ cảnh web nếu có, "
                        "và trả lời ngắn gọn, có cấu trúc. Nếu câu hỏi liên quan dự đoán rating, "
                        "hãy gợi ý dùng phần Predict để có con số từ mô hình nội bộ."
                    )
                    context_text = "\n\n".join(context_snippets) if context_snippets else ""
                    # Kết hợp thêm dữ liệu từ Financial Statement API vào ngữ cảnh
                    if financial_context:
                        context_text += f"\n\n[NGỮ CẢNH BÁO CÁO TÀI CHÍNH]\n{financial_context}"
                    full_prompt = (
                        f"[HỆ THỐNG]\n{system_prompt}\n\n"
                        f"[NGỮ CẢNH WEB]\n{context_text}\n\n"
                        f"[CÂU HỎI]\n{user_query}\n"
                    )
                    resp = model.generate_content(full_prompt)
                    bot_reply = resp.text or "(Không có nội dung)"
                    st.session_state["chat_history"].append({"role": "bot", "content": bot_reply})
                    st.markdown(f"<div style='text-align:left; color:green'><b>Bot:</b> {bot_reply}</div>", unsafe_allow_html=True)
                    if context_text:
                        with st.expander("Nguồn ngữ cảnh (Tavily + Báo cáo tài chính)"):
                            st.write(context_text)
                except Exception as e:
                    st.error(f"Gemini lỗi: {e}")

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