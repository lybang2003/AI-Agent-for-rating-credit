import os
import json
import pandas as pd
import plotly.graph_objects as go
from app.agents.plotly_charts import (
    plot_stock_candlestick, plot_stock_line,
    plot_revenue_profit, plot_debt_equity, plot_cashflow,
    plot_financial_radar, plot_revenue_treemap, plot_balance_sunburst,
    plot_company_bubble, plot_ratio_boxplot, plot_debt_histogram, plot_stock_animated
)
import streamlit as st
from app.agents.credit_charts import (
    plot_rating_distribution, plot_sector_rating_grouped, plot_financial_boxplot,
    plot_financial_histogram, plot_performance_scatter, plot_performance_bubble,
    plot_trend_line, plot_correlation_heatmap
)
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


    # Tự động nạp ETL khi khởi động
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
            # Trợ lý vẽ biểu đồ: nếu ngữ cảnh có đủ dữ liệu, tự động vẽ và trả lời
            def try_plot_from_context(user_query, company_intent, year_intent):
                import pandas as pd
                csv_path = "d:\LLM\corporateCreditRatingWithFinancialRatios.csv"
                if not os.path.exists(csv_path):
                    return None, "Không tìm thấy file dữ liệu!"
                df = pd.read_csv(csv_path)
                # Nếu có năm, lọc theo năm
                if year_intent and 'Rating Date' in df.columns:
                    df = df[df['Rating Date'].str.startswith(str(year_intent))]
                # Nếu có công ty, lọc theo công ty
                if company_intent:
                    df = df[df['Corporation'].str.lower() == company_intent.lower()]
                # Ví dụ: scatter plot ROE vs DebtEquityRatio
                if 'ROE - Return On Equity' in df.columns and 'Debt/Equity Ratio' in df.columns and not df.empty:
                    import plotly.express as px
                    fig = px.scatter(df, x='Debt/Equity Ratio', y='ROE - Return On Equity', color='Rating', hover_name='Corporation', title='ROE vs Debt/Equity Ratio')
                    return fig, None
                return None, "Dữ liệu về ROE (Return on Equity) và Debt/Equity Ratio không có trong ngữ cảnh nội bộ được cung cấp."
            # Phân tích intent lấy mã công ty, năm nếu có
            import re
            match = re.search(r"(\w+).*?(\d{4})", user_query)
            company_intent = company
            year_intent = None
            if match:
                company_intent = match.group(1)
                year_intent = match.group(2)
            from app.agents.credit_charts import get_credit_context
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
                chart_fig = None
                context_text = ""
                try:
                    # Trợ lý vẽ biểu đồ: thử vẽ từ dữ liệu nội bộ
                    chart_fig, plot_error = try_plot_from_context(user_query, company_intent, year_intent)
                    if chart_fig:
                        st.plotly_chart(chart_fig, use_container_width=True)
                    elif plot_error:
                        st.info(plot_error + "\nBạn có thể bổ sung dữ liệu hoặc upload file mới để vẽ biểu đồ.")
                    # Tiếp tục lấy ngữ cảnh như cũ
                    year_int = year_intent if year_intent else None
                    csv_path = "d:\LLM\corporateCreditRatingWithFinancialRatios.csv"
                    context_text = get_credit_context(csv_path, company_intent, year_int)
                    if "Không tìm thấy dữ liệu phù hợp" in context_text or "Không tìm thấy file dữ liệu" in context_text:
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
                        context_text = "\n".join(context_snippets)
                except Exception as e:
                    st.error(f"Lỗi vẽ biểu đồ/ngữ cảnh tự động: {e}")

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
                    # Vẽ biểu đồ giá cổ phiếu
                    try:
                        alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
                        fig_line = plot_stock_line(company_intent, alphavantage_api_key)
                        st.plotly_chart(fig_line, use_container_width=True)
                        fig_candle = plot_stock_candlestick(company_intent, alphavantage_api_key)
                        st.plotly_chart(fig_candle, use_container_width=True)
                        # Có thể thêm volume chart nếu cần
                    except Exception as e:
                        st.error(f"Lỗi vẽ biểu đồ giá cổ phiếu: {e}")
                # Tin tức tài chính, insider, earnings: Finnhub
                elif any(kw in user_query.lower() for kw in finnhub_keywords):
                    news_data = get_finnhub_news(company_intent)
                    if news_data:
                        financial_context += "\n\n[FINNHUB NEWS]\n" + str(news_data)
                # Financial ratios & statements: FMP
                elif any(kw.lower() in user_query.lower() for kw in ratio_keywords):
                    try:
                        financial_api_key = os.getenv("FINANCIAL_API_KEY")
                        fig_revenue = plot_revenue_profit(company_intent, financial_api_key)
                        st.plotly_chart(fig_revenue, use_container_width=True)
                        fig_debt = plot_debt_equity(company_intent, financial_api_key)
                        st.plotly_chart(fig_debt, use_container_width=True)
                        fig_cashflow = plot_cashflow(company_intent, financial_api_key)
                        st.plotly_chart(fig_cashflow, use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi vẽ biểu đồ tài chính: {e}")
            # Lưu câu hỏi vào lịch sử
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            # Ưu tiên lấy ngữ cảnh từ data CSV
            year_int = year_intent if 'year_intent' in locals() and year_intent else None
            csv_path = "d:\LLM\corporateCreditRatingWithFinancialRatios.csv"
            context_text = get_credit_context(csv_path, company_intent, year_int)
            # Nếu không có dữ liệu phù hợp, fallback sang web
            if "Không tìm thấy dữ liệu phù hợp" in context_text or "Không tìm thấy file dữ liệu" in context_text:
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
                context_text = "\n".join(context_snippets)

            if not settings.gemini_api_key:
                st.error("Chưa cấu hình GEMINI_API_KEY. Đặt trong .env hoặc môi trường rồi thử lại.")
            else:
                try:
                    genai.configure(api_key=settings.gemini_api_key)
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    system_prompt = (
                        "Bạn là trợ lý tài chính. Sử dụng ngữ cảnh dữ liệu nội bộ nếu có, nếu không thì dùng ngữ cảnh web. "
                        "Trả lời ngắn gọn, có cấu trúc. Nếu câu hỏi liên quan dự đoán rating, "
                        "hãy gợi ý dùng phần Predict để có con số từ mô hình nội bộ."
                    )
                    # Kết hợp thêm dữ liệu từ Financial Statement API vào ngữ cảnh
                    full_prompt = (
                        f"[HỆ THỐNG]\n{system_prompt}\n\n"
                        f"[NGỮ CẢNH]\n{context_text}\n\n"
                        f"[CÂU HỎI]\n{user_query}\n"
                    )
                    resp = model.generate_content(full_prompt)
                    bot_reply = resp.text or "(Không có nội dung)"
                    st.session_state["chat_history"].append({"role": "bot", "content": bot_reply})
                    st.markdown(f"<div style='text-align:left; color:green'><b>Bot:</b> {bot_reply}</div>", unsafe_allow_html=True)
                    if context_text:
                        with st.expander("Nguồn ngữ cảnh (Data nội bộ/ Web)"):
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
    tab_tools = st.tabs(["Query CSV", "Query Financial Metrics"])

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


        with tab_tools[1]:  # Query Financial Metrics
            st.subheader("Truy vấn chỉ số từ data nội bộ")
            query_company = st.text_input("Tên công ty cần truy vấn", value="American States Water Co.")
            query_date = st.text_input("Ngày cần truy vấn (YYYY-MM-DD)", value="2010-07-30")
            query_metrics = st.text_input("Các chỉ số cần truy vấn (phân cách bằng dấu phẩy)", value="Gross Margin,Operating Margin")
            if st.button("Truy vấn chỉ số"):
                import pandas as pd
                # Đọc cả 2 file CSV
                dfs = []
                for path in ["corporateCreditRatingWithFinancialRatios.csv", "corporate_rating.csv"]:
                    if os.path.exists(path):
                        try:
                            df = pd.read_csv(path)
                            dfs.append(df)
                        except Exception as e:
                            st.error(f"Lỗi đọc {path}: {e}")
                # Gộp data
                if dfs:
                    df_all = pd.concat(dfs, ignore_index=True)
                    # Chuẩn hóa tên cột công ty
                    company_cols = [c for c in df_all.columns if c.lower() in ["corporation", "name"]]
                    date_cols = [c for c in df_all.columns if "date" in c.lower()]
                    # Lọc theo công ty
                    if company_cols:
                        df_all = df_all[df_all[company_cols[0]].str.lower() == query_company.strip().lower()]
                    # Lọc theo ngày
                    if date_cols:
                        df_all = df_all[df_all[date_cols[0]].astype(str).str.startswith(query_date.strip())]
                    # Lấy các chỉ số cần truy vấn
                    metrics = [m.strip() for m in query_metrics.split(",") if m.strip()]
                    available_metrics = [m for m in metrics if m in df_all.columns]
                    if not available_metrics:
                        st.warning("Không tìm thấy chỉ số nào trong data!")
                    else:
                        st.dataframe(df_all[[company_cols[0], date_cols[0]] + available_metrics])
            else:
                st.warning("Không tìm thấy file dữ liệu nào!")