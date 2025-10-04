import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================
# 1. Giá cổ phiếu (Alpha Vantage)
# ==========================
def plot_stock_candlestick(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json().get("Time Series (Daily)", {})
    df = pd.DataFrame(data).T
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"]
    )])
    fig.update_layout(title=f"Biểu đồ nến {symbol}", xaxis_rangeslider_visible=False)
    return fig


def plot_stock_line(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json().get("Time Series (Daily)", {})
    df = pd.DataFrame(data).T
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)

    fig = px.line(df, x=df.index, y="close", title=f"Giá đóng cửa {symbol}")
    return fig


# ==========================
# 2. Tài chính cơ bản (FMP API)
# ==========================
def plot_revenue_profit(symbol, api_key):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=5&apikey={api_key}"
    data = requests.get(url).json()
    df = pd.DataFrame(data)[["date", "revenue", "netIncome"]].sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["revenue"], name="Revenue"))
    fig.add_trace(go.Bar(x=df["date"], y=df["netIncome"], name="Net Income"))
    fig.update_layout(title=f"Revenue & Net Income - {symbol}", barmode="group")
    return fig


def plot_debt_equity(symbol, api_key):
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=5&apikey={api_key}"
    data = requests.get(url).json()
    df = pd.DataFrame(data)[["date", "totalDebt", "totalStockholdersEquity"]].sort_values("date")

    fig = px.bar(df, x="date", y=["totalDebt", "totalStockholdersEquity"],
                 barmode="group", title=f"Debt vs Equity - {symbol}")
    return fig


def plot_cashflow(symbol, api_key):
    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?limit=5&apikey={api_key}"
    data = requests.get(url).json()
    df = pd.DataFrame(data)[["date", "operatingCashFlow", "investingCashFlow", "financingCashFlow"]].sort_values("date")

    fig = go.Figure()
    for col in ["operatingCashFlow", "investingCashFlow", "financingCashFlow"]:
        fig.add_trace(go.Bar(x=df["date"], y=df[col], name=col))
    fig.update_layout(title=f"Cash Flow Breakdown - {symbol}", barmode="stack")
    return fig


# ==========================
# 3. Biểu đồ nâng cao
# ==========================
def plot_financial_radar(df, metrics, company_col="company", title="So sánh chỉ số tài chính"):
    companies = df[company_col].unique()
    fig = go.Figure()
    for company in companies:
        subset = df[df[company_col] == company]
        values = [subset[m].values[0] if m in subset else 0 for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=company
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=title, showlegend=True)
    return fig


def plot_revenue_treemap(df, title="Cơ cấu doanh thu theo mảng"):
    fig = px.treemap(df, path=["segment"], values="revenue", title=title)
    return fig


def plot_balance_sunburst(df, title="Cơ cấu tài sản - nợ"):
    fig = px.sunburst(df, path=["category", "subcategory"], values="amount", title=title)
    return fig


def plot_company_bubble(df, title="So sánh doanh nghiệp"):
    fig = px.scatter(df, x="assets", y="revenue", size="market_cap", color="sector",
                     hover_name="company", title=title, size_max=60)
    return fig


def plot_ratio_boxplot(df, ratio="ROE", title="Phân phối ROE theo ngành"):
    fig = px.box(df, x="sector", y=ratio, points="all", title=title)
    return fig


def plot_debt_histogram(df, title="Phân bố Debt Ratio"):
    fig = px.histogram(df, x="debt_ratio", nbins=30, title=title)
    return fig


def plot_stock_animated(df, title="Biến động giá cổ phiếu"):
    fig = px.line(df, x="date", y="close", color="company", title=title,
                  animation_frame="date", animation_group="company")
    return fig

# ==========================
# NLP intent → chart function mapping
# ==========================
def nlp_map_intent_to_chart(intent: str):
    """
    Map intent string to chart function name.
    """
    import re
    CHART_INTENTS = {
        "plot_stock_candlestick": ["biểu đồ nến", "candlestick", "giá cổ phiếu", "stock price", "ohlc"],
        "plot_stock_line": ["xu hướng", "trend", "line chart", "biểu đồ đường", "giá đóng cửa"],
        "plot_revenue_profit": ["so sánh", "bar chart", "biểu đồ cột", "doanh thu", "lợi nhuận", "revenue", "net income"],
        "plot_cashflow": ["dòng tiền", "cash flow", "cashflow", "waterfall"],
        "plot_debt_equity": ["nợ", "debt", "debt to equity", "đòn bẩy", "vốn chủ"],
        "plot_financial_radar": ["radar", "spider", "đa chiều", "so sánh nhiều chỉ số"],
        "plot_revenue_treemap": ["treemap", "cơ cấu doanh thu", "segment"],
        "plot_balance_sunburst": ["sunburst", "cơ cấu tài sản", "category", "subcategory"],
        "plot_company_bubble": ["bubble", "so sánh doanh nghiệp", "assets", "market cap"],
        "plot_ratio_boxplot": ["boxplot", "phân phối", "roe", "return on equity"],
        "plot_debt_histogram": ["histogram", "phân bố nợ", "debt ratio"],
        "plot_stock_animated": ["animated", "biến động giá", "animation"],
    }
    intent = intent.lower()
    for func_name, keywords in CHART_INTENTS.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", intent):
                return func_name
    return "plot_stock_line"  # default

# --- Router chọn hàm vẽ ---
def chart_router(user_query, data, api_key=None):
    func_name = nlp_map_intent_to_chart(user_query)
    chart_func = globals().get(func_name)
    if chart_func:
        # Nếu hàm cần symbol và api_key
        import inspect
        params = inspect.signature(chart_func).parameters
        if "symbol" in params and "api_key" in params:
            return chart_func(data, api_key)
        else:
            return chart_func(data)
    else:
        print(f"⚡ Chưa có hàm vẽ cho intent {func_name}, fallback sang plot_stock_line")
        return plot_stock_line(data, api_key) if api_key else None
