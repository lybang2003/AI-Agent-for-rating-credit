import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================
# Biểu đồ nâng cao từ DataFrame (CSV)
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
