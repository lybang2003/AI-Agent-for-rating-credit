import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_rating_distribution(df):
    fig = px.bar(df['rating'].value_counts().reset_index(), x='index', y='rating',
                 labels={'index': 'Rating', 'rating': 'Số lượng'}, title='Phân phối xếp hạng tín dụng')
    return fig

def plot_sector_rating_grouped(df):
    grouped = df.groupby(['sector', 'rating']).size().reset_index(name='count')
    fig = px.bar(grouped, x='sector', y='count', color='rating', barmode='group',
                 title='Phân bổ Rating theo từng Sector')
    return fig

def plot_financial_boxplot(df, metric):
    fig = px.box(df, x='sector', y=metric, points='all', title=f'Phân phối {metric} theo ngành')
    return fig

def plot_financial_histogram(df, metric):
    fig = px.histogram(df, x=metric, nbins=30, title=f'Phân phối {metric}')
    return fig

def plot_performance_scatter(df):
    fig = px.scatter(df, x='debtEquityRatio', y='ROE', color='sector',
                     title='ROE vs DebtEquityRatio (Đòn bẩy vs Hiệu quả)')
    return fig

def plot_performance_bubble(df):
    fig = px.scatter(df, x='debtRatio', y='ROE', size='revenue', color='sector',
                     hover_name='company', title='ROE vs DebtRatio, size=Revenue')
    return fig

def plot_trend_line(df, metric):
    fig = px.line(df, x='year', y=metric, color='company', title=f'Trends of {metric} theo năm')
    return fig

def plot_correlation_heatmap(df):
    corr = df.select_dtypes(include='number').corr()
    fig = px.imshow(corr, text_auto=True, title='Ma trận tương quan các chỉ số tài chính')
    return fig

# ==========================
# Tiện ích lấy ngữ cảnh từ file dữ liệu
# ==========================
import os
def get_credit_context(csv_path, company=None, year=None):
    """
    Trích xuất ngữ cảnh (thông tin) từ file corporateCreditRatingWithFinancialRatios.csv hoặc corporate_rating.csv
    """
    import pandas as pd
    if not os.path.exists(csv_path):
        return "Không tìm thấy file dữ liệu!"
    df = pd.read_csv(csv_path)
    context = ""
    if company:
        df = df[df['Corporation'].str.lower() == company.lower()]
    if year and 'Rating Date' in df.columns:
        df = df[df['Rating Date'].str.startswith(str(year))]
    if df.empty:
        return "Không tìm thấy dữ liệu phù hợp!"
    # Lấy thông tin đầu tiên
    row = df.iloc[0]
    context += f"Công ty: {row.get('Corporation', '')}\n"
    context += f"Ngành: {row.get('Sector', '')}\n"
    context += f"Rating: {row.get('Rating', '')} ({row.get('Rating Agency', '')})\n"
    context += f"Ngày rating: {row.get('Rating Date', '')}\n"
    context += f"Ticker: {row.get('Ticker', '')}\n"
    # Thêm một số chỉ số tài chính
    for col in ['Current Ratio', 'Debt/Equity Ratio', 'Net Profit Margin', 'ROE - Return On Equity', 'Gross Margin']:
        if col in df.columns:
            context += f"{col}: {row.get(col, '')}\n"
    return context

