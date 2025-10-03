from __future__ import annotations

from typing import List


# Danh sách >= 30 prompt template cho tài chính/kế toán/doanh nghiệp
PROMPTS: List[str] = [
    "Lấy Current Ratio của {company} theo từng năm.",
    "Lấy Debt to Equity của {company} theo từng năm.",
    "Trích xuất Gross Margin của {company} theo từng năm.",
    "Trích xuất Operating Margin của {company} theo từng năm.",
    "Lấy ROA (Return on Assets) của {company} theo từng năm.",
    "Lấy ROE (Return on Equity) của {company} theo từng năm.",
    "Lấy EBIT Margin của {company} theo từng năm.",
    "Lấy EBITDA Margin của {company} theo từng năm.",
    "Lấy Net Profit Margin của {company} theo từng năm.",
    "Lấy Asset Turnover của {company} theo từng năm.",
    "Lấy Operating Cash Flow Per Share của {company} theo từng năm.",
    "Lấy Free Cash Flow Per Share của {company} theo từng năm.",
    "Thu thập Pre-Tax Profit Margin của {company} theo năm.",
    "Truy vấn Return on Investment của {company} theo năm.",
    "Tìm Long Term Debt to Capital của {company} theo năm.",
    "Tính tỷ lệ nợ/vốn chủ (Debt/Equity) của {company} theo năm.",
    "Lấy Quick Ratio của {company} theo năm nếu có.",
    "Lấy Interest Coverage Ratio của {company} theo năm nếu có.",
    "Lấy Total Debt/EBITDA của {company} theo năm nếu có.",
    "Tổng hợp Doanh thu (Revenue) của {company} theo năm.",
    "Tổng hợp Lợi nhuận ròng (Net Income) của {company} theo năm.",
    "Tổng hợp Tổng tài sản (Total Assets) của {company} theo năm.",
    "Tổng hợp Vốn chủ (Shareholders' Equity) của {company} theo năm.",
    "Xuất timeseries cổ tức (Dividend per Share) của {company} theo năm nếu có.",
    "Xuất timeseries EPS của {company} theo năm nếu có.",
    "Lấy CAPEX của {company} theo năm nếu có.",
    "Lấy Working Capital của {company} theo năm nếu có.",
    "Tìm Interest Expense của {company} theo năm.",
    "Tìm Cash and Equivalents của {company} theo năm.",
    "Tìm Total Liabilities của {company} theo năm.",
    "Truy vấn Market Cap của {company} theo năm (nếu có dữ liệu).",
    "Tìm Enterprise Value của {company} theo năm (nếu có).",
    "Tổng hợp số liệu tài chính quan trọng của {company} theo năm.",
]


def get_prompts() -> List[str]:
    return PROMPTS


