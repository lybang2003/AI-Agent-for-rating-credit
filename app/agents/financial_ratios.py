def safe_div(a, b):
    try:
        return round(float(a) / float(b), 3) if a and b and float(b) != 0 else None
    except:
        return None

# Nhóm thanh khoản
def current_ratio(bs):
    return safe_div(bs.get("totalCurrentAssets"), bs.get("totalCurrentLiabilities"))

def quick_ratio(bs):
    return safe_div(float(bs.get("totalCurrentAssets", 0)) - float(bs.get("inventory", 0)), bs.get("totalCurrentLiabilities"))

def cash_ratio(bs):
    return safe_div(bs.get("cashAndCashEquivalents"), bs.get("totalCurrentLiabilities"))

# Nhóm quản lý khoản phải thu
def days_of_sales_outstanding(bs, is_):
    dso = safe_div(bs.get("accountsReceivable"), is_.get("revenue"))
    return round(dso * 365, 3) if dso else None

# Nhóm lợi nhuận
def net_profit_margin(is_):
    return safe_div(is_.get("netIncome"), is_.get("revenue"))

def pretax_profit_margin(is_):
    return safe_div(is_.get("incomeBeforeTax"), is_.get("revenue"))

def gross_profit_margin(is_):
    return safe_div(is_.get("grossProfit"), is_.get("revenue"))

def operating_profit_margin(is_):
    return safe_div(is_.get("operatingIncome"), is_.get("revenue"))

# Nhóm hiệu quả sử dụng vốn
def return_on_assets(is_, bs):
    return safe_div(is_.get("netIncome"), bs.get("totalAssets"))

def return_on_capital_employed(is_, bs):
    return safe_div(is_.get("ebit"), float(bs.get("totalAssets", 0)) - float(bs.get("totalCurrentLiabilities", 0)))

def return_on_equity(is_, bs):
    return safe_div(is_.get("netIncome"), bs.get("totalStockholdersEquity"))

def asset_turnover(is_, bs):
    return safe_div(is_.get("revenue"), bs.get("totalAssets"))

def fixed_asset_turnover(is_, bs):
    return safe_div(is_.get("revenue"), bs.get("netPPE"))

# Nhóm nợ & đòn bẩy
def debt_equity_ratio(bs):
    return safe_div(bs.get("totalDebt"), bs.get("totalStockholdersEquity"))

def debt_ratio(bs):
    return safe_div(bs.get("totalDebt"), bs.get("totalAssets"))

def company_equity_multiplier(bs):
    return safe_div(bs.get("totalAssets"), bs.get("totalStockholdersEquity"))

# Thuế
def effective_tax_rate(is_):
    return safe_div(is_.get("incomeTaxExpense"), is_.get("incomeBeforeTax"))

# Dòng tiền
def free_cash_flow_operating_cash_flow_ratio(cf):
    return safe_div(cf.get("freeCashFlow"), cf.get("operatingCashFlow"))

def free_cash_flow_per_share(cf, bs):
    return safe_div(cf.get("freeCashFlow"), bs.get("weightedAverageShsOut"))

def operating_cash_flow_per_share(cf, bs):
    return safe_div(cf.get("operatingCashFlow"), bs.get("weightedAverageShsOut"))

def operating_cash_flow_sales_ratio(cf, is_):
    return safe_div(cf.get("operatingCashFlow"), is_.get("revenue"))

def cash_per_share(bs):
    return safe_div(bs.get("cashAndCashEquivalents"), bs.get("weightedAverageShsOut"))

# Định giá
def ebit_per_revenue(is_):
    return safe_div(is_.get("ebit"), is_.get("revenue"))

def enterprise_value_multiple(bs, is_):
    return safe_div(bs.get("enterpriseValue"), is_.get("ebitda"))

# Quản lý khoản phải trả
def payables_turnover(bs, is_):
    purchases = bs.get("purchases") or is_.get("costOfGoodsSold")
    return safe_div(purchases, bs.get("accountsPayable"))
