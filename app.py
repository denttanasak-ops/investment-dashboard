from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Investment Dashboard", layout="wide")
st.title("Investment Intelligence Dashboard")

BASE_CURRENCY = "THB"
GOOGLE_SHEET_ID = "1NfxJUlUyFmeSFjFCNLF7Xoeuu_vP_dfk9Hi2HP7Yl_c"

SHEET_TABS = {
    "portfolio": "portfolio",
    "bank_accounts": "bank_accounts",
    "properties": "properties",
    "mortgage": "mortgage",
    "property_cashflow": "property_cashflow",
}

DEFAULT_MARKET_ASSETS = {
    "SPY": "US Market",
    "QQQ": "US Tech / AI",
    "SOXX": "Semiconductor",
    "XLV": "Healthcare",
    "ITA": "Aerospace",
    "XLE": "Energy",
    "BRK-B": "Berkshire Hathaway",
    "INDA": "India",
    "MCHI": "China",
    "THD": "Thailand ETF",
    "GLD": "Gold",
    "BTC-USD": "Bitcoin",
}

PORTFOLIO_COLUMNS = {
    "Broker": "",
    "Ticker": "",
    "Currency": "THB",
    "Quantity": 0.0,
    "AvgCost": 0.0,
    "ManualPrice": 0.0,
}
BANK_COLUMNS = {"Bank": "", "Account": "", "Balance": 0.0, "Currency": "THB"}
PROPERTY_COLUMNS = {"Property": "", "Type": "", "EstimatedValue": 0.0, "Location": ""}
MORTGAGE_COLUMNS = {"Property": "", "OutstandingDebt": 0.0, "MonthlyPayment": 0.0, "InterestRate": 0.0}
CASHFLOW_COLUMNS = {"Property": "", "Month": "", "Rent": 0.0, "Expense": 0.0, "ExtraPayment": 0.0}


def clean_ticker(ticker):
    if pd.isna(ticker):
        return ""
    return str(ticker).strip().upper()


def clean_currency(currency):
    if pd.isna(currency) or str(currency).strip() == "":
        return BASE_CURRENCY
    return str(currency).strip().upper()


def to_number(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def format_thb(value):
    return f"{value:,.2f} THB"


def get_asset_name(ticker):
    ticker = clean_ticker(ticker)
    return DEFAULT_MARKET_ASSETS.get(ticker, ticker)


def ensure_columns(df, columns):
    df = df.copy()
    for col, default in columns.items():
        if col not in df.columns:
            df[col] = default
    return df[list(columns.keys())]


@st.cache_data(ttl=60)
def read_google_sheet_tab(sheet_name):
    encoded_sheet = quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={encoded_sheet}"
    return pd.read_csv(url)


@st.cache_data(ttl=300)
def download_prices(tickers, start=None, end=None, period=None):
    tickers = [clean_ticker(t) for t in tickers if clean_ticker(t)]
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"] if "Close" in data.columns.get_level_values(0) else data.xs("Close", level=1, axis=1)
    else:
        close = data["Close"] if "Close" in data.columns else data
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.columns = [clean_ticker(c) for c in close.columns]
    return close


def get_current_prices(tickers):
    tickers = [clean_ticker(t) for t in tickers if clean_ticker(t) and clean_ticker(t) != "CASH"]
    if not tickers:
        return pd.Series(dtype=float)
    prices = download_prices(tickers, period="7d")
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.ffill().iloc[-1]


@st.cache_data(ttl=300)
def get_usdthb_rate():
    prices = download_prices(["USDTHB=X"], period="7d")
    if prices.empty or "USDTHB=X" not in prices.columns:
        return 32.43
    rate = float(prices["USDTHB=X"].ffill().iloc[-1])
    return rate if rate > 0 else 32.43


def fx_to_thb(currency):
    currency = clean_currency(currency)
    if currency == "THB":
        return 1.0
    if currency == "USD":
        return get_usdthb_rate()
    return 1.0


def load_portfolio():
    df = ensure_columns(read_google_sheet_tab(SHEET_TABS["portfolio"]), PORTFOLIO_COLUMNS)
    df["Broker"] = df["Broker"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].apply(clean_ticker)
    df["Currency"] = df["Currency"].apply(clean_currency)
    df["Quantity"] = to_number(df["Quantity"])
    df["AvgCost"] = to_number(df["AvgCost"])
    df["ManualPrice"] = to_number(df["ManualPrice"])
    return df


def load_banks():
    df = ensure_columns(read_google_sheet_tab(SHEET_TABS["bank_accounts"]), BANK_COLUMNS)
    df["Currency"] = df["Currency"].apply(clean_currency)
    df["Balance"] = to_number(df["Balance"])
    df["FxRateToTHB"] = df["Currency"].apply(fx_to_thb)
    df["BalanceTHB"] = df["Balance"] * df["FxRateToTHB"]
    return df


def load_properties():
    df = ensure_columns(read_google_sheet_tab(SHEET_TABS["properties"]), PROPERTY_COLUMNS)
    df["EstimatedValue"] = to_number(df["EstimatedValue"])
    return df


def load_mortgage():
    df = ensure_columns(read_google_sheet_tab(SHEET_TABS["mortgage"]), MORTGAGE_COLUMNS)
    df["OutstandingDebt"] = to_number(df["OutstandingDebt"])
    df["MonthlyPayment"] = to_number(df["MonthlyPayment"])
    df["InterestRate"] = to_number(df["InterestRate"])
    return df


def load_property_cashflow():
    df = ensure_columns(read_google_sheet_tab(SHEET_TABS["property_cashflow"]), CASHFLOW_COLUMNS)
    df["Rent"] = to_number(df["Rent"])
    df["Expense"] = to_number(df["Expense"])
    df["ExtraPayment"] = to_number(df["ExtraPayment"])
    return df


def calculate_portfolio(portfolio):
    portfolio = portfolio.copy()
    portfolio["Ticker"] = portfolio["Ticker"].apply(clean_ticker)
    portfolio = portfolio[portfolio["Ticker"] != ""]
    portfolio["Currency"] = portfolio["Currency"].apply(clean_currency)
    portfolio["Quantity"] = to_number(portfolio["Quantity"])
    portfolio["AvgCost"] = to_number(portfolio["AvgCost"])
    portfolio["ManualPrice"] = to_number(portfolio["ManualPrice"])

    tickers = portfolio.loc[portfolio["Ticker"] != "CASH", "Ticker"].dropna().unique().tolist()
    current_prices = get_current_prices(tickers)

    portfolio["YFinancePrice"] = portfolio["Ticker"].map(current_prices)
    portfolio.loc[portfolio["Ticker"] == "CASH", "YFinancePrice"] = 1
    portfolio["YFinancePrice"] = to_number(portfolio["YFinancePrice"])
    portfolio["CurrentPrice"] = portfolio["YFinancePrice"]

    use_manual = (portfolio["CurrentPrice"] == 0) & (portfolio["ManualPrice"] > 0)
    portfolio.loc[use_manual, "CurrentPrice"] = portfolio.loc[use_manual, "ManualPrice"]
    portfolio.loc[portfolio["Ticker"] == "CASH", "CurrentPrice"] = 1

    portfolio["PriceSource"] = "yfinance"
    portfolio.loc[use_manual, "PriceSource"] = "manual"
    portfolio.loc[portfolio["Ticker"] == "CASH", "PriceSource"] = "cash"

    portfolio["FxRateToTHB"] = portfolio["Currency"].apply(fx_to_thb)
    portfolio["CostBasisNative"] = portfolio["Quantity"] * portfolio["AvgCost"]
    portfolio["MarketValueNative"] = portfolio["Quantity"] * portfolio["CurrentPrice"]
    portfolio["PnLNative"] = portfolio["MarketValueNative"] - portfolio["CostBasisNative"]
    portfolio["CostBasisTHB"] = portfolio["CostBasisNative"] * portfolio["FxRateToTHB"]
    portfolio["MarketValueTHB"] = portfolio["MarketValueNative"] * portfolio["FxRateToTHB"]
    portfolio["PnLTHB"] = portfolio["MarketValueTHB"] - portfolio["CostBasisTHB"]
    portfolio["IsCash"] = portfolio["Ticker"] == "CASH"
    portfolio["ReturnPct"] = 0.0
    mask = portfolio["CostBasisNative"] != 0
    portfolio.loc[mask, "ReturnPct"] = portfolio.loc[mask, "PnLNative"] / portfolio.loc[mask, "CostBasisNative"] * 100
    return portfolio


def get_portfolio_stats(portfolio_calc):
    cash = portfolio_calc[portfolio_calc["IsCash"]]
    inv = portfolio_calc[~portfolio_calc["IsCash"]]
    portfolio_cash = cash["MarketValueTHB"].sum()
    investment_value = inv["MarketValueTHB"].sum()
    investment_cost = inv["CostBasisTHB"].sum()
    investment_pnl = investment_value - investment_cost
    investment_return = investment_pnl / investment_cost * 100 if investment_cost else 0
    return {
        "portfolio_value": portfolio_cash + investment_value,
        "portfolio_cash": portfolio_cash,
        "investment_value": investment_value,
        "investment_cost": investment_cost,
        "investment_pnl": investment_pnl,
        "investment_return_pct": investment_return,
    }


def get_real_estate_summary(properties, mortgage, cashflow):
    real_estate = properties.merge(
        mortgage[["Property", "OutstandingDebt", "MonthlyPayment", "InterestRate"]],
        on="Property",
        how="left",
    )
    real_estate["OutstandingDebt"] = to_number(real_estate["OutstandingDebt"])
    real_estate["MonthlyPayment"] = to_number(real_estate["MonthlyPayment"])
    real_estate["Equity"] = real_estate["EstimatedValue"] - real_estate["OutstandingDebt"]

    cashflow = cashflow.copy()
    cashflow["NetCashFlow"] = cashflow["Rent"] - cashflow["Expense"] - cashflow["ExtraPayment"]
    monthly = {
        "rent": cashflow["Rent"].sum(),
        "expense": cashflow["Expense"].sum(),
        "mortgage": mortgage["MonthlyPayment"].sum(),
        "net": cashflow["Rent"].sum() - cashflow["Expense"].sum() - mortgage["MonthlyPayment"].sum(),
    }
    return real_estate["EstimatedValue"].sum(), real_estate["OutstandingDebt"].sum(), real_estate["Equity"].sum(), real_estate, cashflow, monthly


def render_net_worth_dashboard(stats, bank_cash, property_value, property_debt, property_equity, monthly):
    total_cash = stats["portfolio_cash"] + bank_cash
    investment_value = stats["investment_value"]
    net_worth = total_cash + investment_value + property_equity
    st.header("Net Worth Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Worth", format_thb(net_worth))
    c2.metric("Total Cash", format_thb(total_cash))
    c3.metric("Investments", format_thb(investment_value))
    c4.metric("Property Equity", format_thb(property_equity))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Property Value", format_thb(property_value))
    c6.metric("Property Debt", format_thb(property_debt))
    c7.metric("Investment P/L", format_thb(stats["investment_pnl"]))
    c8.metric("Investment Return", f"{stats['investment_return_pct']:.2f}%")
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Monthly Rent", format_thb(monthly["rent"]))
    c10.metric("Monthly Expense", format_thb(monthly["expense"]))
    c11.metric("Monthly Mortgage", format_thb(monthly["mortgage"]))
    c12.metric("Monthly Net Cashflow", format_thb(monthly["net"]))

    allocation = pd.DataFrame({"Category": ["Cash", "Investments", "Property Equity"], "ValueTHB": [total_cash, investment_value, property_equity]})
    debt = pd.DataFrame({"Category": ["Cash", "Investments", "Property Gross Value", "Property Debt", "Net Worth"], "ValueTHB": [total_cash, investment_value, property_value, -property_debt, net_worth]})
    c1, c2 = st.columns(2)
    with c1:
        if allocation["ValueTHB"].sum() != 0:
            st.plotly_chart(px.pie(allocation, names="Category", values="ValueTHB", title="Net Worth Allocation"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(debt, x="Category", y="ValueTHB", title="Assets, Debt, and Net Worth"), use_container_width=True)


def render_portfolio_section(portfolio_calc):
    st.subheader("Investment Portfolio")
    st.caption(f"Base Currency = THB | USD/THB = {get_usdthb_rate():,.4f}")
    stats = get_portfolio_stats(portfolio_calc)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", format_thb(stats["portfolio_value"]))
    c2.metric("Portfolio Cash", format_thb(stats["portfolio_cash"]))
    c3.metric("Investment Value", format_thb(stats["investment_value"]))
    c4.metric("Investment Return", f"{stats['investment_return_pct']:.2f}%")
    summary = ["Broker", "Ticker", "Currency", "Quantity", "AvgCost", "CurrentPrice", "PriceSource", "MarketValueNative", "MarketValueTHB", "PnLTHB", "ReturnPct"]
    st.dataframe(portfolio_calc[summary].round(2), use_container_width=True)
    missing = portfolio_calc[(portfolio_calc["CurrentPrice"] == 0) & (portfolio_calc["Ticker"] != "CASH")]
    if not missing.empty:
        st.warning("ยังไม่มีราคาสำหรับ: " + ", ".join(missing["Ticker"].unique()) + " — ให้ใส่ ManualPrice ใน Google Sheet")
    c1, c2 = st.columns(2)
    with c1:
        broker_summary = portfolio_calc.groupby("Broker", dropna=False)["MarketValueTHB"].sum().reset_index()
        if broker_summary["MarketValueTHB"].sum() != 0:
            st.plotly_chart(px.pie(broker_summary, names="Broker", values="MarketValueTHB", title="Allocation by Broker"), use_container_width=True)
    with c2:
        ticker_summary = portfolio_calc.groupby("Ticker", dropna=False)["MarketValueTHB"].sum().reset_index()
        if ticker_summary["MarketValueTHB"].sum() != 0:
            st.plotly_chart(px.pie(ticker_summary, names="Ticker", values="MarketValueTHB", title="Allocation by Ticker"), use_container_width=True)


def render_bank_section(banks):
    st.subheader("Bank Accounts")
    st.metric("Bank Cash", format_thb(banks["BalanceTHB"].sum()))
    st.dataframe(banks.round(2), use_container_width=True)
    if banks["BalanceTHB"].sum() != 0:
        st.plotly_chart(px.pie(banks, names="Bank", values="BalanceTHB", title="Bank Cash Allocation"), use_container_width=True)


def render_real_estate_section(real_estate, cashflow):
    st.subheader("Real Estate")
    c1, c2, c3 = st.columns(3)
    c1.metric("Property Value", format_thb(real_estate["EstimatedValue"].sum()))
    c2.metric("Property Debt", format_thb(real_estate["OutstandingDebt"].sum()))
    c3.metric("Property Equity", format_thb(real_estate["Equity"].sum()))
    st.markdown("### Real Estate Summary")
    st.dataframe(real_estate.round(2), use_container_width=True)
    if real_estate["Equity"].sum() != 0:
        st.plotly_chart(px.pie(real_estate, names="Property", values="Equity", title="Real Estate Equity Allocation"), use_container_width=True)
    st.markdown("### Monthly Property Cash Flow")
    st.dataframe(cashflow.round(2), use_container_width=True)
    if not cashflow.empty:
        summary = cashflow.groupby("Property", dropna=False)["NetCashFlow"].sum().reset_index()
        st.plotly_chart(px.bar(summary, x="Property", y="NetCashFlow", title="Net Cash Flow by Property"), use_container_width=True)


def get_portfolio_tickers_for_market():
    portfolio = load_portfolio()
    tickers = portfolio["Ticker"].apply(clean_ticker).replace("", pd.NA).dropna().unique().tolist()
    manual_only = {"CASH", "BRKB80", "K-USXNDQ-A(A)", "MTS-GOLD"}
    return [t for t in tickers if t not in manual_only]


def render_market_analysis():
    st.header("Market Analysis")
    portfolio_tickers = get_portfolio_tickers_for_market()
    default_tickers = list(dict.fromkeys(list(DEFAULT_MARKET_ASSETS.keys()) + portfolio_tickers))
    with st.sidebar:
        st.subheader("Market Settings")
        ticker_input = st.text_area("Tickers ที่ต้องการติดตาม", value=",".join(default_tickers))
        today = date.today()
        start_date = st.date_input("วันที่เริ่มต้น", value=today - timedelta(days=365 * 5))
        end_date = st.date_input("วันที่สิ้นสุด", value=today)
        momentum_choice = st.selectbox("เลือกช่วง Momentum", ["1M", "3M", "6M", "1Y"], index=1)
    selected = [clean_ticker(t) for t in ticker_input.split(",") if clean_ticker(t)]
    selected = list(dict.fromkeys(selected))
    if start_date >= end_date or not selected:
        st.warning("กรุณาเลือกช่วงเวลาและ ticker ให้ถูกต้อง")
        return
    data = download_prices(selected, start=start_date, end=end_date)
    if data.empty:
        st.warning("ไม่พบข้อมูลราคาจาก yfinance สำหรับ ticker ที่เลือก")
        return
    data = data.ffill().bfill().dropna(axis=1, how="all")
    returns = data.pct_change().dropna()
    normalized = data.div(data.iloc[0]).mul(100)
    st.subheader(f"Performance Comparison: {start_date} to {end_date}")
    fig = go.Figure()
    for ticker in normalized.columns:
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker], mode="lines", name=f"{ticker} - {get_asset_name(ticker)}"))
    fig.update_layout(template="plotly_dark", height=650, hovermode="x unified", yaxis=dict(type="log", title="Normalized Performance"))
    st.plotly_chart(fig, use_container_width=True)
    days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}[momentum_choice]
    if len(data) > days:
        momentum = ((data.iloc[-1] / data.iloc[-days]) - 1) * 100
        st.subheader(f"Momentum Ranking - {momentum_choice}")
        st.dataframe(pd.DataFrame({"Ticker": momentum.sort_values(ascending=False).index, f"Momentum {momentum_choice} %": momentum.sort_values(ascending=False).values}).round(2), use_container_width=True)
    if len(data.columns) >= 2 and not returns.empty:
        st.subheader("Correlation Heatmap")
        heatmap = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r")
        heatmap.update_layout(template="plotly_dark", height=650)
        st.plotly_chart(heatmap, use_container_width=True)


st.sidebar.caption("แก้ข้อมูลหลักใน Google Sheet แล้วกด Refresh Data")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

tab_wealth, tab_market = st.tabs(["My Wealth", "Market Analysis"])

with tab_wealth:
    portfolio = load_portfolio()
    banks = load_banks()
    properties = load_properties()
    mortgage = load_mortgage()
    cashflow = load_property_cashflow()
    portfolio_calc = calculate_portfolio(portfolio)
    stats = get_portfolio_stats(portfolio_calc)
    property_value, property_debt, property_equity, real_estate, cashflow_calc, monthly = get_real_estate_summary(properties, mortgage, cashflow)
    render_net_worth_dashboard(stats, banks["BalanceTHB"].sum(), property_value, property_debt, property_equity, monthly)
    st.divider()
    render_portfolio_section(portfolio_calc)
    st.divider()
    render_bank_section(banks)
    st.divider()
    render_real_estate_section(real_estate, cashflow_calc)

with tab_market:
    render_market_analysis()
