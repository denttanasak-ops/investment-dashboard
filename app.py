import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

st.set_page_config(
    page_title="Investment Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CONFIG
# =====================================================

GOOGLE_SHEET_ID = "1NfxJUlUyFmeSFjFCNLF7Xoeuu_vP_dfk9Hi2HP7Yl_c"

# ใช้ Google Sheet เป็นฐานข้อมูลหลัก
# ชื่อด้านขวาต้องตรงกับชื่อแท็บล่างใน Google Sheet
SHEET_NAMES = {
    "portfolio": "portfolio",
    "cash": "bank_accounts",
    "properties": "properties",
    "mortgage": "mortgage",
    "property_cashflow": "property_cashflow",
    "targets": "targets",
    "transactions": "transactions",
}

DEFAULT_TARGET_VALUE = 20_000_000
DEFAULT_MONTHLY_CONTRIBUTION = 60_000
DEFAULT_EXPECTED_RETURN = 0.08

# =====================================================
# HELPERS
# =====================================================

def google_sheet_csv_url(sheet_id: str, sheet_name: str) -> str:
    from urllib.parse import quote
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}"


@st.cache_data(ttl=300)
def load_google_sheet(sheet_name: str) -> pd.DataFrame:
    url = google_sheet_csv_url(GOOGLE_SHEET_ID, sheet_name)
    return pd.read_csv(url)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_load_sheet(sheet_key: str, fallback: pd.DataFrame) -> pd.DataFrame:
    try:
        real_sheet_name = SHEET_NAMES.get(sheet_key)
        if not real_sheet_name:
            st.sidebar.warning(f"ยังไม่ได้ตั้งชื่อแท็บสำหรับ '{sheet_key}'")
            return fallback.copy()

        df = load_google_sheet(real_sheet_name)
        if df.empty:
            st.sidebar.warning(f"แท็บ '{real_sheet_name}' ว่างหรืออ่านไม่ได้")
            return fallback.copy()

        st.sidebar.success(f"โหลด Google Sheet แท็บ '{real_sheet_name}' ได้")
        return normalize_columns(df)
    except Exception as e:
        st.sidebar.error(f"โหลด Google Sheet แท็บ '{SHEET_NAMES.get(sheet_key, sheet_key)}' ไม่ได้: {e}")
        return fallback.copy()


def to_number(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)


def money(value):
    try:
        return f"฿{value:,.0f}"
    except Exception:
        return "฿0"


def pct(value):
    try:
        return f"{value:.2f}%"
    except Exception:
        return "0.00%"


@st.cache_data(ttl=300)
def get_price_yfinance(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return np.nan
        return float(hist["Close"].iloc[-1])
    except Exception:
        return np.nan


def normalize_symbol_for_yfinance(symbol: str) -> str:
    mapping = {
        "BRK.B": "BRK-B",
        "BRKB": "BRK-B",
        "BRKB80": "BRK-B",
        "BTC": "BTC-USD",
        "BITCOIN": "BTC-USD",
        "GOLD": "GC=F",
        "MTS-GOLD": "GC=F",
    }
    s = str(symbol).strip().upper()
    return mapping.get(s, s)


def fallback_portfolio():
    return pd.DataFrame({
        "Symbol": ["MMYT", "BRK.B", "MELI", "OKLO", "RKLB", "SERV", "TEM", "K-USXNDQ-A(A)", "MTS-GOLD"],
        "Name": ["MakeMyTrip", "Berkshire Hathaway", "MercadoLibre", "Oklo", "Rocket Lab", "Serve Robotics", "Tempus AI", "NASDAQ Fund", "Gold"],
        "Asset Class": ["US Stock", "US Stock", "US Stock", "US Stock", "US Stock", "US Stock", "US Stock", "Fund", "Gold"],
        "Qty": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Avg Cost": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Manual Price": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Currency": ["USD", "USD", "USD", "USD", "USD", "USD", "USD", "THB", "THB"],
        "FX": [36, 36, 36, 36, 36, 36, 36, 1, 1],
    })


def fallback_cash():
    return pd.DataFrame({
        "Account": ["Bank Account", "Broker Cash", "Emergency Cash"],
        "Amount": [0, 0, 0],
        "Currency": ["THB", "THB", "THB"],
        "FX": [1, 1, 1],
    })


def fallback_properties():
    return pd.DataFrame({
        "Property": ["Home"],
        "Estimated Value": [0],
        "Note": [""],
    })


def fallback_mortgage():
    return pd.DataFrame({
        "Debt Name": ["Mortgage"],
        "Outstanding Balance": [0],
        "Interest Rate": [0],
        "Monthly Payment": [0],
    })


def pick_col(df: pd.DataFrame, candidates, default=None):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]
    return default


def prepare_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # รองรับชื่อคอลัมน์หลายแบบ เช่น symbol / Symbol, qty / Shares, avg_cost / Avg Cost
    rename_map = {}
    mappings = {
        "Symbol": ["Symbol", "Ticker", "Code", "Asset", "Stock"],
        "Name": ["Name", "Company", "Asset Name"],
        "Asset Class": ["Asset Class", "AssetClass", "Class", "Type", "Category"],
        "Qty": ["Qty", "Quantity", "Shares", "Units", "Amount"],
        "Avg Cost": ["Avg Cost", "Average Cost", "AvgCost", "Cost", "Buy Price", "Average Price"],
        "Manual Price": ["Manual Price", "Current Price", "Price", "Market Price", "Last Price"],
        "Currency": ["Currency", "CCY"],
        "FX": ["FX", "Exchange Rate", "Fx Rate", "THB Rate"],
    }
    for std_col, candidates in mappings.items():
        found = pick_col(df, candidates)
        if found is not None and found != std_col:
            rename_map[found] = std_col
    df = df.rename(columns=rename_map)

    required = ["Symbol", "Name", "Asset Class", "Qty", "Avg Cost", "Manual Price", "Currency", "FX"]
    for col in required:
        if col not in df.columns:
            if col in ["Qty", "Avg Cost", "Manual Price", "FX"]:
                df[col] = 0
            elif col == "Currency":
                df[col] = "THB"
            else:
                df[col] = ""

    df["Qty"] = to_number(df["Qty"])
    df["Avg Cost"] = to_number(df["Avg Cost"])
    df["Manual Price"] = to_number(df["Manual Price"])
    df["FX"] = to_number(df["FX"]).replace(0, 1)

    prices = []
    for _, row in df.iterrows():
        manual_price = row["Manual Price"]
        if manual_price > 0:
            prices.append(manual_price)
        else:
            yf_symbol = normalize_symbol_for_yfinance(row["Symbol"])
            prices.append(get_price_yfinance(yf_symbol))

    df["Current Price"] = pd.Series(prices).fillna(0)
    df["Cost Value"] = df["Qty"] * df["Avg Cost"] * df["FX"]
    df["Market Value"] = df["Qty"] * df["Current Price"] * df["FX"]
    df["Gain/Loss"] = df["Market Value"] - df["Cost Value"]
    df["Return %"] = np.where(df["Cost Value"] > 0, df["Gain/Loss"] / df["Cost Value"] * 100, 0)
    return df


def calculate_goal_projection(current_value, monthly_contribution, target_value, expected_return):
    if current_value >= target_value:
        return 0, current_value

    monthly_rate = expected_return / 12
    value = current_value
    months = 0
    while value < target_value and months < 600:
        value = value * (1 + monthly_rate) + monthly_contribution
        months += 1
    return months, value


def get_news_links(symbols):
    rows = []
    for symbol in symbols:
        if not symbol:
            continue
        query = str(symbol).replace(" ", "+")
        rows.append({
            "Asset": symbol,
            "Google News": f"https://news.google.com/search?q={query}",
            "Yahoo Finance": f"https://finance.yahoo.com/quote/{normalize_symbol_for_yfinance(symbol)}"
        })
    return pd.DataFrame(rows)

# =====================================================
# LOAD DATA
# =====================================================

portfolio_raw = safe_load_sheet("portfolio", fallback_portfolio())
cash_raw = safe_load_sheet("cash", fallback_cash())
properties_raw = safe_load_sheet("properties", fallback_properties())
mortgage_raw = safe_load_sheet("mortgage", fallback_mortgage())

portfolio = prepare_portfolio(portfolio_raw)

cash = normalize_columns(cash_raw)
amount_col = pick_col(cash, ["Amount", "Balance", "Value", "THB Value", "Cash"])
fx_col = pick_col(cash, ["FX", "Exchange Rate", "Fx Rate"])
if amount_col is None:
    cash["Amount"] = 0
else:
    cash["Amount"] = to_number(cash[amount_col])
if fx_col is None:
    cash["FX"] = 1
else:
    cash["FX"] = to_number(cash[fx_col]).replace(0, 1)
cash["THB Value"] = cash["Amount"] * cash["FX"]

properties = normalize_columns(properties_raw)
property_value_col = pick_col(properties, ["Estimated Value", "Value", "Market Value", "Price", "Asset Value"])
if property_value_col is None:
    properties["Estimated Value"] = 0
else:
    properties["Estimated Value"] = to_number(properties[property_value_col])

mortgage = normalize_columns(mortgage_raw)
debt_col = pick_col(mortgage, ["Outstanding Balance", "Balance", "Debt", "Loan", "Principal"])
if debt_col is None:
    mortgage["Outstanding Balance"] = 0
else:
    mortgage["Outstanding Balance"] = to_number(mortgage[debt_col])

portfolio_value = float(portfolio["Market Value"].sum())
portfolio_cost = float(portfolio["Cost Value"].sum())
portfolio_gain = portfolio_value - portfolio_cost
portfolio_return = (portfolio_gain / portfolio_cost * 100) if portfolio_cost > 0 else 0
cash_value = float(cash["THB Value"].sum())
property_value = float(properties["Estimated Value"].sum())
debt_value = float(mortgage["Outstanding Balance"].sum())
net_worth = portfolio_value + cash_value + property_value - debt_value

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📊 Investment Dashboard")
st.sidebar.caption("Version 1")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.metric("Net Worth", money(net_worth))
st.sidebar.metric("Portfolio", money(portfolio_value))
st.sidebar.metric("Cash", money(cash_value))

# =====================================================
# TABS
# =====================================================

tab_wealth, tab_portfolio, tab_retirement, tab_news, tab_macro, tab_market = st.tabs([
    "💰 My Wealth",
    "📈 Portfolio Dashboard",
    "🎯 Retirement Plan",
    "📰 Portfolio News",
    "🌍 Macro Dashboard",
    "🔎 Market Analysis",
])

# =====================================================
# TAB 1: MY WEALTH
# =====================================================

with tab_wealth:
    st.header("💰 My Wealth")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Worth", money(net_worth))
    c2.metric("Portfolio", money(portfolio_value))
    c3.metric("Cash", money(cash_value))
    c4.metric("Debt", money(debt_value))

    st.subheader("Wealth Breakdown")
    wealth_df = pd.DataFrame({
        "Category": ["Portfolio", "Cash", "Properties", "Debt"],
        "Value": [portfolio_value, cash_value, property_value, -debt_value]
    })
    st.bar_chart(wealth_df.set_index("Category"))

    st.subheader("Portfolio")
    st.dataframe(portfolio[["Symbol", "Name", "Asset Class", "Qty", "Market Value", "Gain/Loss", "Return %"]], use_container_width=True)

    st.subheader("Bank / Cash Accounts")
    st.dataframe(cash, use_container_width=True)

    st.subheader("Properties")
    st.dataframe(properties, use_container_width=True)

    st.subheader("Mortgage / Debt")
    st.dataframe(mortgage, use_container_width=True)

# =====================================================
# TAB 2: PORTFOLIO DASHBOARD
# =====================================================

with tab_portfolio:
    st.header("📈 Portfolio Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", money(portfolio_value))
    c2.metric("Cost", money(portfolio_cost))
    c3.metric("Gain / Loss", money(portfolio_gain))
    c4.metric("Return", pct(portfolio_return))

    st.subheader("Holdings")
    show_cols = ["Symbol", "Name", "Asset Class", "Qty", "Avg Cost", "Current Price", "Market Value", "Gain/Loss", "Return %"]
    st.dataframe(portfolio[show_cols].sort_values("Market Value", ascending=False), use_container_width=True)

    st.subheader("Allocation by Asset Class")
    allocation = portfolio.groupby("Asset Class", as_index=False)["Market Value"].sum()
    allocation = allocation[allocation["Market Value"] > 0]
    if allocation.empty:
        st.info("ยังไม่มีมูลค่าพอร์ต ให้กรอก Qty และราคาใน Google Sheet ก่อน")
    else:
        st.bar_chart(allocation.set_index("Asset Class"))

    st.subheader("Top Holdings")
    top_holdings = portfolio[portfolio["Market Value"] > 0].sort_values("Market Value", ascending=False).head(10)
    if top_holdings.empty:
        st.info("ยังไม่มีสินทรัพย์ที่มีมูลค่า")
    else:
        st.bar_chart(top_holdings.set_index("Symbol")[["Market Value"]])

# =====================================================
# TAB 3: RETIREMENT PLAN
# =====================================================

with tab_retirement:
    st.header("🎯 Retirement Plan")

    st.caption("ค่าเริ่มต้นตามที่คุยกัน: เป้าหมาย 20 ล้านบาท เติมเงินเดือนละ 60,000 บาท")

    c1, c2, c3 = st.columns(3)
    target_value = c1.number_input("Target Value", min_value=0, value=DEFAULT_TARGET_VALUE, step=100_000)
    monthly_contribution = c2.number_input("Monthly Contribution", min_value=0, value=DEFAULT_MONTHLY_CONTRIBUTION, step=5_000)
    expected_return = c3.number_input("Expected Return / Year (%)", min_value=0.0, max_value=30.0, value=DEFAULT_EXPECTED_RETURN * 100, step=0.5) / 100

    progress = min(net_worth / target_value, 1) if target_value > 0 else 0
    months_needed, projected_value = calculate_goal_projection(net_worth, monthly_contribution, target_value, expected_return)
    years_needed = months_needed / 12

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Net Worth", money(net_worth))
    c2.metric("Target", money(target_value))
    c3.metric("Progress", pct(progress * 100))
    c4.metric("Estimated Time", f"{years_needed:.1f} years")

    st.progress(progress)

    st.subheader("Projection")
    projection_rows = []
    value = net_worth
    monthly_rate = expected_return / 12
    for month in range(0, 121):
        if month > 0:
            value = value * (1 + monthly_rate) + monthly_contribution
        if month % 12 == 0:
            projection_rows.append({
                "Year": month // 12,
                "Projected Value": value
            })
    projection_df = pd.DataFrame(projection_rows)
    st.line_chart(projection_df.set_index("Year"))
    st.dataframe(projection_df, use_container_width=True)

# =====================================================
# TAB 4: PORTFOLIO NEWS
# =====================================================

with tab_news:
    st.header("📰 Portfolio News")
    st.caption("Version 1 จะทำเป็นลิงก์ข่าวเฉพาะสินทรัพย์ที่ถือก่อน ต่อไปค่อยทำ AI Summary")

    symbols = portfolio["Symbol"].dropna().astype(str).unique().tolist()
    news_df = get_news_links(symbols)
    st.dataframe(
        news_df,
        column_config={
            "Google News": st.column_config.LinkColumn("Google News"),
            "Yahoo Finance": st.column_config.LinkColumn("Yahoo Finance"),
        },
        use_container_width=True,
        hide_index=True,
    )

    st.info("ถ้าจะให้ระบบสรุปข่าวจริงในหน้าเว็บ ต้องเพิ่ม News API หรือใช้ RSS ในเวอร์ชันถัดไป")

# =====================================================
# TAB 5: MACRO DASHBOARD
# =====================================================

with tab_macro:
    st.header("🌍 Macro Dashboard")

    macro_assets = pd.DataFrame({
        "Name": ["S&P 500", "Nasdaq 100", "Bitcoin", "Gold Futures", "US 10Y Yield", "Dollar Index"],
        "Symbol": ["^GSPC", "^NDX", "BTC-USD", "GC=F", "^TNX", "DX-Y.NYB"]
    })

    rows = []
    for _, row in macro_assets.iterrows():
        price = get_price_yfinance(row["Symbol"])
        rows.append({"Name": row["Name"], "Symbol": row["Symbol"], "Latest": price})
    macro_df = pd.DataFrame(rows)

    st.dataframe(macro_df, use_container_width=True)

    st.subheader("Macro Watchlist")
    st.markdown(
        """
        - S&P 500 / Nasdaq: ภาพรวมตลาดหุ้นสหรัฐ
        - Bitcoin: สินทรัพย์เสี่ยงและกระแสเงินในตลาดคริปโต
        - Gold: ความกลัว เงินเฟ้อ และ real yield
        - US 10Y Yield: ต้นทุนเงินทุนของตลาดโลก
        - Dollar Index: ค่าเงินดอลลาร์ กระทบสินทรัพย์เสี่ยงและทองคำ
        """
    )

# =====================================================
# TAB 6: MARKET ANALYSIS
# =====================================================

with tab_market:
    st.header("🔎 Market Analysis")
    st.caption("คงแท็บนี้ไว้สำหรับต่อยอดการวิเคราะห์ตลาด")

    selected_symbol = st.text_input("Enter Symbol", value="MMYT")
    yf_symbol = normalize_symbol_for_yfinance(selected_symbol)

    if selected_symbol:
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1y")
            if hist.empty:
                st.warning("ไม่พบข้อมูลราคา")
            else:
                st.subheader(f"Price Chart: {selected_symbol}")
                st.line_chart(hist[["Close"]])

                last_price = hist["Close"].iloc[-1]
                first_price = hist["Close"].iloc[0]
                one_year_return = (last_price / first_price - 1) * 100

                c1, c2 = st.columns(2)
                c1.metric("Latest Price", f"{last_price:,.2f}")
                c2.metric("1Y Return", pct(one_year_return))
        except Exception as e:
            st.error(f"โหลดข้อมูลไม่ได้: {e}")
