import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

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

    # ตัดคอลัมน์ว่างที่ Google Sheet ส่งมา เช่น Unnamed: 4, Unnamed: 5
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # ตัดแถวว่างทั้งหมด
    df = df.dropna(how="all")
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
        "Manual Price": ["Manual Price", "Current Price", "CurrentPrice", "Price", "Market Price", "MarketPrice", "Last Price", "LastPrice"],
        "Currency": ["Currency", "CCY"],
        "FX": ["FX", "Exchange Rate", "Fx Rate", "THB Rate", "ExchangeRate", "FxRate"],
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

    # ถ้ามีคอลัมน์มูลค่ารวมในชีต ให้ใช้ค่าจากชีตเป็นหลัก
    # เพื่อกันกรณี BRKB80 / กองทุนไทย / ทอง ที่ yfinance อาจตีราคาเป็นหุ้นสหรัฐผิดตัว
    market_value_col = pick_col(df, ["Market Value", "MarketValue", "Current Value", "CurrentValue", "Value", "THB Value", "THBValue"])
    cost_value_col = pick_col(df, ["Cost Value", "CostValue", "Total Cost", "TotalCost", "Invested", "Investment"])

    prices = []
    for _, row in df.iterrows():
        manual_price = row["Manual Price"]
        symbol_text = str(row["Symbol"]).strip().upper()
        asset_class_text = str(row["Asset Class"]).strip().lower()

        if symbol_text in ["CASH", "CASH THB", "THB CASH", "เงินสด"]:
            prices.append(1)
        elif manual_price > 0:
            prices.append(manual_price)
        elif symbol_text.endswith("80") or "FUND" in asset_class_text or "GOLD" in asset_class_text or "กองทุน" in asset_class_text:
            prices.append(0)
        else:
            yf_symbol = normalize_symbol_for_yfinance(row["Symbol"])
            prices.append(get_price_yfinance(yf_symbol))

    df["Current Price"] = pd.Series(prices).fillna(0)

    # CASH ใน portfolio ให้ถือว่าเป็นเงินบาท 1:1 ไม่ให้ yfinance ไปตีเป็นหุ้นชื่อ CASH
    cash_mask = df["Symbol"].astype(str).str.strip().str.upper().isin(["CASH", "CASH THB", "THB CASH", "เงินสด"])
    df.loc[cash_mask, "FX"] = 1
    df.loc[cash_mask, "Current Price"] = 1
    df.loc[cash_mask & (df["Avg Cost"] == 0), "Avg Cost"] = 1
    df.loc[cash_mask & (df["Asset Class"].astype(str).str.strip() == ""), "Asset Class"] = "Cash"

    df["Cost Value"] = df["Qty"] * df["Avg Cost"] * df["FX"]
    df["Market Value"] = df["Qty"] * df["Current Price"] * df["FX"]

    if cost_value_col is not None:
        df["Cost Value"] = to_number(df[cost_value_col])
    if market_value_col is not None:
        sheet_market_value = to_number(df[market_value_col])
        # ใช้ค่าจากชีตเฉพาะแถวที่มีค่ามากกว่า 0; ถ้าเป็น 0 ให้ใช้ราคาที่คำนวณแทน
        df["Market Value"] = np.where(sheet_market_value > 0, sheet_market_value, df["Market Value"])
        df["Current Price"] = np.where(df["Qty"] > 0, df["Market Value"] / df["Qty"] / df["FX"], df["Current Price"])

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
amount_col = pick_col(cash, ["Amount", "Balance", "Value", "THB Value", "THBValue", "Cash"])
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
property_value_col = pick_col(properties, ["Estimated Value", "EstimatedValue", "Value", "Market Value", "MarketValue", "Price", "Asset Value", "AssetValue"])
if property_value_col is None:
    properties["Estimated Value"] = 0
else:
    properties["Estimated Value"] = to_number(properties[property_value_col])

mortgage = normalize_columns(mortgage_raw)
debt_col = pick_col(mortgage, ["Outstanding Balance", "OutstandingBalance", "Outstanding Debt", "OutstandingDebt", "Balance", "Debt", "Loan", "Principal"])
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

def calculate_risk_metrics(price_df: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    returns = price_df.pct_change().dropna()
    rows = []
    for asset in price_df.columns:
        series = price_df[asset].dropna()
        r = returns[asset].dropna() if asset in returns.columns else pd.Series(dtype=float)
        if len(series) < 2 or r.empty:
            continue

        total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
        years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
        cagr = ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100
        volatility = r.std() * np.sqrt(252) * 100
        sharpe = ((r.mean() * 252) - risk_free_rate) / (r.std() * np.sqrt(252)) if r.std() != 0 else 0
        drawdown = series / series.cummax() - 1
        max_drawdown = drawdown.min() * 100
        best_day = r.max() * 100
        worst_day = r.min() * 100

        rows.append({
            "Asset": asset,
            "Total Return %": total_return,
            "CAGR %": cagr,
            "Volatility %": volatility,
            "Sharpe": sharpe,
            "Max Drawdown %": max_drawdown,
            "Best Day %": best_day,
            "Worst Day %": worst_day,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=900)
def get_portfolio_price_history(symbols: list, period: str = "1y") -> pd.DataFrame:
    y_symbols = []
    symbol_name_map = {}
    skip_symbols = {"CASH", "CASH THB", "THB CASH", "เงินสด", "BRKB80", "K-USXNDQ-A(A)", "MTS-GOLD"}

    for symbol in symbols:
        raw = str(symbol).strip().upper()
        if raw in skip_symbols or raw.endswith("80"):
            continue
        yf_symbol = normalize_symbol_for_yfinance(raw)
        y_symbols.append(yf_symbol)
        symbol_name_map[yf_symbol] = raw

    if not y_symbols:
        return pd.DataFrame()

    try:
        raw_data = yf.download(
            tickers=list(dict.fromkeys(y_symbols)),
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw_data.empty:
            return pd.DataFrame()

        if isinstance(raw_data.columns, pd.MultiIndex):
            close = raw_data["Close"] if "Close" in raw_data.columns.get_level_values(0) else raw_data.xs("Close", level=1, axis=1)
        else:
            close = raw_data["Close"] if "Close" in raw_data.columns else raw_data

        if isinstance(close, pd.Series):
            close = close.to_frame(name=y_symbols[0])

        close.columns = [symbol_name_map.get(str(c).upper(), str(c).upper()) for c in close.columns]
        close = close.dropna(how="all").ffill().bfill().dropna(axis=1, how="all")
        return close
    except Exception:
        return pd.DataFrame()


with tab_portfolio:
    st.header("📈 Portfolio Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", money(portfolio_value))
    c2.metric("Cost", money(portfolio_cost))
    c3.metric("Gain / Loss", money(portfolio_gain))
    c4.metric("Return", pct(portfolio_return))

    st.subheader("Holdings")
    show_cols = ["Symbol", "Name", "Asset Class", "Qty", "Avg Cost", "Current Price", "Market Value", "Gain/Loss", "Return %"]
    view_portfolio = portfolio[show_cols].sort_values("Market Value", ascending=False)
    st.dataframe(view_portfolio, use_container_width=True)

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

    st.subheader("Portfolio Risk Metrics")
    risk_period = st.selectbox("ช่วงเวลาคำนวณความเสี่ยง", ["6mo", "1y", "3y", "5y"], index=1, key="portfolio_risk_period")
    portfolio_symbols = portfolio["Symbol"].dropna().astype(str).tolist()
    portfolio_price_history = get_portfolio_price_history(portfolio_symbols, period=risk_period)

    if portfolio_price_history.empty:
        st.info("ยังไม่มีข้อมูลราคาย้อนหลังสำหรับคำนวณความเสี่ยงของหุ้นในพอร์ต หรือสินทรัพย์บางตัวต้องใช้ Manual Price")
    else:
        portfolio_risk_df = calculate_risk_metrics(portfolio_price_history)
        if portfolio_risk_df.empty:
            st.info("ข้อมูลยังไม่พอสำหรับคำนวณ risk metrics")
        else:
            st.dataframe(portfolio_risk_df.round(2), use_container_width=True, hide_index=True)

        if len(portfolio_price_history.columns) >= 2:
            st.subheader("Portfolio Holdings Correlation")
            portfolio_returns = portfolio_price_history.pct_change().dropna()
            portfolio_corr = portfolio_returns.corr()
            portfolio_heatmap = px.imshow(portfolio_corr, text_auto=".2f", color_continuous_scale="RdBu_r")
            portfolio_heatmap.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(portfolio_heatmap, use_container_width=True)

        st.caption("หมายเหตุ: ค่านี้คำนวณได้เฉพาะสินทรัพย์ที่ yfinance มีข้อมูล เช่น MMYT, MELI, OKLO, RKLB ฯลฯ ส่วน BRKB80, กองทุนไทย, ทองไทย หรือ CASH จะยังไม่ถูกนำมาคำนวณในตารางนี้")

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

@st.cache_data(ttl=900)
def get_macro_history(symbol_map: dict, start=None, end=None, period: str | None = None) -> pd.DataFrame:
    tickers = [symbol for symbol in symbol_map.values() if symbol]
    if not tickers:
        return pd.DataFrame()

    try:
        if period:
            raw = yf.download(tickers=tickers, period=period, auto_adjust=True, progress=False, group_by="column", threads=True)
        else:
            raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column", threads=True)

        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", level=1, axis=1)
        else:
            close = raw["Close"] if "Close" in raw.columns else raw

        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])

        reverse_map = {v: k for k, v in symbol_map.items()}
        close.columns = [reverse_map.get(str(c), str(c)) for c in close.columns]
        close = close.dropna(how="all").ffill().bfill().dropna(axis=1, how="all")
        return close
    except Exception:
        return pd.DataFrame()


def calculate_risk_metrics(price_df: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    returns = price_df.pct_change().dropna()
    rows = []
    for asset in price_df.columns:
        series = price_df[asset].dropna()
        r = returns[asset].dropna() if asset in returns.columns else pd.Series(dtype=float)
        if len(series) < 2 or r.empty:
            continue

        total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
        years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
        cagr = ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100
        volatility = r.std() * np.sqrt(252) * 100
        sharpe = ((r.mean() * 252) - risk_free_rate) / (r.std() * np.sqrt(252)) if r.std() != 0 else 0
        drawdown = series / series.cummax() - 1
        max_drawdown = drawdown.min() * 100
        best_day = r.max() * 100
        worst_day = r.min() * 100

        rows.append({
            "Asset": asset,
            "Total Return %": total_return,
            "CAGR %": cagr,
            "Volatility %": volatility,
            "Sharpe": sharpe,
            "Max Drawdown %": max_drawdown,
            "Best Day %": best_day,
            "Worst Day %": worst_day,
        })
    return pd.DataFrame(rows)


with tab_macro:
    st.header("🌍 Macro Dashboard")
    st.caption("เปรียบเทียบสินทรัพย์โลกแบบ log scale, momentum, risk metrics และ correlation")

    macro_universe = {
        "SPY - US Market": "SPY",
        "QQQ - US Tech / AI": "QQQ",
        "SOXX - Semiconductor": "SOXX",
        "XLV - Healthcare": "XLV",
        "ITA - Aerospace": "ITA",
        "XLE - Energy": "XLE",
        "BRK-B - Berkshire": "BRK-B",
        "INDA - India": "INDA",
        "MCHI - China": "MCHI",
        "THD - Thailand ETF": "THD",
        "GLD - Gold": "GLD",
        "BTC - Bitcoin": "BTC-USD",
        "EEM - Emerging Markets": "EEM",
        "EWJ - Japan": "EWJ",
        "DXY - Dollar Index": "DX-Y.NYB",
        "US10Y - US 10Y Yield": "^TNX",
        "WTI - Oil": "CL=F",
    }

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    selected_assets = c1.multiselect(
        "เลือกสินทรัพย์ที่ต้องการเปรียบเทียบ",
        options=list(macro_universe.keys()),
        default=["SPY - US Market", "QQQ - US Tech / AI", "BTC - Bitcoin", "GLD - Gold", "INDA - India", "MCHI - China"],
    )
    today = datetime.now().date()
    start_date_macro = c2.date_input("วันที่เริ่มต้น", value=today - timedelta(days=365 * 5), key="macro_start")
    end_date_macro = c3.date_input("วันที่สิ้นสุด", value=today, key="macro_end")
    momentum_choice = c4.selectbox("Momentum", ["1M", "3M", "6M", "1Y"], index=1)

    if start_date_macro >= end_date_macro or len(selected_assets) == 0:
        st.warning("กรุณาเลือกช่วงเวลาและสินทรัพย์ให้ถูกต้อง")
    else:
        selected_map = {name: macro_universe[name] for name in selected_assets}
        price_df = get_macro_history(selected_map, start=start_date_macro, end=end_date_macro + timedelta(days=1))

        if price_df.empty:
            st.warning("ยังโหลดข้อมูลไม่ได้จาก yfinance")
        else:
            returns = price_df.pct_change().dropna()
            normalized = price_df.div(price_df.iloc[0]).mul(100)

            latest_rows = []
            for name in price_df.columns:
                latest = price_df[name].dropna().iloc[-1]
                first = price_df[name].dropna().iloc[0]
                latest_rows.append({
                    "Asset": name,
                    "Symbol": selected_map.get(name, ""),
                    "Latest": latest,
                    "Total Return %": (latest / first - 1) * 100,
                })
            latest_df = pd.DataFrame(latest_rows)
            st.subheader("Latest Macro Prices")
            st.dataframe(latest_df.round(2), use_container_width=True, hide_index=True)

            st.subheader("Performance Comparison: Indexed to 100 / Log Scale")
            fig = go.Figure()
            for asset in normalized.columns:
                fig.add_trace(go.Scatter(x=normalized.index, y=normalized[asset], mode="lines", name=asset))
            fig.update_layout(
                template="plotly_dark",
                height=650,
                hovermode="x unified",
                yaxis=dict(type="log", title="Normalized Performance, start = 100"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}[momentum_choice]
            if len(price_df) > days:
                momentum = ((price_df.iloc[-1] / price_df.iloc[-days]) - 1) * 100
                momentum_df = pd.DataFrame({
                    "Asset": momentum.sort_values(ascending=False).index,
                    f"Momentum {momentum_choice} %": momentum.sort_values(ascending=False).values,
                })
                st.subheader(f"Momentum Ranking - {momentum_choice}")
                st.dataframe(momentum_df.round(2), use_container_width=True, hide_index=True)

            st.subheader("Risk Metrics")
            risk_df = calculate_risk_metrics(price_df)
            if risk_df.empty:
                st.info("ข้อมูลยังไม่พอสำหรับคำนวณ risk metrics")
            else:
                st.dataframe(risk_df.round(2), use_container_width=True, hide_index=True)

            if len(price_df.columns) >= 2 and not returns.empty:
                st.subheader("Correlation Heatmap")
                corr = returns.corr()
                heatmap = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r")
                heatmap.update_layout(template="plotly_dark", height=650)
                st.plotly_chart(heatmap, use_container_width=True)

            st.caption("หมายเหตุ: Risk metrics คำนวณจาก daily returns และ annualize ด้วย 252 trading days; Sharpe ในเวอร์ชันนี้ใช้ risk-free rate = 0 เพื่อดูเปรียบเทียบเบื้องต้น")

# =====================================================
# TAB 6: MARKET ANALYSIS
# =====================================================

with tab_market:
    st.header("🔎 Market Analysis")
    st.caption("คงแท็บนี้ไว้สำหรับต่อยอดการวิเคราะห์ตลาด")

    c1, c2, c3 = st.columns([2, 1, 1])
    selected_symbol = c1.text_input("Enter Symbol", value="MMYT")
    start_date = c2.date_input("Start Date", value=datetime.now().date() - timedelta(days=365))
    end_date = c3.date_input("End Date", value=datetime.now().date())

    yf_symbol = normalize_symbol_for_yfinance(selected_symbol)

    if selected_symbol:
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
            if hist.empty:
                st.warning("ไม่พบข้อมูลราคา")
            else:
                st.subheader(f"Price Chart: {selected_symbol}")
                st.line_chart(hist[["Close"]])

                last_price = hist["Close"].iloc[-1]
                first_price = hist["Close"].iloc[0]
                period_return = (last_price / first_price - 1) * 100

                c1, c2 = st.columns(2)
                c1.metric("Latest Price", f"{last_price:,.2f}")
                c2.metric("Selected Period Return", pct(period_return))
        except Exception as e:
            st.error(f"โหลดข้อมูลไม่ได้: {e}")
