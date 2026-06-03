from datetime import date, datetime, timedelta
from urllib.parse import quote, quote_plus
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# =====================================================
# APP CONFIG
# =====================================================

st.set_page_config(
    page_title="Investment Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE_CURRENCY = "THB"
GOOGLE_SHEET_ID = "1NfxJUlUyFmeSFjFCNLF7Xoeuu_vP_dfk9Hi2HP7Yl_c"

SHEET_TABS = {
    "portfolio": "portfolio",
    "bank_accounts": "bank_accounts",
    "properties": "properties",
    "mortgage": "mortgage",
    "property_cashflow": "property_cashflow",
    "watchlist": "watchlist",
    "options": "options",
}

DEFAULT_TARGET_VALUE = 20_000_000
DEFAULT_MONTHLY_CONTRIBUTION = 60_000
DEFAULT_EXPECTED_RETURN = 0.08

DEFAULT_SCAN_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "HD", "BKNG", "COST", "NFLX", "AMD", "MU", "CRWD", "PANW",
    "MELI", "MMYT", "RKLB", "OKLO", "SERV", "TEM", "SYM",
    "QQQ", "SPY", "SOXX", "XLV", "XLE", "INDA", "MCHI", "GLD"
]

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
    "EEM": "Emerging Markets",
    "EWJ": "Japan",
    "DX-Y.NYB": "Dollar Index",
    "^TNX": "US 10Y Yield",
    "CL=F": "Oil WTI",
}

MANUAL_ONLY_TICKERS = {
    "CASH",
    "CASH THB",
    "THB CASH",
    "เงินสด",
    "BRKB80",
    "K-USXNDQ-A(A)",
    "MTS-GOLD",
}


# =====================================================
# BASIC HELPERS
# =====================================================

def clean_ticker(ticker):
    if pd.isna(ticker):
        return ""
    return str(ticker).strip().upper()


def clean_currency(currency):
    if pd.isna(currency) or str(currency).strip() == "":
        return BASE_CURRENCY
    return str(currency).strip().upper()


def to_number(series, default=0):
    if isinstance(series, pd.Series):
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(default)
    return pd.to_numeric(pd.Series(series), errors="coerce").fillna(default)


def format_thb(value):
    try:
        return f"฿{float(value):,.0f}"
    except Exception:
        return "฿0"


def format_thb_2(value):
    try:
        return f"{float(value):,.2f} THB"
    except Exception:
        return "0.00 THB"


def pct(value):
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "0.00%"


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


def get_asset_name(ticker):
    ticker = clean_ticker(ticker)
    return DEFAULT_MARKET_ASSETS.get(ticker, ticker)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.dropna(how="all")


def pick_col(df: pd.DataFrame, candidates, default=None):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]
    return default


def ensure_columns(df, columns):
    df = normalize_columns(df)
    for col, default in columns.items():
        if col not in df.columns:
            df[col] = default
    return df[list(columns.keys())]


# =====================================================
# GOOGLE SHEET LOADERS
# =====================================================

@st.cache_data(ttl=60)
def read_google_sheet_tab(sheet_name):
    encoded_sheet = quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={encoded_sheet}"
    return pd.read_csv(url)


def safe_read_tab(tab_key, fallback):
    sheet_name = SHEET_TABS.get(tab_key)
    if not sheet_name:
        st.sidebar.warning(f"ยังไม่ได้ตั้งชื่อแท็บสำหรับ {tab_key}")
        return fallback.copy()
    try:
        df = read_google_sheet_tab(sheet_name)
        if df.empty:
            st.sidebar.warning(f"แท็บ {sheet_name} ว่าง")
            return fallback.copy()
        st.sidebar.success(f"โหลด Google Sheet แท็บ '{sheet_name}' ได้")
        return normalize_columns(df)
    except Exception as e:
        st.sidebar.warning(f"โหลดแท็บ '{sheet_name}' ไม่ได้: {e}")
        return fallback.copy()


# =====================================================
# FALLBACK STRUCTURES
# =====================================================

PORTFOLIO_COLUMNS = {
    "Broker": "",
    "Ticker": "",
    "Currency": "THB",
    "Quantity": 0.0,
    "AvgCost": 0.0,
    "ManualPrice": 0.0,
}

BANK_COLUMNS = {
    "Bank": "",
    "Account": "",
    "Balance": 0.0,
    "Currency": "THB",
}

PROPERTY_COLUMNS = {
    "Property": "",
    "Type": "",
    "EstimatedValue": 0.0,
    "Location": "",
}

MORTGAGE_COLUMNS = {
    "Property": "",
    "OutstandingDebt": 0.0,
    "MonthlyPayment": 0.0,
    "InterestRate": 0.0,
}

CASHFLOW_COLUMNS = {
    "Property": "",
    "Month": "",
    "Rent": 0.0,
    "Expense": 0.0,
    "ExtraPayment": 0.0,
}

WATCHLIST_COLUMNS = {
    "Symbol": "",
    "Name": "",
    "Theme": "",
    "TargetPrice": 0.0,
    "Thesis": "",
}

OPTIONS_COLUMNS = {
    "Underlying": "",
    "OptionType": "",
    "PositionSide": "SELL",
    "Strike": 0.0,
    "Expiry": "",
    "Contracts": 0.0,
    "EntryPrice": 0.0,
    "CurrentBid": 0.0,
    "CurrentMark": 0.0,
    "CurrentAsk": 0.0,
    "Delta": 0.0,
    "Gamma": 0.0,
    "Theta": 0.0,
    "Vega": 0.0,
    "IV": 0.0,
    "Volume": 0.0,
    "OpenInterest": 0.0,
    "Status": "OPEN",
    "UnderlyingPrice": 0.0,
    "Note": "",
}


def fallback_portfolio():
    return pd.DataFrame(PORTFOLIO_COLUMNS, index=[])


def fallback_banks():
    return pd.DataFrame(BANK_COLUMNS, index=[])


def fallback_properties():
    return pd.DataFrame(PROPERTY_COLUMNS, index=[])


def fallback_mortgage():
    return pd.DataFrame(MORTGAGE_COLUMNS, index=[])


def fallback_cashflow():
    return pd.DataFrame(CASHFLOW_COLUMNS, index=[])


def fallback_watchlist():
    return pd.DataFrame({
        "Symbol": ["SPY", "QQQ", "BTC-USD", "GLD", "INDA", "MCHI", "NVDA", "GOOGL", "AMZN"],
        "Name": ["S&P 500 ETF", "Nasdaq 100 ETF", "Bitcoin", "Gold ETF", "India ETF", "China ETF", "NVIDIA", "Alphabet", "Amazon"],
        "Theme": ["US Market", "US Tech", "Crypto", "Gold", "India", "China", "AI", "Big Tech", "Big Tech"],
        "TargetPrice": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Thesis": ["", "", "", "", "", "", "", "", ""],
    })


def fallback_options():
    return pd.DataFrame({
        "Underlying": ["HD", "BKNG", "RKLB"],
        "OptionType": ["PUT", "PUT", "CALL"],
        "PositionSide": ["SELL", "SELL", "BUY"],
        "Strike": [320, 3800, 30],
        "Expiry": ["2026-07-17", "2026-07-17", "2026-07-17"],
        "Contracts": [1, 1, 2],
        "EntryPrice": [14.40, 65.00, 2.10],
        "CurrentBid": [14.15, 60.00, 2.15],
        "CurrentMark": [14.15, 62.50, 2.35],
        "CurrentAsk": [16.30, 68.00, 2.55],
        "Delta": [-0.35, -0.28, 0.42],
        "Gamma": [0.03, 0.01, 0.06],
        "Theta": [-0.08, -0.12, -0.04],
        "Vega": [0.22, 0.35, 0.18],
        "IV": [0.32, 0.41, 0.58],
        "Volume": [7000000, 8500000, 1609],
        "OpenInterest": [1609, 32, 210],
        "Status": ["WATCH", "WATCH", "OPEN"],
        "UnderlyingPrice": [370, 3950, 27],
        "Note": ["ตัวอย่าง CSP", "ตัวอย่าง CSP", "ตัวอย่าง CALL"],
    })


# =====================================================
# PRICE / FX
# =====================================================

@st.cache_data(ttl=300)
def download_prices(tickers, start=None, end=None, period=None):
    tickers = [normalize_symbol_for_yfinance(clean_ticker(t)) for t in tickers if clean_ticker(t)]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()

    try:
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
    except Exception:
        return pd.DataFrame()


def get_current_prices(tickers):
    tickers = [
        clean_ticker(t) for t in tickers
        if clean_ticker(t) and clean_ticker(t) not in MANUAL_ONLY_TICKERS and not clean_ticker(t).endswith("80")
    ]
    if not tickers:
        return pd.Series(dtype=float)

    prices = download_prices(tickers, period="7d")
    if prices.empty:
        return pd.Series(dtype=float)

    latest = prices.ffill().iloc[-1]
    latest.index = [clean_ticker(c) for c in latest.index]
    return latest


@st.cache_data(ttl=300)
def get_usdthb_rate():
    prices = download_prices(["USDTHB=X"], period="7d")
    if prices.empty or "USDTHB=X" not in prices.columns:
        return 36.0
    rate = float(prices["USDTHB=X"].ffill().iloc[-1])
    return rate if rate > 0 else 36.0


def fx_to_thb(currency):
    currency = clean_currency(currency)
    if currency == "THB":
        return 1.0
    if currency == "USD":
        return get_usdthb_rate()
    return 1.0


# =====================================================
# LOAD DATA
# =====================================================

def load_portfolio():
    raw = safe_read_tab("portfolio", fallback_portfolio())
    df = normalize_columns(raw)

    rename_map = {}
    mappings = {
        "Broker": ["Broker", "Account", "Platform"],
        "Ticker": ["Ticker", "Symbol", "Code", "Asset", "Stock"],
        "Currency": ["Currency", "CCY"],
        "Quantity": ["Quantity", "Qty", "Shares", "Units", "Amount"],
        "AvgCost": ["AvgCost", "Avg Cost", "Average Cost", "AveragePrice", "Average Price", "Cost"],
        "ManualPrice": ["ManualPrice", "Manual Price", "CurrentPrice", "Current Price", "Price", "MarketPrice", "Market Price"],
    }

    for std_col, candidates in mappings.items():
        found = pick_col(df, candidates)
        if found is not None and found != std_col:
            rename_map[found] = std_col
    df = df.rename(columns=rename_map)

    df = ensure_columns(df, PORTFOLIO_COLUMNS)
    df["Broker"] = df["Broker"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].apply(clean_ticker)
    df["Currency"] = df["Currency"].apply(clean_currency)
    df["Quantity"] = to_number(df["Quantity"])
    df["AvgCost"] = to_number(df["AvgCost"])
    df["ManualPrice"] = to_number(df["ManualPrice"])
    df = df[df["Ticker"] != ""]
    return df


def load_banks():
    raw = safe_read_tab("bank_accounts", fallback_banks())
    df = normalize_columns(raw)

    rename_map = {}
    mappings = {
        "Bank": ["Bank", "Name"],
        "Account": ["Account", "AccountName", "Account Name"],
        "Balance": ["Balance", "Amount", "Value", "Cash", "THBValue", "THB Value"],
        "Currency": ["Currency", "CCY"],
    }
    for std_col, candidates in mappings.items():
        found = pick_col(df, candidates)
        if found is not None and found != std_col:
            rename_map[found] = std_col
    df = df.rename(columns=rename_map)

    df = ensure_columns(df, BANK_COLUMNS)
    df["Currency"] = df["Currency"].apply(clean_currency)
    df["Balance"] = to_number(df["Balance"])
    df["FxRateToTHB"] = df["Currency"].apply(fx_to_thb)
    df["BalanceTHB"] = df["Balance"] * df["FxRateToTHB"]
    return df


def load_properties():
    raw = safe_read_tab("properties", fallback_properties())
    df = normalize_columns(raw)

    rename_map = {}
    mappings = {
        "Property": ["Property", "Name"],
        "Type": ["Type", "AssetType", "Asset Type"],
        "EstimatedValue": ["EstimatedValue", "Estimated Value", "Value", "MarketValue", "Market Value", "Price"],
        "Location": ["Location", "Province"],
    }
    for std_col, candidates in mappings.items():
        found = pick_col(df, candidates)
        if found is not None and found != std_col:
            rename_map[found] = std_col
    df = df.rename(columns=rename_map)

    df = ensure_columns(df, PROPERTY_COLUMNS)
    df["EstimatedValue"] = to_number(df["EstimatedValue"])
    return df


def load_mortgage():
    raw = safe_read_tab("mortgage", fallback_mortgage())
    df = normalize_columns(raw)

    rename_map = {}
    mappings = {
        "Property": ["Property", "Name"],
        "OutstandingDebt": ["OutstandingDebt", "Outstanding Debt", "OutstandingBalance", "Outstanding Balance", "Debt", "Loan", "Principal"],
        "MonthlyPayment": ["MonthlyPayment", "Monthly Payment", "Payment"],
        "InterestRate": ["InterestRate", "Interest Rate", "Rate"],
    }
    for std_col, candidates in mappings.items():
        found = pick_col(df, candidates)
        if found is not None and found != std_col:
            rename_map[found] = std_col
    df = df.rename(columns=rename_map)

    df = ensure_columns(df, MORTGAGE_COLUMNS)
    df["OutstandingDebt"] = to_number(df["OutstandingDebt"])
    df["MonthlyPayment"] = to_number(df["MonthlyPayment"])
    df["InterestRate"] = to_number(df["InterestRate"])
    return df


def load_property_cashflow():
    raw = safe_read_tab("property_cashflow", fallback_cashflow())
    df = normalize_columns(raw)

    rename_map = {}
    mappings = {
        "Property": ["Property", "Name"],
        "Month": ["Month"],
        "Rent": ["Rent", "Income"],
        "Expense": ["Expense", "Cost"],
        "ExtraPayment": ["ExtraPayment", "Extra Payment", "Prepay"],
    }
    for std_col, candidates in mappings.items():
        found = pick_col(df, candidates)
        if found is not None and found != std_col:
            rename_map[found] = std_col
    df = df.rename(columns=rename_map)

    df = ensure_columns(df, CASHFLOW_COLUMNS)
    df["Rent"] = to_number(df["Rent"])
    df["Expense"] = to_number(df["Expense"])
    df["ExtraPayment"] = to_number(df["ExtraPayment"])
    return df


def load_watchlist():
    raw = safe_read_tab("watchlist", fallback_watchlist())
    df = normalize_columns(raw)

    rename_map = {}
    mappings = {
        "Symbol": ["Symbol", "Ticker", "Code", "Asset", "Stock"],
        "Name": ["Name", "Company", "Asset Name"],
        "Theme": ["Theme", "Category", "AssetClass", "Asset Class", "Type"],
        "TargetPrice": ["TargetPrice", "Target Price", "Target", "BuyPrice", "Buy Price"],
        "Thesis": ["Thesis", "Reason", "Note", "Why"],
    }

    for std_col, candidates in mappings.items():
        found = pick_col(df, candidates)
        if found is not None and found != std_col:
            rename_map[found] = std_col
    df = df.rename(columns=rename_map)

    df = ensure_columns(df, WATCHLIST_COLUMNS)
    df["Symbol"] = df["Symbol"].apply(clean_ticker)
    df["TargetPrice"] = to_number(df["TargetPrice"])
    df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    return df



def load_options():
    raw = safe_read_tab("options", fallback_options())
    df = normalize_columns(raw)

    rename_map = {}
    mappings = {
        "Underlying": ["Underlying", "Ticker", "Symbol", "Stock"],
        "OptionType": ["OptionType", "Type", "CallPut", "PutCall", "C/P"],
        "PositionSide": ["PositionSide", "Side", "BuySell", "Buy/Sell", "Action"],
        "Strike": ["Strike", "StrikePrice", "Strike Price"],
        "Expiry": ["Expiry", "Expiration", "ExpirationDate", "Expiration Date"],
        "Contracts": ["Contracts", "Contract", "Qty", "Quantity"],
        "EntryPrice": ["EntryPrice", "Entry Price", "Entry", "Cost", "AvgCost", "Avg Cost"],
        "CurrentBid": ["CurrentBid", "Bid"],
        "CurrentMark": ["CurrentMark", "Mark", "Mid", "MarketPrice"],
        "CurrentAsk": ["CurrentAsk", "Ask"],
        "Delta": ["Delta"],
        "Gamma": ["Gamma"],
        "Theta": ["Theta"],
        "Vega": ["Vega"],
        "IV": ["IV", "ImpliedVolatility", "Implied Volatility"],
        "Volume": ["Volume", "Vol"],
        "OpenInterest": ["OpenInterest", "Open Interest", "OI"],
        "UnderlyingPrice": ["UnderlyingPrice", "Underlying Price", "StockPrice", "Stock Price", "Spot"],
        "Status": ["Status"],
        "Note": ["Note", "Notes"],
    }

    for std_col, candidates in mappings.items():
        found = pick_col(df, candidates)
        if found is not None and found != std_col:
            rename_map[found] = std_col
    df = df.rename(columns=rename_map)

    df = ensure_columns(df, OPTIONS_COLUMNS)
    df["Underlying"] = df["Underlying"].apply(clean_ticker)
    df["OptionType"] = df["OptionType"].astype(str).str.upper().str.strip()
    df["PositionSide"] = df["PositionSide"].astype(str).str.upper().str.strip()
    df.loc[~df["PositionSide"].isin(["BUY", "SELL", "WATCH"]), "PositionSide"] = "SELL"
    for col in ["Strike", "Contracts", "EntryPrice", "CurrentBid", "CurrentMark", "CurrentAsk", "Delta", "Gamma", "Theta", "Vega", "IV", "Volume", "OpenInterest", "UnderlyingPrice"]:
        df[col] = to_number(df[col])

    df = df[df["Underlying"] != ""]
    return df


# =====================================================
# CALCULATIONS
# =====================================================

def calculate_portfolio(portfolio):
    portfolio = portfolio.copy()
    portfolio["Ticker"] = portfolio["Ticker"].apply(clean_ticker)
    portfolio = portfolio[portfolio["Ticker"] != ""]
    portfolio["Currency"] = portfolio["Currency"].apply(clean_currency)
    portfolio["Quantity"] = to_number(portfolio["Quantity"])
    portfolio["AvgCost"] = to_number(portfolio["AvgCost"])
    portfolio["ManualPrice"] = to_number(portfolio["ManualPrice"])

    tickers = portfolio.loc[
        ~portfolio["Ticker"].isin(MANUAL_ONLY_TICKERS) & ~portfolio["Ticker"].str.endswith("80"),
        "Ticker"
    ].dropna().unique().tolist()

    current_prices = get_current_prices(tickers)
    portfolio["YFinanceSymbol"] = portfolio["Ticker"].apply(normalize_symbol_for_yfinance)
    portfolio["YFinancePrice"] = portfolio["YFinanceSymbol"].map(current_prices)
    portfolio.loc[portfolio["Ticker"].isin(["CASH", "CASH THB", "THB CASH", "เงินสด"]), "YFinancePrice"] = 1
    portfolio["YFinancePrice"] = to_number(portfolio["YFinancePrice"])

    portfolio["CurrentPrice"] = portfolio["YFinancePrice"]

    # BTC and other crypto symbols from yfinance are USD-denominated.
    # If the sheet records them as THB assets, convert the yfinance USD quote into THB once,
    # then set FxRateToTHB = 1 to avoid double conversion.
    crypto_usd_price_mask = portfolio["YFinanceSymbol"].astype(str).str.endswith("-USD", na=False)
    crypto_recorded_thb_mask = crypto_usd_price_mask & portfolio["Currency"].eq("THB") & (portfolio["YFinancePrice"] > 0)
    portfolio.loc[crypto_recorded_thb_mask, "CurrentPrice"] = portfolio.loc[crypto_recorded_thb_mask, "YFinancePrice"] * get_usdthb_rate()

    use_manual = (portfolio["CurrentPrice"] == 0) & (portfolio["ManualPrice"] > 0)
    portfolio.loc[use_manual, "CurrentPrice"] = portfolio.loc[use_manual, "ManualPrice"]
    portfolio.loc[portfolio["Ticker"].isin(["CASH", "CASH THB", "THB CASH", "เงินสด"]), "CurrentPrice"] = 1

    portfolio["PriceSource"] = "yfinance"
    portfolio.loc[crypto_recorded_thb_mask, "PriceSource"] = "yfinance USD→THB"
    portfolio.loc[use_manual, "PriceSource"] = "manual"
    portfolio.loc[portfolio["Ticker"].isin(["CASH", "CASH THB", "THB CASH", "เงินสด"]), "PriceSource"] = "cash"
    portfolio.loc[portfolio["CurrentPrice"] == 0, "PriceSource"] = "missing"

    portfolio["FxRateToTHB"] = portfolio["Currency"].apply(fx_to_thb)
    portfolio.loc[crypto_recorded_thb_mask, "FxRateToTHB"] = 1
    portfolio.loc[portfolio["Ticker"].isin(["CASH", "CASH THB", "THB CASH", "เงินสด"]), "FxRateToTHB"] = 1

    portfolio["CostBasisNative"] = portfolio["Quantity"] * portfolio["AvgCost"]
    portfolio["MarketValueNative"] = portfolio["Quantity"] * portfolio["CurrentPrice"]
    portfolio["PnLNative"] = portfolio["MarketValueNative"] - portfolio["CostBasisNative"]

    portfolio["CostBasisTHB"] = portfolio["CostBasisNative"] * portfolio["FxRateToTHB"]
    portfolio["MarketValueTHB"] = portfolio["MarketValueNative"] * portfolio["FxRateToTHB"]
    portfolio["PnLTHB"] = portfolio["MarketValueTHB"] - portfolio["CostBasisTHB"]
    portfolio["IsCash"] = portfolio["Ticker"].isin(["CASH", "CASH THB", "THB CASH", "เงินสด"])

    portfolio["ReturnPct"] = 0.0
    mask = portfolio["CostBasisNative"] != 0
    portfolio.loc[mask, "ReturnPct"] = portfolio.loc[mask, "PnLNative"] / portfolio.loc[mask, "CostBasisNative"] * 100

    return portfolio


def get_portfolio_stats(portfolio_calc):
    cash = portfolio_calc[portfolio_calc["IsCash"]]
    inv = portfolio_calc[~portfolio_calc["IsCash"]]

    portfolio_cash = float(cash["MarketValueTHB"].sum())
    investment_value = float(inv["MarketValueTHB"].sum())
    investment_cost = float(inv["CostBasisTHB"].sum())
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
        "rent": float(cashflow["Rent"].sum()),
        "expense": float(cashflow["Expense"].sum()),
        "mortgage": float(mortgage["MonthlyPayment"].sum()),
        "net": float(cashflow["Rent"].sum() - cashflow["Expense"].sum() - mortgage["MonthlyPayment"].sum()),
    }

    return (
        float(real_estate["EstimatedValue"].sum()),
        float(real_estate["OutstandingDebt"].sum()),
        float(real_estate["Equity"].sum()),
        real_estate,
        cashflow,
        monthly,
    )


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


def calculate_risk_metrics(price_df: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()

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


def calculate_portfolio_level_risk(portfolio_calc, period="1y"):
    inv = portfolio_calc[(~portfolio_calc["IsCash"]) & (portfolio_calc["MarketValueTHB"] > 0)].copy()
    inv = inv[~inv["Ticker"].isin(MANUAL_ONLY_TICKERS)]
    inv = inv[~inv["Ticker"].str.endswith("80")]

    if inv.empty:
        return {}, pd.DataFrame()

    tickers = inv["Ticker"].tolist()
    prices = download_prices(tickers, period=period)

    if prices.empty:
        return {}, prices

    usable = [c for c in prices.columns if c in [normalize_symbol_for_yfinance(t) for t in tickers] or c in tickers]
    prices = prices[usable].dropna(axis=1, how="all")
    if prices.empty:
        return {}, prices

    weights = {}
    total = inv["MarketValueTHB"].sum()
    for _, row in inv.iterrows():
        y = clean_ticker(normalize_symbol_for_yfinance(row["Ticker"]))
        if y in prices.columns and total > 0:
            weights[y] = row["MarketValueTHB"] / total
        elif row["Ticker"] in prices.columns and total > 0:
            weights[row["Ticker"]] = row["MarketValueTHB"] / total

    if not weights:
        return {}, prices

    returns = prices.pct_change().dropna()
    returns = returns[[c for c in returns.columns if c in weights]]
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    portfolio_returns = returns.dot(w)

    if portfolio_returns.empty:
        return {}, prices

    nav = (1 + portfolio_returns).cumprod()
    total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1 / 365.25)
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) * 100
    volatility = portfolio_returns.std() * np.sqrt(252) * 100
    sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() != 0 else 0
    max_drawdown = (nav / nav.cummax() - 1).min() * 100

    metrics = {
        "Portfolio Total Return": total_return,
        "Portfolio CAGR": cagr,
        "Portfolio Volatility": volatility,
        "Portfolio Sharpe": sharpe,
        "Portfolio Max Drawdown": max_drawdown,
    }

    return metrics, prices



# =====================================================
# OPTION CANDIDATE SCREENER
# =====================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(price: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Close": price.dropna()})
    if df.empty:
        return df

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"] = df["MACD"] - df["MACDSignal"]

    ema8 = df["Close"].ewm(span=8, adjust=False).mean()
    ema21 = df["Close"].ewm(span=21, adjust=False).mean()
    df["MACDShort"] = ema8 - ema21
    df["MACDShortSignal"] = df["MACDShort"].ewm(span=5, adjust=False).mean()

    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["High252"] = df["Close"].rolling(252, min_periods=30).max()
    df["PctFromHigh252"] = np.where(df["High252"] > 0, (df["Close"] / df["High252"] - 1) * 100, 0)
    return df


@st.cache_data(ttl=1800)
def get_yfinance_info(symbol: str) -> dict:
    try:
        info = yf.Ticker(normalize_symbol_for_yfinance(symbol)).info or {}
        return {
            "marketCap": info.get("marketCap", np.nan),
            "forwardPE": info.get("forwardPE", np.nan),
            "trailingEps": info.get("trailingEps", np.nan),
            "forwardEps": info.get("forwardEps", np.nan),
            "targetMeanPrice": info.get("targetMeanPrice", np.nan),
            "recommendationMean": info.get("recommendationMean", np.nan),
        }
    except Exception:
        return {}


def fair_value_lite(symbol: str, current_price: float) -> dict:
    info = get_yfinance_info(symbol)
    target = info.get("targetMeanPrice", np.nan)
    forward_eps = info.get("forwardEps", np.nan)
    trailing_eps = info.get("trailingEps", np.nan)
    eps = forward_eps if pd.notna(forward_eps) and forward_eps > 0 else trailing_eps

    fair_pe = 22
    pe_fv = eps * fair_pe if pd.notna(eps) and eps > 0 else np.nan

    if pd.notna(target) and target > 0 and pd.notna(pe_fv) and pe_fv > 0:
        fv = (target + pe_fv) / 2
        source = "Analyst target + EPS x PE"
    elif pd.notna(target) and target > 0:
        fv = target
        source = "Analyst target"
    elif pd.notna(pe_fv) and pe_fv > 0:
        fv = pe_fv
        source = "EPS x PE"
    else:
        fv = np.nan
        source = "N/A"

    mos = (fv / current_price - 1) * 100 if pd.notna(fv) and current_price > 0 else 0

    if mos >= 15:
        fair_score = 10
    elif mos >= 5:
        fair_score = 7
    elif mos >= -5:
        fair_score = 5
    else:
        fair_score = 2

    return {
        "FairValue": fv,
        "FairValueSource": source,
        "MarginSafety%": mos,
        "FairValueScore": fair_score,
        "ForwardPE": info.get("forwardPE", np.nan),
        "MarketCap": info.get("marketCap", np.nan),
        "AnalystTarget": target,
    }


def analyze_trend_for_symbol(symbol: str, period: str = "2y") -> dict:
    prices = download_prices([symbol], period=period)
    yf_symbol = clean_ticker(normalize_symbol_for_yfinance(symbol))

    if prices.empty:
        return {}

    series = prices[yf_symbol].dropna() if yf_symbol in prices.columns else prices.iloc[:, 0].dropna()
    tech = add_technical_indicators(series)

    if tech.empty or len(tech) < 60:
        return {}

    last = tech.iloc[-1]
    prev = tech.iloc[-2] if len(tech) >= 2 else last

    close = float(last["Close"])
    rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan

    short_score = 0
    short_score += 1 if last["Close"] > last["EMA20"] else 0
    short_score += 1 if last["MACDShort"] > last["MACDShortSignal"] else 0
    short_score += 1 if pd.notna(rsi) and rsi > 50 else 0

    medium_score = 0
    medium_score += 1 if last["Close"] > last["EMA50"] else 0
    medium_score += 1 if last["EMA20"] > last["EMA50"] else 0
    medium_score += 1 if last["MACD"] > 0 else 0

    long_score = 0
    if pd.notna(last["EMA200"]):
        long_score += 1 if last["Close"] > last["EMA200"] else 0
        long_score += 1 if last["EMA50"] > last["EMA200"] else 0
    long_score += 1 if last["PctFromHigh252"] >= -15 else 0

    trend_score = ((short_score / 3) * 0.20 + (medium_score / 3) * 0.30 + (long_score / 3) * 0.50) * 10

    momentum_1m = (series.iloc[-1] / series.iloc[-21] - 1) * 100 if len(series) > 21 else 0
    momentum_3m = (series.iloc[-1] / series.iloc[-63] - 1) * 100 if len(series) > 63 else 0
    momentum_6m = (series.iloc[-1] / series.iloc[-126] - 1) * 100 if len(series) > 126 else 0
    momentum_12m = (series.iloc[-1] / series.iloc[-252] - 1) * 100 if len(series) > 252 else 0

    macd_cross_today = bool((last["MACD"] > last["MACDSignal"]) and (prev["MACD"] <= prev["MACDSignal"]))
    signal_today = "🟢 MACD Bullish Cross" if macd_cross_today else ""

    return {
        "Symbol": clean_ticker(symbol),
        "Price": close,
        "EMA20": float(last["EMA20"]),
        "EMA50": float(last["EMA50"]),
        "EMA200": float(last["EMA200"]) if pd.notna(last["EMA200"]) else np.nan,
        "RSI14": rsi,
        "MACD": float(last["MACD"]),
        "MACDSignal": float(last["MACDSignal"]),
        "MACDHist": float(last["MACDHist"]),
        "ShortTrend": short_score,
        "MediumTrend": medium_score,
        "LongTrend": long_score,
        "TrendScore": trend_score,
        "Momentum1M%": momentum_1m,
        "Momentum3M%": momentum_3m,
        "Momentum6M%": momentum_6m,
        "Momentum12M%": momentum_12m,
        "PctFromHigh252%": float(last["PctFromHigh252"]),
        "SignalToday": signal_today,
    }


def get_candidate_universe(portfolio_calc: pd.DataFrame, watchlist: pd.DataFrame, extra_symbols: list | None = None) -> list:
    portfolio_symbols = portfolio_calc.loc[~portfolio_calc["IsCash"], "Ticker"].dropna().astype(str).tolist()
    watch_symbols = watchlist["Symbol"].dropna().astype(str).tolist() if not watchlist.empty else []
    symbols = portfolio_symbols + watch_symbols + (extra_symbols or DEFAULT_SCAN_UNIVERSE)

    cleaned = []
    for s in symbols:
        s = clean_ticker(s)
        if not s or s in MANUAL_ONLY_TICKERS or s.endswith("80"):
            continue
        cleaned.append(s)

    return sorted(list(dict.fromkeys(cleaned)))


def build_option_candidate_screener(symbols: list, max_symbols: int = 80) -> pd.DataFrame:
    rows = []
    for symbol in symbols[:max_symbols]:
        trend = analyze_trend_for_symbol(symbol)
        if not trend:
            continue

        fv = fair_value_lite(symbol, trend["Price"])
        row = {**trend, **fv}

        momentum_score = 0
        momentum_score += 2.5 if row["Momentum1M%"] > 0 else 0
        momentum_score += 2.5 if row["Momentum3M%"] > 0 else 0
        momentum_score += 2.5 if row["Momentum6M%"] > 0 else 0
        momentum_score += 2.5 if row["Momentum12M%"] > 0 else 0
        row["MomentumScore"] = momentum_score

        risk_score = 10
        if row["RSI14"] > 80:
            risk_score -= 3
        elif row["RSI14"] > 70:
            risk_score -= 1.5
        if row["PctFromHigh252%"] < -25:
            risk_score -= 3
        row["RiskScore"] = max(risk_score, 0)

        row["TotalScore"] = (
            row["TrendScore"] * 0.40
            + row["FairValueScore"] * 0.30
            + row["MomentumScore"] * 0.20
            + row["RiskScore"] * 0.10
        )

        if row["TotalScore"] >= 9:
            row["Conviction"] = "🟢 High"
        elif row["TotalScore"] >= 8:
            row["Conviction"] = "🟢 Good"
        elif row["TotalScore"] >= 7:
            row["Conviction"] = "🟡 Watch"
        else:
            row["Conviction"] = "🔴 Avoid"

        if row["TrendScore"] >= 8 and row["MarginSafety%"] >= 5:
            row["SuggestedSetup"] = "CSP"
        elif row["TrendScore"] >= 8 and row["MarginSafety%"] < 5:
            row["SuggestedSetup"] = "LEAPS / Wait Pullback"
        elif row["TrendScore"] >= 7:
            row["SuggestedSetup"] = "Watch"
        else:
            row["SuggestedSetup"] = "Avoid"

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("TotalScore", ascending=False)



def apply_manual_target_prices(candidates: pd.DataFrame, watchlist_df: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or watchlist_df is None or watchlist_df.empty:
        return candidates

    df = candidates.copy()
    target_map = {}
    thesis_map = {}
    theme_map = {}
    name_map = {}

    for _, row in watchlist_df.iterrows():
        symbol = clean_ticker(row.get("Symbol", ""))
        if not symbol:
            continue
        target_map[symbol] = float(row.get("TargetPrice", 0) or 0)
        thesis_map[symbol] = str(row.get("Thesis", "") or "")
        theme_map[symbol] = str(row.get("Theme", "") or "")
        name_map[symbol] = str(row.get("Name", "") or "")

    df["ManualTarget"] = df["Symbol"].map(target_map).fillna(0)
    df["Theme"] = df["Symbol"].map(theme_map).fillna("")
    df["Name"] = df["Symbol"].map(name_map).fillna("")
    df["Thesis"] = df["Symbol"].map(thesis_map).fillna("")

    manual_mask = df["ManualTarget"] > 0
    df.loc[manual_mask, "FairValue"] = df.loc[manual_mask, "ManualTarget"]
    df.loc[manual_mask, "AnalystTarget"] = df.loc[manual_mask, "ManualTarget"]
    df.loc[manual_mask, "FairValueSource"] = "Manual TargetPrice from Google Sheet"

    df["FairValueDisplay"] = df["FairValue"].apply(lambda x: "N/A" if pd.isna(x) or float(x) <= 0 else f"{float(x):,.2f}")
    df["AnalystTargetDisplay"] = df["AnalystTarget"].apply(lambda x: "N/A" if pd.isna(x) or float(x) <= 0 else f"{float(x):,.2f}")
    df["ForwardPEDisplay"] = df["ForwardPE"].apply(lambda x: "N/A" if pd.isna(x) or float(x) <= 0 else f"{float(x):,.2f}")

    valid_fv = df["FairValue"].notna() & (df["FairValue"] > 0) & (df["Price"] > 0)
    df.loc[valid_fv, "MarginSafety%"] = (df.loc[valid_fv, "FairValue"] / df.loc[valid_fv, "Price"] - 1) * 100

    df["FairValueScore"] = np.select(
        [
            df["MarginSafety%"] >= 15,
            df["MarginSafety%"] >= 5,
            df["MarginSafety%"] >= -5,
        ],
        [10, 7, 5],
        default=2,
    )

    # If there is no fair value data at all, make it neutral rather than misleadingly bearish.
    no_fv = ~valid_fv
    df.loc[no_fv, "FairValueScore"] = 5
    df.loc[no_fv, "FairValueSource"] = df.loc[no_fv, "FairValueSource"].replace("", "N/A")

    df["TotalScore"] = (
        df["TrendScore"] * 0.40
        + df["FairValueScore"] * 0.30
        + df["MomentumScore"] * 0.20
        + df["RiskScore"] * 0.10
    )

    df["Conviction"] = np.select(
        [
            df["TotalScore"] >= 9,
            df["TotalScore"] >= 8,
            df["TotalScore"] >= 7,
        ],
        ["🟢 High", "🟢 Good", "🟡 Watch"],
        default="🔴 Avoid",
    )

    df["SuggestedSetup"] = np.select(
        [
            (df["TrendScore"] >= 8) & (df["MarginSafety%"] >= 5),
            (df["TrendScore"] >= 8) & (df["MarginSafety%"] < 5),
            df["TrendScore"] >= 7,
        ],
        ["CSP", "LEAPS / Wait Pullback", "Watch"],
        default="Avoid",
    )

    return df.sort_values("TotalScore", ascending=False)



def ensure_candidate_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make Auto Watchlist robust when some optional columns are missing."""
    if df is None or df.empty:
        return df

    df = df.copy()

    defaults = {
        "Name": "",
        "Theme": "",
        "Thesis": "",
        "Conviction": "",
        "SuggestedSetup": "",
        "SignalToday": "",
        "FairValueSource": "N/A",
        "FairValue": np.nan,
        "AnalystTarget": np.nan,
        "ForwardPE": np.nan,
        "MarginSafety%": 0.0,
        "FairValueScore": 5.0,
        "TotalScore": 0.0,
        "TrendScore": 0.0,
        "MomentumScore": 0.0,
        "RiskScore": 0.0,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    def _display_number(x):
        try:
            if pd.isna(x) or float(x) <= 0:
                return "N/A"
            return f"{float(x):,.2f}"
        except Exception:
            return "N/A"

    if "FairValueDisplay" not in df.columns:
        df["FairValueDisplay"] = df["FairValue"].apply(_display_number)
    if "AnalystTargetDisplay" not in df.columns:
        df["AnalystTargetDisplay"] = df["AnalystTarget"].apply(_display_number)
    if "ForwardPEDisplay" not in df.columns:
        df["ForwardPEDisplay"] = df["ForwardPE"].apply(_display_number)

    return df


def existing_columns(df: pd.DataFrame, cols: list) -> list:
    """Return only columns that exist in df, preserving order."""
    if df is None or df.empty:
        return []
    return [c for c in cols if c in df.columns]



# =====================================================
# NEWS
# =====================================================

@st.cache_data(ttl=1800)
def fetch_google_news_rss(query: str, max_items: int = 5) -> list:
    try:
        q = quote_plus(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = []

        for item in root.findall("./channel/item")[:max_items]:
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            source_node = item.find("source")
            source = source_node.text.strip() if source_node is not None and source_node.text else ""

            items.append({
                "title": title,
                "source": source,
                "published": pub_date,
                "link": link,
            })

        return items
    except Exception:
        return []


def build_news_query(symbol: str, name: str = "") -> str:
    symbol = clean_ticker(symbol)
    name = str(name).strip()

    manual_queries = {
        "BTC": "Bitcoin BTC",
        "BTC-USD": "Bitcoin BTC",
        "MTS-GOLD": "gold price gold market",
        "GLD": "gold ETF gold price",
        "GC=F": "gold futures price",
        "BRKB80": "Berkshire Hathaway BRK.B",
        "BRK.B": "Berkshire Hathaway BRK.B",
        "BRK-B": "Berkshire Hathaway BRK.B",
        "K-USXNDQ-A(A)": "Nasdaq 100 QQQ",
        "CASH": "Federal Reserve interest rates US dollar",
    }

    if symbol in manual_queries:
        return manual_queries[symbol]

    if name:
        return f"{symbol} {name} stock"

    return f"{symbol} stock"


def get_direct_news_watchlist(symbols: list, watchlist_df: pd.DataFrame, max_symbols: int = 25, max_items: int = 3) -> pd.DataFrame:
    name_map = {}
    theme_map = {}
    if watchlist_df is not None and not watchlist_df.empty:
        for _, row in watchlist_df.iterrows():
            s = clean_ticker(row.get("Symbol", ""))
            if s:
                name_map[s] = str(row.get("Name", "")).strip()
                theme_map[s] = str(row.get("Theme", "")).strip()

    rows = []
    for symbol in symbols[:max_symbols]:
        symbol = clean_ticker(symbol)
        if not symbol or symbol in MANUAL_ONLY_TICKERS or symbol.endswith("80"):
            continue

        name = name_map.get(symbol, "")
        theme = theme_map.get(symbol, "")
        query = build_news_query(symbol, name)
        items = fetch_google_news_rss(query, max_items=max_items)

        headlines = " | ".join([item.get("title", "") for item in items])
        sources = ", ".join(sorted(list({item.get("source", "") for item in items if item.get("source", "")})))
        latest_time = items[0].get("published", "") if items else ""
        news_count = len(items)

        if news_count >= 3:
            heat = "🔥 Hot"
            score = 3
        elif news_count == 2:
            heat = "🟡 Active"
            score = 2
        elif news_count == 1:
            heat = "🔵 Watch"
            score = 1
        else:
            heat = "⚪ Quiet"
            score = 0

        rows.append({
            "Symbol": symbol,
            "Name": name,
            "Theme": theme,
            "NewsType": "Direct",
            "NewsHeat": heat,
            "NewsScore": score,
            "LatestTime": latest_time,
            "Sources": sources,
            "Headlines": headlines,
            "SearchQuery": query,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["NewsScore", "Symbol"], ascending=[False, True])


def get_indirect_news_watchlist(max_items: int = 3) -> pd.DataFrame:
    macro_themes = [
        {
            "MacroTheme": "US-Iran / Middle East conflict",
            "SearchQuery": "US Iran conflict oil prices gold market stocks",
            "AffectedAssets": "Oil, XLE, XOM, CVX, GLD, Defense/ITA, BTC",
            "Reason": "ความเสี่ยงภูมิรัฐศาสตร์มักกระทบน้ำมัน ทอง หุ้นพลังงาน หุ้นกลาโหม และสินทรัพย์เสี่ยง",
        },
        {
            "MacroTheme": "Fed rate / US yields",
            "SearchQuery": "Federal Reserve rate cut Treasury yield Nasdaq gold",
            "AffectedAssets": "QQQ, SPY, GLD, BTC, Banks, TLT",
            "Reason": "ดอกเบี้ยและ bond yield กระทบ valuation หุ้น growth, ทอง, Bitcoin และธนาคาร",
        },
        {
            "MacroTheme": "AI chip cycle",
            "SearchQuery": "AI chip demand Nvidia AMD semiconductor stocks",
            "AffectedAssets": "NVDA, AMD, AVGO, SOXX, TSM",
            "Reason": "ข่าวชิป AI ส่งผลต่อหุ้น semiconductor และหุ้นที่อยู่ใน supply chain",
        },
        {
            "MacroTheme": "China stimulus / China economy",
            "SearchQuery": "China stimulus economy stocks ETF",
            "AffectedAssets": "MCHI, BABA, JD, Emerging Markets, Commodities",
            "Reason": "นโยบายจีนกระทบหุ้นจีน ตลาดเกิดใหม่ และสินค้าโภคภัณฑ์",
        },
        {
            "MacroTheme": "India growth / travel",
            "SearchQuery": "India economy travel demand MakeMyTrip stock",
            "AffectedAssets": "INDA, MMYT, India consumer/travel",
            "Reason": "เศรษฐกิจและการเดินทางในอินเดียกระทบหุ้นธีม India growth",
        },
        {
            "MacroTheme": "Crypto regulation / Bitcoin ETF flow",
            "SearchQuery": "Bitcoin ETF inflow crypto regulation market",
            "AffectedAssets": "BTC-USD, COIN, Crypto-related stocks",
            "Reason": "เงินไหลเข้า ETF และกฎเกณฑ์คริปโตกระทบ Bitcoin และหุ้นเกี่ยวข้อง",
        },
    ]

    rows = []
    for theme in macro_themes:
        items = fetch_google_news_rss(theme["SearchQuery"], max_items=max_items)
        headlines = " | ".join([item.get("title", "") for item in items])
        sources = ", ".join(sorted(list({item.get("source", "") for item in items if item.get("source", "")})))
        latest_time = items[0].get("published", "") if items else ""
        news_count = len(items)

        if news_count >= 3:
            heat = "🔥 Hot"
            score = 3
        elif news_count == 2:
            heat = "🟡 Active"
            score = 2
        elif news_count == 1:
            heat = "🔵 Watch"
            score = 1
        else:
            heat = "⚪ Quiet"
            score = 0

        rows.append({
            "MacroTheme": theme["MacroTheme"],
            "NewsType": "Indirect",
            "NewsHeat": heat,
            "NewsScore": score,
            "AffectedAssets": theme["AffectedAssets"],
            "Reason": theme["Reason"],
            "LatestTime": latest_time,
            "Sources": sources,
            "Headlines": headlines,
            "SearchQuery": theme["SearchQuery"],
        })

    return pd.DataFrame(rows).sort_values(["NewsScore", "MacroTheme"], ascending=[False, True])




# =====================================================
# OPTION CHAIN / OPTION CALCULATOR HELPERS
# =====================================================

@st.cache_data(ttl=900)
def get_option_expirations(symbol: str) -> list:
    """Return available expirations from yfinance. Empty list if not available."""
    try:
        yf_symbol = normalize_symbol_for_yfinance(clean_ticker(symbol))
        return list(yf.Ticker(yf_symbol).options or [])
    except Exception:
        return []


@st.cache_data(ttl=900)
def get_option_chain(symbol: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return calls, puts for a symbol/expiry from yfinance."""
    try:
        yf_symbol = normalize_symbol_for_yfinance(clean_ticker(symbol))
        chain = yf.Ticker(yf_symbol).option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        calls["OptionType"] = "CALL"
        puts["OptionType"] = "PUT"
        calls["Expiry"] = expiry
        puts["Expiry"] = expiry
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def parse_expiry_date(expiry: str):
    try:
        return pd.to_datetime(expiry).date()
    except Exception:
        return None


def pick_expiries_by_dte(expiries: list, min_dte: int = 30, max_dte: int = 75, limit: int = 5) -> list:
    today_d = date.today()
    rows = []
    for e in expiries:
        d = parse_expiry_date(e)
        if d is None:
            continue
        dte = (d - today_d).days
        if min_dte <= dte <= max_dte:
            rows.append((e, dte))
    rows = sorted(rows, key=lambda x: abs(x[1] - 49))
    return [e for e, _ in rows[:limit]]


def enrich_option_chain(df: pd.DataFrame, underlying_price: float, option_type: str, expiry: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["strike", "lastPrice", "bid", "ask", "change", "percentChange", "volume", "openInterest", "impliedVolatility"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = to_number(out[col])

    out["UnderlyingPrice"] = float(underlying_price or 0)
    out["OptionType"] = option_type
    out["Expiry"] = expiry
    expiry_date = parse_expiry_date(expiry)
    out["DTE"] = (expiry_date - date.today()).days if expiry_date else 0
    out["Mid"] = np.where((out["bid"] > 0) & (out["ask"] > 0), (out["bid"] + out["ask"]) / 2, out["lastPrice"])
    out["Spread"] = out["ask"] - out["bid"]
    out["SpreadPct"] = np.where(out["Mid"] > 0, out["Spread"] / out["Mid"] * 100, 0)
    out["MoneynessPct"] = np.where(out["strike"] > 0, (underlying_price / out["strike"] - 1) * 100, 0)

    # For CSP: premium received / required cash
    out["NetAssignPrice"] = out["strike"] - out["Mid"]
    out["RequiredCash"] = out["strike"] * 100
    out["PremiumIncome"] = out["Mid"] * 100
    out["PremiumReturnPct"] = np.where(out["RequiredCash"] > 0, out["PremiumIncome"] / out["RequiredCash"] * 100, 0)
    out["AnnualizedReturnPct"] = np.where(out["DTE"] > 0, out["PremiumReturnPct"] * 365 / out["DTE"], 0)
    out["DownsideBufferPct"] = np.where(underlying_price > 0, (underlying_price - out["NetAssignPrice"]) / underlying_price * 100, 0)

    # For long call: break-even and leverage proxy
    out["Breakeven"] = out["strike"] + out["Mid"]
    out["BreakevenMovePct"] = np.where(underlying_price > 0, (out["Breakeven"] / underlying_price - 1) * 100, 0)
    out["ContractCost"] = out["Mid"] * 100
    out["LeverageNotional"] = np.where(out["ContractCost"] > 0, (underlying_price * 100) / out["ContractCost"], 0)

    # yfinance usually does not provide delta. Use rough strike-distance buckets.
    out["DeltaZone"] = np.where(
        option_type == "PUT",
        np.select(
            [out["MoneynessPct"] > 15, out["MoneynessPct"] > 8, out["MoneynessPct"] > 3, out["MoneynessPct"] > -3],
            ["Very OTM", "OTM", "Near OTM", "ATM"],
            default="ITM"
        ),
        np.select(
            [out["BreakevenMovePct"] < 3, out["BreakevenMovePct"] < 8, out["BreakevenMovePct"] < 15],
            ["ATM/Near", "Moderate", "Far"],
            default="Very Far"
        ),
    )

    return out


def build_csp_candidates_from_yfinance(symbols: list, min_dte: int = 30, max_dte: int = 75, max_symbols: int = 20) -> pd.DataFrame:
    rows = []
    prices = get_current_prices(symbols[:max_symbols])
    for symbol in symbols[:max_symbols]:
        symbol = clean_ticker(symbol)
        yf_symbol = clean_ticker(normalize_symbol_for_yfinance(symbol))
        underlying_price = float(prices.get(yf_symbol, prices.get(symbol, 0)) or 0)
        if underlying_price <= 0:
            continue
        expiries = pick_expiries_by_dte(get_option_expirations(symbol), min_dte, max_dte, limit=3)
        trend = analyze_trend_for_symbol(symbol) or {}
        fv = fair_value_lite(symbol, underlying_price)
        for expiry in expiries:
            _, puts = get_option_chain(symbol, expiry)
            puts = enrich_option_chain(puts, underlying_price, "PUT", expiry)
            if puts.empty:
                continue
            # prefer OTM puts: net assign below current price, liquid-ish, positive premium
            p = puts[(puts["Mid"] > 0) & (puts["strike"] < underlying_price) & (puts["openInterest"] >= 10)].copy()
            if p.empty:
                p = puts[(puts["Mid"] > 0) & (puts["strike"] < underlying_price)].copy()
            if p.empty:
                continue
            p["Symbol"] = symbol
            p["TrendScore"] = trend.get("TrendScore", np.nan)
            p["RSI14"] = trend.get("RSI14", np.nan)
            p["SignalToday"] = trend.get("SignalToday", "")
            p["FairValue"] = fv.get("FairValue", np.nan)
            p["MarginSafetyPct"] = fv.get("MarginSafety%", 0)
            # Score: trend + annualized premium + buffer + liquidity, penalize wide spreads/too hot RSI
            p["CSPScore"] = (
                np.nan_to_num(p["TrendScore"], nan=5) * 0.35
                + np.clip(p["AnnualizedReturnPct"], 0, 40) / 4 * 0.25
                + np.clip(p["DownsideBufferPct"], 0, 25) / 2.5 * 0.20
                + np.clip(np.log1p(p["openInterest"]), 0, 8) / 8 * 10 * 0.10
                + np.clip(np.log1p(p["volume"]), 0, 8) / 8 * 10 * 0.10
            )
            p.loc[p["SpreadPct"] > 30, "CSPScore"] -= 1.5
            p.loc[p["RSI14"] > 80, "CSPScore"] -= 1.0
            p["CSPScore"] = p["CSPScore"].clip(0, 10)
            rows.append(p)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["CSPScore", "AnnualizedReturnPct", "DownsideBufferPct"], ascending=False)


def build_long_call_candidates_from_yfinance(symbols: list, min_dte: int = 90, max_dte: int = 540, max_symbols: int = 20) -> pd.DataFrame:
    rows = []
    prices = get_current_prices(symbols[:max_symbols])
    for symbol in symbols[:max_symbols]:
        symbol = clean_ticker(symbol)
        yf_symbol = clean_ticker(normalize_symbol_for_yfinance(symbol))
        underlying_price = float(prices.get(yf_symbol, prices.get(symbol, 0)) or 0)
        if underlying_price <= 0:
            continue
        expiries = pick_expiries_by_dte(get_option_expirations(symbol), min_dte, max_dte, limit=4)
        trend = analyze_trend_for_symbol(symbol) or {}
        fv = fair_value_lite(symbol, underlying_price)
        for expiry in expiries:
            calls, _ = get_option_chain(symbol, expiry)
            calls = enrich_option_chain(calls, underlying_price, "CALL", expiry)
            if calls.empty:
                continue
            c = calls[(calls["Mid"] > 0) & (calls["strike"] >= underlying_price * 0.85) & (calls["strike"] <= underlying_price * 1.20)].copy()
            if c.empty:
                continue
            c["Symbol"] = symbol
            c["TrendScore"] = trend.get("TrendScore", np.nan)
            c["RSI14"] = trend.get("RSI14", np.nan)
            c["SignalToday"] = trend.get("SignalToday", "")
            c["FairValue"] = fv.get("FairValue", np.nan)
            c["MarginSafetyPct"] = fv.get("MarginSafety%", 0)
            c["CallScore"] = (
                np.nan_to_num(c["TrendScore"], nan=5) * 0.45
                + np.clip(c["MarginSafetyPct"], -20, 60) / 60 * 10 * 0.20
                + np.clip(20 - c["BreakevenMovePct"], 0, 20) / 20 * 10 * 0.20
                + np.clip(np.log1p(c["openInterest"]), 0, 8) / 8 * 10 * 0.15
            )
            c.loc[c["SpreadPct"] > 30, "CallScore"] -= 1.5
            c.loc[c["RSI14"] > 80, "CallScore"] -= 1.0
            c["CallScore"] = c["CallScore"].clip(0, 10)
            rows.append(c)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["CallScore", "TrendScore", "BreakevenMovePct"], ascending=[False, False, True])


# =====================================================
# LOAD ALL DATA ONCE
# =====================================================

portfolio_raw = load_portfolio()
banks = load_banks()
properties = load_properties()
mortgage = load_mortgage()
cashflow = load_property_cashflow()
watchlist = load_watchlist()
options_df = load_options()

portfolio_calc = calculate_portfolio(portfolio_raw)
portfolio_stats = get_portfolio_stats(portfolio_calc)
property_value, property_debt, property_equity, real_estate, cashflow_calc, monthly = get_real_estate_summary(properties, mortgage, cashflow)

bank_cash = float(banks["BalanceTHB"].sum())
total_cash = float(portfolio_stats["portfolio_cash"] + bank_cash)
investment_value = float(portfolio_stats["investment_value"])
net_worth = float(total_cash + investment_value + property_equity)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📊 Investment Dashboard")
st.sidebar.caption("Version 2 - Investment Command Center")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.metric("Net Worth", format_thb(net_worth))
st.sidebar.metric("Investments", format_thb(investment_value))
st.sidebar.metric("Cash", format_thb(total_cash))
st.sidebar.metric("Property Equity", format_thb(property_equity))


# =====================================================
# TABS
# =====================================================

tab_wealth, tab_portfolio, tab_news, tab_macro, tab_watchlist, tab_option_calc, tab_options, tab_market = st.tabs([
    "💰 My Wealth",
    "📈 Portfolio Dashboard",
    "📰 Portfolio News",
    "🌍 Macro Dashboard",
    "👀 Watchlist",
    "🧮 Option Calculator",
    "🧨 Options War Room",
    "🔎 Market Analysis",
])


# =====================================================
# TAB 1: MY WEALTH
# =====================================================

with tab_wealth:
    st.header("💰 My Wealth")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Worth", format_thb(net_worth))
    c2.metric("Total Cash", format_thb(total_cash))
    c3.metric("Investments", format_thb(investment_value))
    c4.metric("Property Equity", format_thb(property_equity))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Property Value", format_thb(property_value))
    c6.metric("Property Debt", format_thb(property_debt))
    c7.metric("Investment P/L", format_thb(portfolio_stats["investment_pnl"]))
    c8.metric("Investment Return", pct(portfolio_stats["investment_return_pct"]))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Monthly Rent", format_thb(monthly["rent"]))
    c10.metric("Monthly Expense", format_thb(monthly["expense"]))
    c11.metric("Monthly Mortgage", format_thb(monthly["mortgage"]))
    c12.metric("Monthly Net Cashflow", format_thb(monthly["net"]))

    allocation = pd.DataFrame({
        "Category": ["Cash", "Investments", "Property Equity"],
        "ValueTHB": [total_cash, investment_value, property_equity],
    })

    debt = pd.DataFrame({
        "Category": ["Cash", "Investments", "Property Gross Value", "Property Debt", "Net Worth"],
        "ValueTHB": [total_cash, investment_value, property_value, -property_debt, net_worth],
    })

    c1, c2 = st.columns(2)
    with c1:
        if allocation["ValueTHB"].sum() != 0:
            st.plotly_chart(px.pie(allocation, names="Category", values="ValueTHB", title="Net Worth Allocation"), use_container_width=True)

    with c2:
        st.plotly_chart(px.bar(debt, x="Category", y="ValueTHB", title="Assets, Debt, and Net Worth"), use_container_width=True)

    st.divider()
    st.subheader("Investment Portfolio")
    st.caption(f"Base Currency = THB | USD/THB = {get_usdthb_rate():,.4f}")
    summary_cols = ["Broker", "Ticker", "Currency", "Quantity", "AvgCost", "CurrentPrice", "PriceSource", "MarketValueTHB", "PnLTHB", "ReturnPct"]
    st.dataframe(portfolio_calc[summary_cols].round(2), use_container_width=True, hide_index=True)

    st.subheader("Bank Accounts")
    st.dataframe(banks.round(2), use_container_width=True, hide_index=True)

    st.subheader("Real Estate")
    st.dataframe(real_estate.round(2), use_container_width=True, hide_index=True)

    st.subheader("Mortgage / Debt")
    st.dataframe(mortgage.round(2), use_container_width=True, hide_index=True)

    missing = portfolio_calc[(portfolio_calc["CurrentPrice"] == 0) & (~portfolio_calc["IsCash"])]
    if not missing.empty:
        st.warning("ยังไม่มีราคาสำหรับ: " + ", ".join(missing["Ticker"].unique()) + " — ให้ใส่ ManualPrice ใน Google Sheet")


# =====================================================
# TAB 2: PORTFOLIO DASHBOARD
# =====================================================

with tab_portfolio:
    st.header("📈 Portfolio Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", format_thb(portfolio_stats["portfolio_value"]))
    c2.metric("Investment Value", format_thb(portfolio_stats["investment_value"]))
    c3.metric("Investment P/L", format_thb(portfolio_stats["investment_pnl"]))
    c4.metric("Investment Return", pct(portfolio_stats["investment_return_pct"]))

    st.subheader("Holdings")
    summary_cols = ["Broker", "Ticker", "Currency", "Quantity", "AvgCost", "CurrentPrice", "PriceSource", "MarketValueNative", "MarketValueTHB", "PnLTHB", "ReturnPct"]
    st.dataframe(portfolio_calc[summary_cols].round(2).sort_values("MarketValueTHB", ascending=False), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        broker_summary = portfolio_calc.groupby("Broker", dropna=False)["MarketValueTHB"].sum().reset_index()
        if broker_summary["MarketValueTHB"].sum() != 0:
            st.plotly_chart(px.pie(broker_summary, names="Broker", values="MarketValueTHB", title="Allocation by Broker"), use_container_width=True)

    with c2:
        ticker_summary = portfolio_calc.groupby("Ticker", dropna=False)["MarketValueTHB"].sum().reset_index()
        if ticker_summary["MarketValueTHB"].sum() != 0:
            st.plotly_chart(px.pie(ticker_summary, names="Ticker", values="MarketValueTHB", title="Allocation by Ticker"), use_container_width=True)

    st.subheader("Portfolio Level Risk")
    risk_period = st.selectbox("ช่วงเวลาคำนวณความเสี่ยง", ["6mo", "1y", "3y", "5y"], index=1, key="portfolio_risk_period")
    portfolio_level_metrics, portfolio_price_history = calculate_portfolio_level_risk(portfolio_calc, period=risk_period)

    if portfolio_level_metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", pct(portfolio_level_metrics["Portfolio Total Return"]))
        c2.metric("CAGR", pct(portfolio_level_metrics["Portfolio CAGR"]))
        c3.metric("Volatility", pct(portfolio_level_metrics["Portfolio Volatility"]))
        c4.metric("Sharpe", f"{portfolio_level_metrics['Portfolio Sharpe']:.2f}")
        c5.metric("Max Drawdown", pct(portfolio_level_metrics["Portfolio Max Drawdown"]))
    else:
        st.info("ยังคำนวณ Portfolio Level Risk ไม่ได้ เพราะมีสินทรัพย์ manual price หรือข้อมูล yfinance ไม่พอ")

    st.subheader("Risk Metrics by Holding")
    if portfolio_price_history.empty:
        st.info("ยังไม่มีข้อมูลราคาย้อนหลังสำหรับคำนวณความเสี่ยงรายตัว")
    else:
        risk_df = calculate_risk_metrics(portfolio_price_history)
        st.dataframe(risk_df.round(2), use_container_width=True, hide_index=True)

        if len(portfolio_price_history.columns) >= 2:
            st.subheader("Portfolio Holdings Correlation")
            corr = portfolio_price_history.pct_change().dropna().corr()
            heatmap = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r")
            heatmap.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(heatmap, use_container_width=True)

    st.caption("หมายเหตุ: Risk metrics คำนวณเฉพาะสินทรัพย์ที่ yfinance มีข้อมูล ส่วน BRKB80, กองทุนไทย, ทองไทย หรือ CASH ต้องใช้ manual price จึงยังไม่รวมใน risk calculation")


# =====================================================
# TAB 3: PORTFOLIO NEWS
# =====================================================

with tab_news:
    st.header("📰 Portfolio News")
    st.caption("ข่าวล่าสุดจาก Google News RSS ทั้งจากสินทรัพย์ในพอร์ตและ watchlist")

    portfolio_news_assets = portfolio_calc[["Ticker", "Broker"]].copy()
    portfolio_news_assets = portfolio_news_assets.rename(columns={"Ticker": "Symbol", "Broker": "Theme"})
    portfolio_news_assets["Name"] = ""
    portfolio_news_assets["Source"] = "Portfolio"

    watchlist_news_assets = watchlist[["Symbol", "Name", "Theme"]].copy()
    watchlist_news_assets["Source"] = "Watchlist"

    news_assets = pd.concat([portfolio_news_assets, watchlist_news_assets], ignore_index=True)
    news_assets["Symbol"] = news_assets["Symbol"].apply(clean_ticker)
    news_assets = news_assets[news_assets["Symbol"] != ""]
    news_assets = news_assets[~news_assets["Symbol"].isin(["CASH", "CASH THB", "THB CASH", "เงินสด"])]
    news_assets = news_assets.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    selected_news_symbols = c1.multiselect(
        "เลือกสินทรัพย์ที่ต้องการดูข่าว",
        options=news_assets["Symbol"].tolist(),
        default=news_assets["Symbol"].head(8).tolist(),
    )
    news_per_asset = c2.selectbox("จำนวนข่าวต่อสินทรัพย์", [1, 2, 3, 5], index=1)
    source_filter = c3.selectbox("แหล่งรายการ", ["ทั้งหมด", "Portfolio", "Watchlist"], index=0)

    if source_filter != "ทั้งหมด":
        display_assets = news_assets[(news_assets["Symbol"].isin(selected_news_symbols)) & (news_assets["Source"] == source_filter)]
    else:
        display_assets = news_assets[news_assets["Symbol"].isin(selected_news_symbols)]

    if display_assets.empty:
        st.info("ยังไม่มีสินทรัพย์ให้แสดงข่าว")
    else:
        for _, asset in display_assets.iterrows():
            symbol = asset.get("Symbol", "")
            name = asset.get("Name", "")
            theme = asset.get("Theme", "")
            source_type = asset.get("Source", "")
            query = build_news_query(symbol, name)
            news_items = fetch_google_news_rss(query, max_items=news_per_asset)

            st.markdown(f"### {symbol} {f'— {name}' if str(name).strip() else ''}")
            st.caption(f"{source_type} | {theme} | query: {query}")

            if not news_items:
                st.info("ยังไม่พบข่าวจาก RSS สำหรับรายการนี้")
            else:
                for item in news_items:
                    title = item.get("title", "")
                    source = item.get("source", "")
                    published = item.get("published", "")
                    link = item.get("link", "")

                    st.markdown(f"- **{title}**")
                    detail_line = ""
                    if source:
                        detail_line += f"แหล่งข่าว: {source}"
                    if published:
                        detail_line += f" | เวลา: {published}"
                    if detail_line:
                        st.caption(detail_line)
                    if link:
                        st.markdown(f"  [อ่านข่าวต้นฉบับ]({link})")
            st.divider()


# =====================================================
# TAB 4: MACRO DASHBOARD
# =====================================================

with tab_macro:
    st.header("🌍 Macro Dashboard")
    st.caption("เปรียบเทียบสินทรัพย์โลกแบบ log scale, momentum, risk metrics และ correlation")

    macro_options = {
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
        options=list(macro_options.keys()),
        default=["SPY - US Market", "QQQ - US Tech / AI", "BTC - Bitcoin", "GLD - Gold", "INDA - India", "MCHI - China"],
    )
    today = date.today()
    start_date_macro = c2.date_input("วันที่เริ่มต้น", value=today - timedelta(days=365 * 5), key="macro_start")
    end_date_macro = c3.date_input("วันที่สิ้นสุด", value=today, key="macro_end")
    momentum_choice = c4.selectbox("Momentum", ["1M", "3M", "6M", "1Y"], index=1)

    if start_date_macro >= end_date_macro or len(selected_assets) == 0:
        st.warning("กรุณาเลือกช่วงเวลาและสินทรัพย์ให้ถูกต้อง")
    else:
        selected_map = {name: macro_options[name] for name in selected_assets}
        symbol_map = {v: k for k, v in selected_map.items()}
        data = download_prices(list(selected_map.values()), start=start_date_macro, end=end_date_macro + timedelta(days=1))

        if data.empty:
            st.warning("ยังโหลดข้อมูลไม่ได้จาก yfinance")
        else:
            renamed_cols = {}
            for col in data.columns:
                renamed_cols[col] = symbol_map.get(col, col)
            data = data.rename(columns=renamed_cols)
            data = data.ffill().bfill().dropna(axis=1, how="all")
            returns = data.pct_change().dropna()
            normalized = data.div(data.iloc[0]).mul(100)

            latest_rows = []
            for name in data.columns:
                latest = data[name].dropna().iloc[-1]
                first = data[name].dropna().iloc[0]
                latest_rows.append({
                    "Asset": name,
                    "Latest": latest,
                    "Total Return %": (latest / first - 1) * 100,
                })

            st.subheader("Latest Macro Prices")
            st.dataframe(pd.DataFrame(latest_rows).round(2), use_container_width=True, hide_index=True)

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
            if len(data) > days:
                momentum = ((data.iloc[-1] / data.iloc[-days]) - 1) * 100
                momentum_df = pd.DataFrame({
                    "Asset": momentum.sort_values(ascending=False).index,
                    f"Momentum {momentum_choice} %": momentum.sort_values(ascending=False).values,
                })
                st.subheader(f"Momentum Ranking - {momentum_choice}")
                st.dataframe(momentum_df.round(2), use_container_width=True, hide_index=True)

            st.subheader("Risk Metrics")
            risk_df = calculate_risk_metrics(data)
            st.dataframe(risk_df.round(2), use_container_width=True, hide_index=True)

            if len(data.columns) >= 2 and not returns.empty:
                st.subheader("Correlation Heatmap")
                heatmap = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r")
                heatmap.update_layout(template="plotly_dark", height=650)
                st.plotly_chart(heatmap, use_container_width=True)

            st.caption("Risk metrics คำนวณจาก daily returns และ annualize ด้วย 252 trading days; Sharpe ใช้ risk-free rate = 0")



# =====================================================
# TAB 5: WATCHLIST + AUTO WATCHLIST
# =====================================================

with tab_watchlist:
    st.header("👀 Watchlist + Auto Watchlist")
    st.caption("รวมรายการเฝ้าดูจาก Google Sheet กับ Auto Watchlist ที่ระบบสแกนจาก Portfolio + Watchlist + Scan Universe")

    st.subheader("📌 Manual Watchlist from Google Sheet")
    if watchlist.empty:
        st.info("ยังไม่มี watchlist ให้สร้างแท็บ watchlist ใน Google Sheet โดยมีคอลัมน์ Symbol, Name, Theme, TargetPrice, Thesis")
    else:
        tickers = watchlist["Symbol"].tolist()
        current_prices = get_current_prices(tickers)

        watch = watchlist.copy()
        watch["YFinanceSymbol"] = watch["Symbol"].apply(normalize_symbol_for_yfinance)
        watch["CurrentPrice"] = watch["YFinanceSymbol"].map(current_prices)
        watch["CurrentPrice"] = to_number(watch["CurrentPrice"])
        watch["UpsideToTarget %"] = np.where(
            (watch["TargetPrice"] > 0) & (watch["CurrentPrice"] > 0),
            (watch["TargetPrice"] / watch["CurrentPrice"] - 1) * 100,
            0,
        )

        st.dataframe(watch.round(2), use_container_width=True, hide_index=True)

        priced_watch = watch[(watch["CurrentPrice"] > 0) & (watch["TargetPrice"] > 0)]
        if not priced_watch.empty:
            st.plotly_chart(
                px.bar(priced_watch, x="Symbol", y="UpsideToTarget %", color="Theme", title="Manual Watchlist: Upside to Target Price"),
                use_container_width=True,
            )

    st.divider()
    st.subheader("⭐ Auto Watchlist / Option Candidate Screener")
    st.caption("คัดหุ้นด้วย Multi-Timeframe Trend, MACD, RSI, Momentum และ Fair Value Lite จาก yfinance")

    c1, c2, c3 = st.columns([2, 1, 1])
    extra_input = c1.text_input(
        "เพิ่มหุ้นที่ต้องการสแกนเอง คั่นด้วย comma",
        value="",
        placeholder="เช่น HD,BKNG,META,MU"
    )
    max_symbols = c2.slider("จำนวนหุ้นสูงสุดที่สแกน", 10, 120, 60, step=10)
    min_score = c3.slider("คะแนนขั้นต่ำที่แสดง", 0.0, 10.0, 7.0, step=0.5)

    extra_symbols = [clean_ticker(x) for x in extra_input.split(",") if clean_ticker(x)]
    universe = get_candidate_universe(portfolio_calc, watchlist, DEFAULT_SCAN_UNIVERSE + extra_symbols)

    st.caption(f"Universe ทั้งหมด {len(universe)} ตัว | สแกนสูงสุด {max_symbols} ตัวแรก")

    with st.spinner("กำลังสแกน trend / fair value / momentum จาก yfinance..."):
        candidates = build_option_candidate_screener(universe, max_symbols=max_symbols)
        candidates = apply_manual_target_prices(candidates, watchlist)
        candidates = ensure_candidate_display_columns(candidates)

    if candidates.empty:
        st.warning("ยังไม่มีข้อมูลพอสำหรับสร้าง Auto Watchlist")
    else:
        candidates = candidates[candidates["TotalScore"] >= min_score].copy()

        if candidates.empty:
            st.info("ไม่มีหุ้นที่ผ่านคะแนนขั้นต่ำ ลองลด min score")
        else:
            high = candidates[candidates["TotalScore"] >= 9]
            good = candidates[(candidates["TotalScore"] >= 8) & (candidates["TotalScore"] < 9)]
            signal_today = candidates[candidates["SignalToday"].astype(str).str.len() > 0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Candidates", len(candidates))
            c2.metric("High Conviction", len(high))
            c3.metric("Good", len(good))
            c4.metric("Signal Today", len(signal_today))

            st.subheader("🔥 High Conviction List")
            high_cols = [
                "Symbol", "TotalScore", "Conviction", "SuggestedSetup",
                "Price", "FairValueDisplay", "MarginSafety%",
                "TrendScore", "ShortTrend", "MediumTrend", "LongTrend",
                "RSI14", "Momentum1M%", "Momentum3M%", "Momentum6M%", "Momentum12M%",
                "SignalToday"
            ]
            st.dataframe(
                candidates[existing_columns(candidates, high_cols)].round(2),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "TotalScore": st.column_config.ProgressColumn("Score", min_value=0, max_value=10),
                    "TrendScore": st.column_config.ProgressColumn("Trend", min_value=0, max_value=10),
                    "MarginSafety%": st.column_config.NumberColumn("MOS", format="%.2f%%"),
                    "Momentum1M%": st.column_config.NumberColumn("1M", format="%.2f%%"),
                    "Momentum3M%": st.column_config.NumberColumn("3M", format="%.2f%%"),
                    "Momentum6M%": st.column_config.NumberColumn("6M", format="%.2f%%"),
                    "Momentum12M%": st.column_config.NumberColumn("12M", format="%.2f%%"),
                },
            )

            st.subheader("📈 Multi-Timeframe Trend")
            trend_cols = ["Symbol", "ShortTrend", "MediumTrend", "LongTrend", "TrendScore", "MACD", "MACDSignal", "MACDHist", "RSI14", "PctFromHigh252%"]
            st.dataframe(
                candidates[existing_columns(candidates, trend_cols)].round(3),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ShortTrend": st.column_config.ProgressColumn("Short 0-3", min_value=0, max_value=3),
                    "MediumTrend": st.column_config.ProgressColumn("Medium 0-3", min_value=0, max_value=3),
                    "LongTrend": st.column_config.ProgressColumn("Long 0-3", min_value=0, max_value=3),
                    "TrendScore": st.column_config.ProgressColumn("Trend Score", min_value=0, max_value=10),
                    "PctFromHigh252%": st.column_config.NumberColumn("% from 52W High", format="%.2f%%"),
                },
            )

            st.subheader("💰 Fair Value Lite")
            fv_cols = ["Symbol", "Name", "Theme", "Price", "FairValueDisplay", "AnalystTargetDisplay", "MarginSafety%", "ForwardPEDisplay", "FairValueSource", "FairValueScore", "Thesis"]
            st.dataframe(
                candidates[existing_columns(candidates, fv_cols)].round(2),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MarginSafety%": st.column_config.NumberColumn("Margin of Safety", format="%.2f%%"),
                    "FairValueDisplay": st.column_config.TextColumn("Fair Value"),
                    "AnalystTargetDisplay": st.column_config.TextColumn("Analyst / Manual Target"),
                    "ForwardPEDisplay": st.column_config.TextColumn("Forward PE"),
                    "FairValueScore": st.column_config.ProgressColumn("FV Score", min_value=0, max_value=10),
                },
            )

            st.subheader("🚨 Signal Today")
            if signal_today.empty:
                st.info("วันนี้ยังไม่มี MACD Bullish Cross ในกลุ่มที่สแกน")
            else:
                st.dataframe(
                    signal_today[existing_columns(signal_today, ["Symbol", "SignalToday", "TotalScore", "SuggestedSetup", "Price", "MarginSafety%"])].round(2),
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("📊 Score Breakdown")
            st.plotly_chart(
                px.bar(
                    candidates.head(20),
                    x="Symbol",
                    y="TotalScore",
                    color="SuggestedSetup",
                    title="Top 20 Option Candidate Scores",
                ),
                use_container_width=True,
            )
            score_cols = ["Symbol", "TotalScore", "TrendScore", "FairValueScore", "MomentumScore", "RiskScore", "SuggestedSetup"]
            st.dataframe(candidates[existing_columns(candidates, score_cols)].round(2), use_container_width=True, hide_index=True)

            st.info("Auto Watchlist นี้ยังเป็น v1 จาก yfinance เท่านั้น ต่อไปค่อยเพิ่ม TradingView technical rating และ options chain API")




# =====================================================
# TAB 6: OPTION CALCULATOR
# =====================================================

with tab_option_calc:
    st.header("🧮 Option Calculator: Cash Secured Put + Long Call")
    st.caption("คำนวณจากกราฟ/เทรนด์ของหุ้นในพอร์ต + Auto Watchlist + Watchlist และดึง option chain จาก yfinance เบื้องต้น")

    st.info(
        "แนวคิดหลัก: ใช้ Watchlist/Portfolio เป็น universe → ตรวจกราฟด้วย Multi-timeframe trend, MACD, RSI → "
        "เลือกสัญญา PUT สำหรับ Cash Secured Put หรือ CALL สำหรับ Long Call. ข้อมูล option chain จาก yfinance อาจไม่ realtime เท่า broker/API เฉพาะทาง"
    )

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    calc_mode = c1.selectbox("โหมดคำนวณ", ["Cash Secured Put เพื่อรับหุ้น", "Long Call เพื่อ leverage ขาขึ้น", "ดูรายตัว"], index=0)
    min_dte_calc = c2.number_input("DTE ขั้นต่ำ", min_value=1, max_value=730, value=30 if calc_mode.startswith("Cash") else 90, step=1)
    max_dte_calc = c3.number_input("DTE สูงสุด", min_value=1, max_value=1000, value=75 if calc_mode.startswith("Cash") else 540, step=1)
    max_scan_calc = c4.slider("จำนวนหุ้นที่สแกน", 5, 60, 20, step=5)

    extra_calc = st.text_input("เพิ่ม ticker เอง คั่นด้วย comma", value="", placeholder="เช่น MU,NVDA,AVGO,RKLB")
    extra_calc_symbols = [clean_ticker(x) for x in extra_calc.split(",") if clean_ticker(x)]

    universe_calc = get_candidate_universe(portfolio_calc, watchlist, DEFAULT_SCAN_UNIVERSE + extra_calc_symbols)
    st.caption(f"Universe จาก Portfolio + Watchlist + Default Scan = {len(universe_calc)} ตัว")

    if calc_mode == "Cash Secured Put เพื่อรับหุ้น":
        st.subheader("🛡️ Cash Secured Put Scanner")
        st.caption("เหมาะกับหุ้นที่อยากรับเพิ่มอยู่แล้ว เลือก strike ที่รับหุ้นได้จริง และดู premium return ต่อเงินสดที่ต้องสำรอง")

        with st.spinner("กำลังดึง option chain และคำนวณ CSP..."):
            csp_df = build_csp_candidates_from_yfinance(universe_calc, min_dte=int(min_dte_calc), max_dte=int(max_dte_calc), max_symbols=max_scan_calc)

        if csp_df.empty:
            st.warning("ยังดึง option chain ไม่ได้ หรือไม่มี PUT ที่เข้าเกณฑ์")
        else:
            f1, f2, f3, f4 = st.columns(4)
            min_score_csp = f1.slider("CSP Score ขั้นต่ำ", 0.0, 10.0, 6.0, step=0.5)
            min_ann_ret = f2.slider("Annualized Return ขั้นต่ำ %", 0.0, 100.0, 8.0, step=1.0)
            min_buffer = f3.slider("Downside Buffer ขั้นต่ำ %", 0.0, 50.0, 5.0, step=1.0)
            max_spread = f4.slider("Spread สูงสุด %", 0.0, 100.0, 35.0, step=5.0)

            show = csp_df[
                (csp_df["CSPScore"] >= min_score_csp)
                & (csp_df["AnnualizedReturnPct"] >= min_ann_ret)
                & (csp_df["DownsideBufferPct"] >= min_buffer)
                & (csp_df["SpreadPct"] <= max_spread)
            ].copy()

            st.metric("จำนวนสัญญาที่ผ่านเกณฑ์", len(show))
            cols = [
                "Symbol", "Expiry", "DTE", "strike", "UnderlyingPrice", "Mid", "bid", "ask",
                "NetAssignPrice", "RequiredCash", "PremiumIncome", "PremiumReturnPct", "AnnualizedReturnPct",
                "DownsideBufferPct", "MoneynessPct", "openInterest", "volume", "SpreadPct",
                "TrendScore", "RSI14", "MarginSafetyPct", "CSPScore", "SignalToday"
            ]
            st.dataframe(
                show[cols].round(3).head(120),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "CSPScore": st.column_config.ProgressColumn("CSP Score", min_value=0, max_value=10),
                    "Mid": st.column_config.NumberColumn("Premium/Mid", format="$%.2f"),
                    "bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                    "ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                    "NetAssignPrice": st.column_config.NumberColumn("Net Assign", format="$%.2f"),
                    "RequiredCash": st.column_config.NumberColumn("Cash Required", format="$%.0f"),
                    "PremiumIncome": st.column_config.NumberColumn("Premium", format="$%.0f"),
                    "PremiumReturnPct": st.column_config.NumberColumn("Return", format="%.2f%%"),
                    "AnnualizedReturnPct": st.column_config.NumberColumn("Annualized", format="%.2f%%"),
                    "DownsideBufferPct": st.column_config.NumberColumn("Buffer", format="%.2f%%"),
                    "SpreadPct": st.column_config.NumberColumn("Spread", format="%.2f%%"),
                }
            )

            if not show.empty:
                top = show.head(20)
                st.plotly_chart(
                    px.bar(top, x="Symbol", y="CSPScore", color="AnnualizedReturnPct", hover_data=["Expiry", "strike", "Mid", "DownsideBufferPct"], title="Top CSP candidates"),
                    use_container_width=True,
                )

    elif calc_mode == "Long Call เพื่อ leverage ขาขึ้น":
        st.subheader("🚀 Long Call Scanner")
        st.caption("เหมาะกับหุ้นที่กราฟขาขึ้นชัด แต่ต้องระวัง premium, break-even และ time decay")

        with st.spinner("กำลังดึง option chain และคำนวณ Long Call... "):
            call_df = build_long_call_candidates_from_yfinance(universe_calc, min_dte=int(min_dte_calc), max_dte=int(max_dte_calc), max_symbols=max_scan_calc)

        if call_df.empty:
            st.warning("ยังดึง option chain ไม่ได้ หรือไม่มี CALL ที่เข้าเกณฑ์")
        else:
            f1, f2, f3 = st.columns(3)
            min_score_call = f1.slider("Call Score ขั้นต่ำ", 0.0, 10.0, 6.0, step=0.5)
            max_be = f2.slider("Break-even move สูงสุด %", 0.0, 80.0, 25.0, step=1.0)
            max_spread_call = f3.slider("Spread สูงสุด %", 0.0, 100.0, 35.0, step=5.0)

            show = call_df[
                (call_df["CallScore"] >= min_score_call)
                & (call_df["BreakevenMovePct"] <= max_be)
                & (call_df["SpreadPct"] <= max_spread_call)
            ].copy()

            st.metric("จำนวนสัญญาที่ผ่านเกณฑ์", len(show))
            cols = [
                "Symbol", "Expiry", "DTE", "strike", "UnderlyingPrice", "Mid", "bid", "ask",
                "Breakeven", "BreakevenMovePct", "ContractCost", "LeverageNotional",
                "openInterest", "volume", "SpreadPct", "TrendScore", "RSI14", "MarginSafetyPct", "CallScore", "SignalToday"
            ]
            st.dataframe(
                show[cols].round(3).head(120),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "CallScore": st.column_config.ProgressColumn("Call Score", min_value=0, max_value=10),
                    "Mid": st.column_config.NumberColumn("Premium/Mid", format="$%.2f"),
                    "Breakeven": st.column_config.NumberColumn("Break-even", format="$%.2f"),
                    "BreakevenMovePct": st.column_config.NumberColumn("BE Move", format="%.2f%%"),
                    "ContractCost": st.column_config.NumberColumn("Cost/contract", format="$%.0f"),
                    "LeverageNotional": st.column_config.NumberColumn("Notional leverage", format="%.1fx"),
                    "SpreadPct": st.column_config.NumberColumn("Spread", format="%.2f%%"),
                }
            )

    else:
        st.subheader("🔍 ดูหุ้นและ Option Chain รายตัว")
        symbol_single = st.selectbox("เลือกหุ้น", universe_calc, index=0 if universe_calc else None)
        if symbol_single:
            prices = get_current_prices([symbol_single])
            yf_symbol = clean_ticker(normalize_symbol_for_yfinance(symbol_single))
            spot = float(prices.get(yf_symbol, prices.get(symbol_single, 0)) or 0)
            st.metric("Underlying Price", f"${spot:,.2f}" if spot else "N/A")

            trend = analyze_trend_for_symbol(symbol_single) or {}
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Trend Score", f"{trend.get('TrendScore', 0):.2f}/10")
            t2.metric("RSI14", f"{trend.get('RSI14', 0):.2f}")
            t3.metric("MACD Hist", f"{trend.get('MACDHist', 0):.3f}")
            t4.metric("Signal", str(trend.get("SignalToday", "-")) or "-")

            hist = download_prices([symbol_single], period="1y")
            if not hist.empty:
                s = hist.iloc[:, 0].dropna()
                tech = add_technical_indicators(s)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tech.index, y=tech["Close"], name="Close", mode="lines"))
                fig.add_trace(go.Scatter(x=tech.index, y=tech["EMA20"], name="EMA20", mode="lines"))
                fig.add_trace(go.Scatter(x=tech.index, y=tech["EMA50"], name="EMA50", mode="lines"))
                fig.add_trace(go.Scatter(x=tech.index, y=tech["EMA200"], name="EMA200", mode="lines"))
                fig.update_layout(template="plotly_dark", height=520, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            expiries = get_option_expirations(symbol_single)
            if expiries:
                expiry_single = st.selectbox("เลือกวันหมดอายุ", expiries, index=min(1, len(expiries)-1))
                calls, puts = get_option_chain(symbol_single, expiry_single)
                calls = enrich_option_chain(calls, spot, "CALL", expiry_single)
                puts = enrich_option_chain(puts, spot, "PUT", expiry_single)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### CALL")
                    st.dataframe(calls[["strike", "bid", "ask", "Mid", "Breakeven", "BreakevenMovePct", "openInterest", "volume", "impliedVolatility", "SpreadPct"]].round(3), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("### PUT")
                    st.dataframe(puts[["strike", "bid", "ask", "Mid", "NetAssignPrice", "PremiumReturnPct", "AnnualizedReturnPct", "DownsideBufferPct", "openInterest", "volume", "impliedVolatility", "SpreadPct"]].round(3), use_container_width=True, hide_index=True)
            else:
                st.info("yfinance ไม่พบ option chain สำหรับหุ้นนี้")

    st.divider()
    st.markdown("""
    **เกณฑ์อ่านค่าเร็ว**  
    - CSP ที่น่าสนใจ: หุ้นอยากรับจริง, TrendScore ดี, DTE ประมาณ 30–75 วัน, premium annualized พอคุ้ม, spread ไม่กว้าง, downside buffer มีระยะพอ  
    - Long Call ที่น่าสนใจ: trend แข็ง, DTE ยาวพอให้ thesis ทำงาน, break-even move ไม่สูงเกินไป, spread ไม่กว้าง  
    - ตารางนี้เป็นเครื่องกรอง ไม่ใช่คำสั่งซื้อขาย ต้องเช็กใน broker อีกครั้งก่อนส่งคำสั่ง
    """)


# =====================================================
# TAB 7: OPTIONS WAR ROOM
# =====================================================

with tab_options:
    st.markdown("""
    <style>
    .option-warroom {
        background: radial-gradient(circle at top left, rgba(239,68,68,0.22), transparent 30%),
                    radial-gradient(circle at top right, rgba(59,130,246,0.18), transparent 30%),
                    linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        border: 1px solid rgba(148,163,184,0.28);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 20px 48px rgba(0,0,0,0.32);
    }
    .option-title {
        font-size: 38px;
        font-weight: 900;
        color: #f8fafc;
        margin-bottom: 4px;
        letter-spacing: -0.04em;
    }
    .option-subtitle {
        color: #94a3b8;
        font-size: 14px;
    }
    .metric-card {
        background: rgba(15,23,42,0.92);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 18px;
        padding: 18px;
        min-height: 108px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.10em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 850;
        margin-top: 8px;
    }
    .metric-good { color: #34d399; }
    .metric-bad { color: #fb7185; }
    .metric-warn { color: #fbbf24; }
    .small-note {
        color:#94a3b8;
        font-size:13px;
        margin-top:8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="option-warroom">
        <div class="option-title">🧨 OPTIONS WAR ROOM</div>
        <div class="option-subtitle">
            Cash Secured Put Scanner • Assignment Risk Dashboard • Contract Monitor
            <br>ข้อมูล options ยังเป็น manual จาก Google Sheet; ถ้าต้องการ realtime bid/ask/greeks ต้องต่อ Options API ภายหลัง
        </div>
    </div>
    """, unsafe_allow_html=True)

    if options_df.empty:
        st.info("ยังไม่มีข้อมูล options ให้สร้างแท็บ options ใน Google Sheet")
    else:
        opt = options_df.copy()
        opt["ExpiryDate"] = pd.to_datetime(opt["Expiry"], errors="coerce")
        today_ts = pd.Timestamp(date.today())
        opt["DTE"] = (opt["ExpiryDate"] - today_ts).dt.days
        opt["DTE"] = opt["DTE"].fillna(0).clip(lower=0)

        # ถ้าไม่ได้กรอก UnderlyingPrice ในชีต ให้พยายามดึงราคาหุ้นอ้างอิงจาก yfinance
        underlying_prices = get_current_prices(opt["Underlying"].dropna().unique().tolist())
        opt["YFinanceUnderlyingPrice"] = opt["Underlying"].map(underlying_prices)
        opt["YFinanceUnderlyingPrice"] = to_number(opt["YFinanceUnderlyingPrice"])
        opt["UnderlyingPrice"] = np.where(
            opt["UnderlyingPrice"] > 0,
            opt["UnderlyingPrice"],
            opt["YFinanceUnderlyingPrice"],
        )

        opt["PremiumUsed"] = np.where(opt["CurrentMark"] > 0, opt["CurrentMark"], opt["CurrentBid"])
        opt["ContractValue"] = opt["CurrentMark"] * opt["Contracts"] * 100
        opt["EntryValue"] = opt["EntryPrice"] * opt["Contracts"] * 100
        opt["PnL"] = np.where(
            opt["PositionSide"].eq("SELL"),
            opt["EntryValue"] - opt["ContractValue"],
            opt["ContractValue"] - opt["EntryValue"],
        )
        opt["PnL%"] = np.where(opt["EntryValue"] > 0, opt["PnL"] / opt["EntryValue"] * 100, 0)

        opt["Spread"] = opt["CurrentAsk"] - opt["CurrentBid"]
        opt["Spread%"] = np.where(opt["CurrentMark"] > 0, opt["Spread"] / opt["CurrentMark"] * 100, 0)
        opt["Moneyness%"] = np.where(
            opt["UnderlyingPrice"] > 0,
            (opt["UnderlyingPrice"] / opt["Strike"] - 1) * 100,
            0,
        )

        opt["CapitalRequired"] = opt["Strike"] * opt["Contracts"] * 100
        opt["PremiumIncome"] = opt["PremiumUsed"] * opt["Contracts"] * 100
        opt["AssignmentPrice"] = opt["Strike"] - opt["PremiumUsed"]
        opt["PremiumReturn%"] = np.where(opt["CapitalRequired"] > 0, opt["PremiumIncome"] / opt["CapitalRequired"] * 100, 0)
        opt["AnnualizedReturn%"] = np.where(
            opt["DTE"] > 0,
            opt["PremiumReturn%"] * 365 / opt["DTE"],
            0,
        )

        opt["LiquidityScore"] = (
            np.where(opt["OpenInterest"] >= 1000, 2, np.where(opt["OpenInterest"] >= 100, 1, 0))
            + np.where(opt["Volume"] >= 1_000_000, 2, np.where(opt["Volume"] >= 100_000, 1, 0))
            + np.where(opt["Spread%"] <= 10, 2, np.where(opt["Spread%"] <= 25, 1, 0))
        )
        opt["GreekHeat"] = (
            abs(opt["Delta"]) * 2
            + abs(opt["Gamma"]) * 10
            + abs(opt["Theta"]) * 5
            + abs(opt["Vega"]) * 2
            + opt["IV"]
        )
        opt["StarRating"] = np.clip((opt["LiquidityScore"] + opt["GreekHeat"]) / 2, 0, 5)

        open_opt = opt[opt["Status"].astype(str).str.upper() != "CLOSED"].copy()
        short_puts = open_opt[
            open_opt["OptionType"].str.contains("P", na=False)
            & open_opt["PositionSide"].eq("SELL")
        ].copy()

        total_value = open_opt["ContractValue"].sum()
        total_pnl = open_opt["PnL"].sum()
        avg_spread = open_opt["Spread%"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        total_assignment_capital = short_puts["CapitalRequired"].sum()
        usd_cash_available = total_cash / get_usdthb_rate() if get_usdthb_rate() else 0
        safety_margin = usd_cash_available - total_assignment_capital
        safety_margin_pct = (safety_margin / total_assignment_capital * 100) if total_assignment_capital > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card"><div class="metric-label">Open Contract Value</div><div class="metric-value">${total_value:,.0f}</div><div class="small-note">Mark value</div></div>""", unsafe_allow_html=True)
        pnl_class = "metric-good" if total_pnl >= 0 else "metric-bad"
        c2.markdown(f"""<div class="metric-card"><div class="metric-label">Open P/L</div><div class="metric-value {pnl_class}">${total_pnl:,.0f}</div><div class="small-note">Short premium adjusted</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><div class="metric-label">Assignment Capital</div><div class="metric-value">${total_assignment_capital:,.0f}</div><div class="small-note">Short puts only</div></div>""", unsafe_allow_html=True)
        margin_class = "metric-good" if safety_margin >= 0 else "metric-bad"
        c4.markdown(f"""<div class="metric-card"><div class="metric-label">Safety Margin</div><div class="metric-value {margin_class}">${safety_margin:,.0f}</div><div class="small-note">{safety_margin_pct:,.1f}% of required capital</div></div>""", unsafe_allow_html=True)

        st.subheader("🛡️ Assignment Risk Dashboard")
        if short_puts.empty:
            st.info("ยังไม่มี Short Put สำหรับคำนวณ assignment risk")
        else:
            assignment_summary = short_puts[[
                "Underlying", "Strike", "Expiry", "DTE", "Contracts", "UnderlyingPrice",
                "PremiumUsed", "AssignmentPrice", "CapitalRequired", "PremiumIncome",
                "PremiumReturn%", "AnnualizedReturn%", "Moneyness%", "Delta", "Status"
            ]].sort_values("CapitalRequired", ascending=False)

            st.dataframe(
                assignment_summary.round(2),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "PremiumUsed": st.column_config.NumberColumn("Premium", format="$%.2f"),
                    "AssignmentPrice": st.column_config.NumberColumn("Net Assign Price", format="$%.2f"),
                    "CapitalRequired": st.column_config.NumberColumn("Required Capital", format="$%.0f"),
                    "PremiumIncome": st.column_config.NumberColumn("Premium Income", format="$%.0f"),
                    "PremiumReturn%": st.column_config.NumberColumn("Return", format="%.2f%%"),
                    "AnnualizedReturn%": st.column_config.NumberColumn("Annualized", format="%.2f%%"),
                    "Moneyness%": st.column_config.NumberColumn("Distance to Strike", format="%.2f%%"),
                },
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.bar(short_puts, x="Underlying", y="CapitalRequired", color="Status", title="Assignment Capital by Underlying"),
                    use_container_width=True,
                )
            with c2:
                risk_meter = pd.DataFrame({
                    "Category": ["USD Cash Available", "Short Put Assignment Capital", "Safety Margin"],
                    "Value": [usd_cash_available, total_assignment_capital, safety_margin],
                })
                st.plotly_chart(
                    px.bar(risk_meter, x="Category", y="Value", title="Cash vs Assignment Requirement"),
                    use_container_width=True,
                )

            if safety_margin < 0:
                st.error("⚠️ Overallocated: ถ้าถูก assign ทุกสัญญาพร้อมกัน เงินสด USD ไม่พอรับหุ้นทั้งหมด")
            elif safety_margin_pct < 20:
                st.warning("🟡 Safety margin ต่ำกว่า 20% ควรระวังการเปิด short put เพิ่ม")
            else:
                st.success("🟢 Assignment risk ยังอยู่ในกรอบเงินสดที่มี")

        st.subheader("💰 Cash Secured Put Scanner")
        scanner = opt[
            opt["OptionType"].str.contains("P", na=False)
            & opt["PositionSide"].isin(["SELL", "WATCH"])
            & (opt["DTE"] > 0)
            & (opt["PremiumUsed"] > 0)
            & (opt["Strike"] > 0)
        ].copy()

        if scanner.empty:
            st.info("ยังไม่มีข้อมูล PUT สำหรับ scanner")
        else:
            c1, c2, c3 = st.columns(3)
            min_annual = c1.slider("ขั้นต่ำ Annualized Return (%)", 0, 100, 10)
            max_spread = c2.slider("Spread สูงสุด (%)", 0, 100, 30)
            min_oi = c3.number_input("Open Interest ขั้นต่ำ", min_value=0, value=0, step=10)

            scanner = scanner[
                (scanner["AnnualizedReturn%"] >= min_annual)
                & (scanner["Spread%"] <= max_spread)
                & (scanner["OpenInterest"] >= min_oi)
            ].sort_values(["AnnualizedReturn%", "StarRating"], ascending=False)

            show_cols = [
                "Underlying", "Strike", "Expiry", "DTE", "UnderlyingPrice",
                "PremiumUsed", "CurrentBid", "CurrentMark", "CurrentAsk",
                "AssignmentPrice", "CapitalRequired", "PremiumReturn%",
                "AnnualizedReturn%", "Moneyness%", "Delta", "IV", "OpenInterest",
                "Volume", "Spread%", "StarRating", "Status", "Note"
            ]

            st.dataframe(
                scanner[show_cols].round(3),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "PremiumUsed": st.column_config.NumberColumn("Premium", format="$%.2f"),
                    "AssignmentPrice": st.column_config.NumberColumn("Net Assign Price", format="$%.2f"),
                    "CapitalRequired": st.column_config.NumberColumn("Required Capital", format="$%.0f"),
                    "PremiumReturn%": st.column_config.NumberColumn("Return", format="%.2f%%"),
                    "AnnualizedReturn%": st.column_config.NumberColumn("Annualized", format="%.2f%%"),
                    "Moneyness%": st.column_config.NumberColumn("Distance to Strike", format="%.2f%%"),
                    "Spread%": st.column_config.NumberColumn("Spread", format="%.2f%%"),
                    "StarRating": st.column_config.ProgressColumn("Star Radar", min_value=0, max_value=5),
                },
            )

        st.subheader("⚔️ Contract Monitor")
        monitor_cols = [
            "Underlying", "OptionType", "PositionSide", "Strike", "Expiry", "DTE", "Contracts",
            "UnderlyingPrice", "CurrentBid", "CurrentMark", "CurrentAsk", "Spread%",
            "Delta", "Gamma", "Theta", "Vega", "IV",
            "OpenInterest", "Volume", "StarRating", "PnL", "PnL%",
            "Status", "Note"
        ]

        st.dataframe(
            opt[monitor_cols].round(3).sort_values("StarRating", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "CurrentBid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                "CurrentMark": st.column_config.NumberColumn("Mark", format="$%.2f"),
                "CurrentAsk": st.column_config.NumberColumn("Ask", format="$%.2f"),
                "Spread%": st.column_config.NumberColumn("Spread", format="%.2f%%"),
                "StarRating": st.column_config.ProgressColumn("Star Radar", min_value=0, max_value=5),
                "PnL": st.column_config.NumberColumn("P/L", format="$%.2f"),
                "PnL%": st.column_config.NumberColumn("P/L %", format="%.2f%%"),
            },
        )

        st.subheader("🔥 Option Heat Map")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(opt, x="Underlying", y="StarRating", color="OptionType", title="Star Rating by Contract"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.scatter(
                    opt,
                    x="Spread%",
                    y="OpenInterest",
                    size="Volume",
                    color="OptionType",
                    hover_name="Underlying",
                    title="Liquidity Map: Spread vs Open Interest",
                ),
                use_container_width=True,
            )

        st.info(
            "แท็บ options ใน Google Sheet ควรมีคอลัมน์: "
            "Underlying, OptionType, PositionSide, Strike, Expiry, Contracts, EntryPrice, "
            "CurrentBid, CurrentMark, CurrentAsk, Delta, Gamma, Theta, Vega, IV, Volume, "
            "OpenInterest, UnderlyingPrice, Status, Note"
        )



# =====================================================
# TAB 7: MARKET ANALYSIS
# =====================================================

with tab_market:
    st.header("🔎 Market Analysis")
    st.caption("วิเคราะห์ ticker รายตัว โดยเลือกช่วงวันที่ในหน้านี้เท่านั้น")

    c1, c2, c3 = st.columns([2, 1, 1])
    selected_symbol = c1.text_input("Enter Symbol", value="MMYT")
    start_date = c2.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = c3.date_input("End Date", value=date.today())

    if selected_symbol:
        yf_symbol = normalize_symbol_for_yfinance(selected_symbol)

        if start_date >= end_date:
            st.warning("กรุณาเลือกช่วงวันที่ให้ถูกต้อง")
        else:
            data = download_prices([yf_symbol], start=start_date, end=end_date + timedelta(days=1))
            if data.empty:
                st.warning("ไม่พบข้อมูลราคา")
            else:
                col = data.columns[0]
                series = data[col].dropna()

                st.subheader(f"Price Chart: {selected_symbol}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=selected_symbol))
                fig.update_layout(template="plotly_dark", height=520, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                latest = series.iloc[-1]
                first = series.iloc[0]
                period_return = (latest / first - 1) * 100
                risk_df = calculate_risk_metrics(data)

                c1, c2 = st.columns(2)
                c1.metric("Latest Price", f"{latest:,.2f}")
                c2.metric("Selected Period Return", pct(period_return))

                st.subheader("Risk Metrics")
                st.dataframe(risk_df.round(2), use_container_width=True, hide_index=True)