import json
from datetime import date, timedelta

import gspread
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from google.oauth2.service_account import Credentials


# =========================================================
# App Config
# =========================================================

st.set_page_config(page_title="Investment Dashboard", layout="wide")
st.title("Investment Intelligence Dashboard")

BASE_CURRENCY = "THB"

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


# =========================================================
# Google Sheets
# =========================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_google_credentials():
    if "gcp_service_account" not in st.secrets:
        st.error(
            "ยังไม่ได้ตั้งค่า Google Service Account ใน Streamlit secrets "
            "ให้เพิ่ม [gcp_service_account] ก่อนใช้งาน"
        )
        st.stop()

    service_account_info = dict(st.secrets["gcp_service_account"])
    return Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES,
    )


@st.cache_resource
def get_gsheet_client():
    credentials = get_google_credentials()
    return gspread.authorize(credentials)


@st.cache_resource
def get_spreadsheet():
    if "GOOGLE_SHEET_ID" not in st.secrets:
        st.error("ยังไม่ได้ตั้งค่า GOOGLE_SHEET_ID ใน Streamlit secrets")
        st.stop()

    client = get_gsheet_client()
    return client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])


def get_or_create_worksheet(sheet_name, headers):
    spreadsheet = get_spreadsheet()

    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=sheet_name,
            rows=200,
            cols=max(len(headers), 8),
        )
        worksheet.update([headers])

    values = worksheet.get_all_values()

    if not values:
        worksheet.update([headers])
    elif values[0] != headers:
        # Keep existing matching columns, add missing columns, preserve old data where possible
        old_headers = values[0]
        rows = values[1:]

        normalized_rows = []
        for row in rows:
            old_row = dict(zip(old_headers, row))
            normalized_rows.append([old_row.get(h, "") for h in headers])

        worksheet.clear()
        worksheet.update([headers] + normalized_rows)

    return worksheet


def read_sheet(sheet_name, columns):
    headers = list(columns.keys())
    worksheet = get_or_create_worksheet(sheet_name, headers)
    records = worksheet.get_all_records()

    if not records:
        return pd.DataFrame(columns)

    df = pd.DataFrame(records)

    for col, default_value in columns.items():
        if col not in df.columns:
            df[col] = default_value

    return df[headers]


def write_sheet(sheet_name, df, columns):
    headers = list(columns.keys())
    worksheet = get_or_create_worksheet(sheet_name, headers)

    save_df = df.copy()

    for col, default_value in columns.items():
        if col not in save_df.columns:
            save_df[col] = default_value

    save_df = save_df[headers]
    save_df = save_df.fillna("")

    worksheet.clear()
    worksheet.update([headers] + save_df.astype(str).values.tolist())

    st.cache_data.clear()
    st.success(f"Saved to Google Sheet: {sheet_name}")
    st.rerun()


# =========================================================
# Helpers
# =========================================================

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


def get_asset_name(ticker):
    ticker = clean_ticker(ticker)
    return DEFAULT_MARKET_ASSETS.get(ticker, ticker)


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
        close = (
            data["Close"]
            if "Close" in data.columns.get_level_values(0)
            else data.xs("Close", level=1, axis=1)
        )
    else:
        close = data["Close"] if "Close" in data.columns else data

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close.columns = [clean_ticker(c) for c in close.columns]
    return close


def get_current_prices(tickers):
    tickers = [
        clean_ticker(t)
        for t in tickers
        if clean_ticker(t) and clean_ticker(t) != "CASH"
    ]

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


# =========================================================
# Data Loaders
# =========================================================

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


def load_portfolio():
    df = read_sheet(SHEET_TABS["portfolio"], PORTFOLIO_COLUMNS)
    df["Broker"] = df["Broker"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].apply(clean_ticker)
    df["Currency"] = df["Currency"].apply(clean_currency)
    df["Quantity"] = to_number(df["Quantity"])
    df["AvgCost"] = to_number(df["AvgCost"])
    df["ManualPrice"] = to_number(df["ManualPrice"])
    return df


def load_banks():
    df = read_sheet(SHEET_TABS["bank_accounts"], BANK_COLUMNS)
    df["Currency"] = df["Currency"].apply(clean_currency)
    df["Balance"] = to_number(df["Balance"])
    df["FxRateToTHB"] = df["Currency"].apply(fx_to_thb)
    df["BalanceTHB"] = df["Balance"] * df["FxRateToTHB"]
    return df


def load_properties():
    df = read_sheet(SHEET_TABS["properties"], PROPERTY_COLUMNS)
    df["EstimatedValue"] = to_number(df["EstimatedValue"])
    return df


def load_mortgage():
    df = read_sheet(SHEET_TABS["mortgage"], MORTGAGE_COLUMNS)
    df["OutstandingDebt"] = to_number(df["OutstandingDebt"])
    df["MonthlyPayment"] = to_number(df["MonthlyPayment"])
    df["InterestRate"] = to_number(df["InterestRate"])
    return df


def load_property_cashflow():
    df = read_sheet(SHEET_TABS["property_cashflow"], CASHFLOW_COLUMNS)
    df["Rent"] = to_number(df["Rent"])
    df["Expense"] = to_number(df["Expense"])
    df["ExtraPayment"] = to_number(df["ExtraPayment"])
    return df


# =========================================================
# Portfolio Logic
# =========================================================

def calculate_portfolio(portfolio):
    portfolio = portfolio.copy()
    portfolio["Ticker"] = portfolio["Ticker"].apply(clean_ticker)
    portfolio = portfolio[portfolio["Ticker"] != ""]

    portfolio["Currency"] = portfolio["Currency"].apply(clean_currency)
    portfolio["Quantity"] = to_number(portfolio["Quantity"])
    portfolio["AvgCost"] = to_number(portfolio["AvgCost"])
    portfolio["ManualPrice"] = to_number(portfolio["ManualPrice"])

    tickers = (
        portfolio.loc[portfolio["Ticker"] != "CASH", "Ticker"]
        .dropna()
        .unique()
        .tolist()
    )

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
    portfolio.loc[mask, "ReturnPct"] = (
        portfolio.loc[mask, "PnLNative"]
        / portfolio.loc[mask, "CostBasisNative"]
        * 100
    )

    return portfolio


def render_portfolio_section():
    st.subheader("Investment Portfolio")
    st.caption(f"Base Currency = THB | USD/THB = {get_usdthb_rate():,.4f}")

    portfolio = load_portfolio()
    portfolio_calc = calculate_portfolio(portfolio)

    display_columns = [
        "Broker",
        "Ticker",
        "Currency",
        "Quantity",
        "AvgCost",
        "ManualPrice",
        "CurrentPrice",
        "PriceSource",
        "FxRateToTHB",
        "MarketValueNative",
        "MarketValueTHB",
        "PnLTHB",
        "ReturnPct",
    ]

    for col in display_columns:
        if col not in portfolio_calc.columns:
            portfolio_calc[col] = ""

    st.markdown("### Edit Portfolio")
    st.caption(
        "CASH จะรวมใน Cash และ Net Worth แต่จะไม่ถูกนำไปคำนวณ Investment Return"
    )

    edited = st.data_editor(
        portfolio_calc[display_columns],
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor_main",
        disabled=[
            "CurrentPrice",
            "PriceSource",
            "FxRateToTHB",
            "MarketValueNative",
            "MarketValueTHB",
            "PnLTHB",
            "ReturnPct",
        ],
        column_config={
            "Broker": st.column_config.TextColumn("Broker"),
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Currency": st.column_config.SelectboxColumn(
                "Currency",
                options=["THB", "USD"],
                required=True,
            ),
            "Quantity": st.column_config.NumberColumn(
                "Quantity",
                step=0.000001,
                format="%.8f",
            ),
            "AvgCost": st.column_config.NumberColumn(
                "AvgCost",
                step=0.000001,
                format="%.6f",
            ),
            "ManualPrice": st.column_config.NumberColumn(
                "ManualPrice",
                help="ใส่ราคาปัจจุบันเองเมื่อ yfinance ดึงไม่ได้ เช่น BRKB80, K-USXNDQ-A(A), MTS-GOLD",
                step=0.000001,
                format="%.6f",
            ),
        },
    )

    save_df = edited[
        ["Broker", "Ticker", "Currency", "Quantity", "AvgCost", "ManualPrice"]
    ].copy()

    save_df["Broker"] = save_df["Broker"].astype(str).str.strip()
    save_df["Ticker"] = save_df["Ticker"].apply(clean_ticker)
    save_df["Currency"] = save_df["Currency"].apply(clean_currency)
    save_df["Quantity"] = to_number(save_df["Quantity"])
    save_df["AvgCost"] = to_number(save_df["AvgCost"])
    save_df["ManualPrice"] = to_number(save_df["ManualPrice"])
    save_df = save_df[save_df["Ticker"] != ""]

    if st.button("Save Portfolio", key="save_portfolio_main"):
        write_sheet(SHEET_TABS["portfolio"], save_df, PORTFOLIO_COLUMNS)

    portfolio_calc = calculate_portfolio(save_df)

    missing = portfolio_calc[
        (portfolio_calc["CurrentPrice"] == 0)
        & (portfolio_calc["Ticker"] != "CASH")
    ]

    if not missing.empty:
        st.warning(
            "ยังไม่มีราคาสำหรับ: "
            + ", ".join(missing["Ticker"].unique())
            + " — ให้ใส่ ManualPrice"
        )

    cash_from_portfolio = portfolio_calc[portfolio_calc["IsCash"]].copy()
    investments = portfolio_calc[~portfolio_calc["IsCash"]].copy()

    portfolio_cash_value = cash_from_portfolio["MarketValueTHB"].sum()
    investment_value = investments["MarketValueTHB"].sum()
    investment_cost = investments["CostBasisTHB"].sum()
    investment_pnl = investment_value - investment_cost
    investment_return_pct = (
        investment_pnl / investment_cost * 100
        if investment_cost
        else 0
    )

    total_portfolio_value = portfolio_cash_value + investment_value

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Value (THB)", f"{total_portfolio_value:,.2f}")
    col2.metric("Portfolio Cash (THB)", f"{portfolio_cash_value:,.2f}")
    col3.metric("Investment Value (THB)", f"{investment_value:,.2f}")
    col4.metric("Investment Return %", f"{investment_return_pct:.2f}%")

    col5, col6 = st.columns(2)
    col5.metric("Investment Gain/Loss (THB)", f"{investment_pnl:,.2f}")
    cash_pct = (
        portfolio_cash_value / total_portfolio_value * 100
        if total_portfolio_value
        else 0
    )
    col6.metric("Portfolio Cash %", f"{cash_pct:.2f}%")

    st.markdown("### Portfolio Summary")
    summary_cols = [
        "Broker",
        "Ticker",
        "Currency",
        "Quantity",
        "AvgCost",
        "CurrentPrice",
        "PriceSource",
        "MarketValueNative",
        "MarketValueTHB",
        "PnLTHB",
        "ReturnPct",
    ]
    st.dataframe(portfolio_calc[summary_cols].round(2), use_container_width=True)

    c1, c2 = st.columns(2)

    broker_summary = (
        portfolio_calc.groupby("Broker", dropna=False)["MarketValueTHB"]
        .sum()
        .reset_index()
    )

    ticker_summary = (
        portfolio_calc.groupby("Ticker", dropna=False)["MarketValueTHB"]
        .sum()
        .reset_index()
    )

    with c1:
        if broker_summary["MarketValueTHB"].sum() != 0:
            st.plotly_chart(
                px.pie(
                    broker_summary,
                    names="Broker",
                    values="MarketValueTHB",
                    title="Allocation by Broker (THB)",
                ),
                use_container_width=True,
            )

    with c2:
        if ticker_summary["MarketValueTHB"].sum() != 0:
            st.plotly_chart(
                px.pie(
                    ticker_summary,
                    names="Ticker",
                    values="MarketValueTHB",
                    title="Allocation by Ticker (THB)",
                ),
                use_container_width=True,
            )

    portfolio_stats = {
        "portfolio_value": total_portfolio_value,
        "portfolio_cash": portfolio_cash_value,
        "investment_value": investment_value,
        "investment_cost": investment_cost,
        "investment_pnl": investment_pnl,
        "investment_return_pct": investment_return_pct,
    }

    return portfolio_stats, portfolio_calc


# =========================================================
# Bank Logic
# =========================================================

def render_bank_section():
    st.subheader("Bank Accounts")

    banks = load_banks()

    edited = st.data_editor(
        banks[["Bank", "Account", "Balance", "Currency"]],
        num_rows="dynamic",
        use_container_width=True,
        key="bank_editor",
        column_config={
            "Balance": st.column_config.NumberColumn("Balance", step=100.0),
            "Currency": st.column_config.SelectboxColumn(
                "Currency",
                options=["THB", "USD"],
                required=True,
            ),
        },
    )

    edited["Currency"] = edited["Currency"].apply(clean_currency)
    edited["Balance"] = to_number(edited["Balance"])
    edited["FxRateToTHB"] = edited["Currency"].apply(fx_to_thb)
    edited["BalanceTHB"] = edited["Balance"] * edited["FxRateToTHB"]

    if st.button("Save Bank Accounts", key="save_bank_accounts"):
        write_sheet(SHEET_TABS["bank_accounts"], edited[["Bank", "Account", "Balance", "Currency"]], BANK_COLUMNS)

    bank_cash_value = edited["BalanceTHB"].sum()

    st.metric("Bank Cash (THB)", f"{bank_cash_value:,.2f}")
    st.dataframe(edited.round(2), use_container_width=True)

    if edited["BalanceTHB"].sum() != 0:
        st.plotly_chart(
            px.pie(
                edited,
                names="Bank",
                values="BalanceTHB",
                title="Bank Cash Allocation (THB)",
            ),
            use_container_width=True,
        )

    return bank_cash_value, edited


# =========================================================
# Real Estate Logic
# =========================================================

def load_and_render_real_estate():
    st.subheader("Real Estate")

    properties = load_properties()
    mortgage = load_mortgage()
    cashflow = load_property_cashflow()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Properties")
        edited_properties = st.data_editor(
            properties,
            num_rows="dynamic",
            use_container_width=True,
            key="properties_editor",
            column_config={
                "EstimatedValue": st.column_config.NumberColumn(
                    "EstimatedValue",
                    step=10000.0,
                )
            },
        )

        if st.button("Save Properties", key="save_properties"):
            write_sheet(SHEET_TABS["properties"], edited_properties, PROPERTY_COLUMNS)

    with c2:
        st.markdown("### Mortgage")
        edited_mortgage = st.data_editor(
            mortgage,
            num_rows="dynamic",
            use_container_width=True,
            key="mortgage_editor",
            column_config={
                "OutstandingDebt": st.column_config.NumberColumn(
                    "OutstandingDebt",
                    step=10000.0,
                ),
                "MonthlyPayment": st.column_config.NumberColumn(
                    "MonthlyPayment",
                    step=1000.0,
                ),
                "InterestRate": st.column_config.NumberColumn(
                    "InterestRate",
                    step=0.01,
                ),
            },
        )

        if st.button("Save Mortgage", key="save_mortgage"):
            write_sheet(SHEET_TABS["mortgage"], edited_mortgage, MORTGAGE_COLUMNS)

    st.markdown("### Property Cash Flow")
    edited_cashflow = st.data_editor(
        cashflow,
        num_rows="dynamic",
        use_container_width=True,
        key="cashflow_editor",
        column_config={
            "Rent": st.column_config.NumberColumn("Rent", step=1000.0),
            "Expense": st.column_config.NumberColumn("Expense", step=1000.0),
            "ExtraPayment": st.column_config.NumberColumn(
                "ExtraPayment",
                step=1000.0,
            ),
        },
    )

    if st.button("Save Property Cash Flow", key="save_cashflow"):
        write_sheet(SHEET_TABS["property_cashflow"], edited_cashflow, CASHFLOW_COLUMNS)

    edited_properties["EstimatedValue"] = to_number(
        edited_properties["EstimatedValue"]
    )
    edited_mortgage["OutstandingDebt"] = to_number(
        edited_mortgage["OutstandingDebt"]
    )

    real_estate = edited_properties.merge(
        edited_mortgage[
            ["Property", "OutstandingDebt", "MonthlyPayment", "InterestRate"]
        ],
        on="Property",
        how="left",
    )

    real_estate["OutstandingDebt"] = to_number(real_estate["OutstandingDebt"])
    real_estate["Equity"] = (
        real_estate["EstimatedValue"] - real_estate["OutstandingDebt"]
    )

    property_value = real_estate["EstimatedValue"].sum()
    property_debt = real_estate["OutstandingDebt"].sum()
    property_equity = real_estate["Equity"].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Property Value (THB)", f"{property_value:,.2f}")
    m2.metric("Property Debt (THB)", f"{property_debt:,.2f}")
    m3.metric("Property Equity (THB)", f"{property_equity:,.2f}")

    st.dataframe(real_estate.round(2), use_container_width=True)

    if property_equity != 0:
        st.plotly_chart(
            px.pie(
                real_estate,
                names="Property",
                values="Equity",
                title="Real Estate Equity Allocation (THB)",
            ),
            use_container_width=True,
        )

    edited_cashflow["Rent"] = to_number(edited_cashflow["Rent"])
    edited_cashflow["Expense"] = to_number(edited_cashflow["Expense"])
    edited_cashflow["ExtraPayment"] = to_number(edited_cashflow["ExtraPayment"])
    edited_cashflow["NetCashFlow"] = (
        edited_cashflow["Rent"]
        - edited_cashflow["Expense"]
        - edited_cashflow["ExtraPayment"]
    )

    st.markdown("### Property Monthly Cash Flow")
    st.dataframe(edited_cashflow.round(2), use_container_width=True)

    if not edited_cashflow.empty:
        cashflow_summary = (
            edited_cashflow.groupby("Property", dropna=False)["NetCashFlow"]
            .sum()
            .reset_index()
        )
        st.plotly_chart(
            px.bar(
                cashflow_summary,
                x="Property",
                y="NetCashFlow",
                title="Net Cash Flow by Property",
            ),
            use_container_width=True,
        )

    return property_value, property_debt, property_equity, real_estate, edited_cashflow


# =========================================================
# Net Worth
# =========================================================

def render_net_worth_summary(
    portfolio_stats,
    bank_cash_value,
    property_value,
    property_debt,
    property_equity,
):
    st.header("Net Worth Dashboard")

    total_cash = portfolio_stats["portfolio_cash"] + bank_cash_value
    investment_value = portfolio_stats["investment_value"]
    investment_pnl = portfolio_stats["investment_pnl"]
    investment_return_pct = portfolio_stats["investment_return_pct"]

    net_worth = total_cash + investment_value + property_equity

    # Monthly cashflow summary from Google Sheets
    mortgage_df = load_mortgage()
    cashflow_df = load_property_cashflow()

    monthly_rent = cashflow_df["Rent"].sum() if not cashflow_df.empty else 0
    monthly_expense = cashflow_df["Expense"].sum() if not cashflow_df.empty else 0
    monthly_extra_payment = cashflow_df["ExtraPayment"].sum() if not cashflow_df.empty else 0
    monthly_mortgage = mortgage_df["MonthlyPayment"].sum() if not mortgage_df.empty else 0
    monthly_net_cashflow = monthly_rent - monthly_expense - monthly_mortgage - monthly_extra_payment

    st.subheader("Overall Wealth")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Worth (THB)", f"{net_worth:,.2f}")
    c2.metric("Total Cash (THB)", f"{total_cash:,.2f}")
    c3.metric("Investment Value (THB)", f"{investment_value:,.2f}")
    c4.metric("Real Estate Equity (THB)", f"{property_equity:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Property Gross Value (THB)", f"{property_value:,.2f}")
    c6.metric("Property Debt (THB)", f"{property_debt:,.2f}")
    c7.metric("Investment Gain/Loss (THB)", f"{investment_pnl:,.2f}")
    c8.metric("Investment Return %", f"{investment_return_pct:.2f}%")

    st.subheader("Monthly Cash Flow")
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Rent (THB/month)", f"{monthly_rent:,.2f}")
    c10.metric("Expense (THB/month)", f"{monthly_expense:,.2f}")
    c11.metric("Mortgage (THB/month)", f"{monthly_mortgage:,.2f}")
    c12.metric("Net Cash Flow (THB/month)", f"{monthly_net_cashflow:,.2f}")

    networth_df = pd.DataFrame(
        {
            "Category": [
                "Cash",
                "Investment Portfolio",
                "Real Estate Equity",
            ],
            "ValueTHB": [
                total_cash,
                investment_value,
                property_equity,
            ],
        }
    )

    debt_df = pd.DataFrame(
        {
            "Category": [
                "Cash",
                "Investment Portfolio",
                "Real Estate Gross Value",
                "Real Estate Debt",
                "Net Worth",
            ],
            "ValueTHB": [
                total_cash,
                investment_value,
                property_value,
                -property_debt,
                net_worth,
            ],
        }
    )

    cashflow_df_summary = pd.DataFrame(
        {
            "Category": ["Rent", "Expense", "Mortgage", "Extra Payment", "Net Cash Flow"],
            "ValueTHB": [monthly_rent, -monthly_expense, -monthly_mortgage, -monthly_extra_payment, monthly_net_cashflow],
        }
    )

    c1, c2 = st.columns(2)

    with c1:
        if networth_df["ValueTHB"].sum() != 0:
            st.plotly_chart(
                px.pie(
                    networth_df,
                    names="Category",
                    values="ValueTHB",
                    title="Net Worth Allocation (THB)",
                ),
                use_container_width=True,
            )

    with c2:
        st.plotly_chart(
            px.bar(
                debt_df,
                x="Category",
                y="ValueTHB",
                title="Assets, Debt, and Net Worth (THB)",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        px.bar(
            cashflow_df_summary,
            x="Category",
            y="ValueTHB",
            title="Monthly Cash Flow Breakdown (THB)",
        ),
        use_container_width=True,
    )

    st.markdown("### Net Worth Breakdown")
    st.dataframe(networth_df.round(2), use_container_width=True)


# =========================================================
# Market Analysis
# =========================================================

def get_portfolio_tickers_for_market():
    portfolio = load_portfolio()
    tickers = (
        portfolio["Ticker"]
        .apply(clean_ticker)
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    manual_only = {"CASH", "BRKB80", "K-USXNDQ-A(A)", "MTS-GOLD"}
    return [t for t in tickers if t not in manual_only]


def render_market_analysis():
    st.header("Market Analysis")

    portfolio_tickers = get_portfolio_tickers_for_market()
    default_tickers = list(
        dict.fromkeys(list(DEFAULT_MARKET_ASSETS.keys()) + portfolio_tickers)
    )

    with st.sidebar:
        st.subheader("Market Settings")

        ticker_input = st.text_area(
            "Tickers ที่ต้องการติดตาม",
            value=",".join(default_tickers),
            help="คั่นด้วย comma เช่น AAPL,MSFT,NVDA,MMYT,BTC-USD",
        )

        today = date.today()
        default_start = today - timedelta(days=365 * 5)

        start_date = st.date_input("วันที่เริ่มต้น", value=default_start)
        end_date = st.date_input("วันที่สิ้นสุด", value=today)

        momentum_choice = st.selectbox(
            "เลือกช่วง Momentum",
            ["1M", "3M", "6M", "1Y"],
            index=1,
        )

    selected_assets = [
        clean_ticker(t) for t in ticker_input.split(",") if clean_ticker(t)
    ]
    selected_assets = list(dict.fromkeys(selected_assets))

    if start_date >= end_date:
        st.warning("วันที่เริ่มต้นต้องมาก่อนวันที่สิ้นสุด")
        return

    if not selected_assets:
        st.warning("กรุณาเลือกสินทรัพย์อย่างน้อย 1 ตัว")
        return

    data = download_prices(selected_assets, start=start_date, end=end_date)

    if data.empty:
        st.warning("ไม่พบข้อมูลราคาจาก yfinance สำหรับ ticker ที่เลือก")
        return

    data = data.ffill().bfill().dropna(axis=1, how="all")

    if data.empty:
        st.warning("ข้อมูลราคาว่างหลังจากล้างข้อมูล")
        return

    missing = sorted(set(selected_assets) - set(data.columns))
    if missing:
        st.info("Ticker ที่ดึงราคาไม่ได้: " + ", ".join(missing))

    returns = data.pct_change().dropna()
    normalized = data.div(data.iloc[0]).mul(100)

    st.subheader(f"Performance Comparison: {start_date} to {end_date}")

    fig = go.Figure()
    for ticker in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[ticker],
                mode="lines",
                name=f"{ticker} - {get_asset_name(ticker)}",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        yaxis=dict(
            type="log",
            title="Normalized Performance (Log Scale)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    momentum_days = {
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }[momentum_choice]

    st.subheader(f"Momentum Ranking - {momentum_choice}")

    if len(data) > momentum_days:
        momentum = (data.iloc[-1] / data.iloc[-momentum_days] - 1) * 100
        momentum = momentum.sort_values(ascending=False)
        momentum_df = pd.DataFrame(
            {
                "Ticker": momentum.index,
                "Asset": [get_asset_name(t) for t in momentum.index],
                f"Momentum {momentum_choice} %": momentum.values,
            }
        )
        st.dataframe(momentum_df.round(2), use_container_width=True)
    else:
        st.info("ข้อมูลยังไม่พอสำหรับคำนวณ Momentum ช่วงนี้")

    st.subheader("Correlation Heatmap")

    if len(data.columns) >= 2 and not returns.empty:
        heatmap = px.imshow(
            returns.corr(),
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
        )
        heatmap.update_layout(template="plotly_dark", height=650)
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("ต้องเลือกอย่างน้อย 2 สินทรัพย์เพื่อดู Correlation")

    st.subheader("Risk Analysis")

    if len(data) < 2 or returns.empty:
        st.info("ข้อมูลยังไม่พอสำหรับคำนวณ Risk Analysis")
        return

    total_return = (data.iloc[-1] / data.iloc[0] - 1) * 100
    years_count = max(len(data) / 252, 1 / 252)
    annual_return = ((data.iloc[-1] / data.iloc[0]) ** (1 / years_count) - 1) * 100
    annual_volatility = returns.std() * (252 ** 0.5) * 100

    drawdown = (data / data.cummax() - 1) * 100
    max_drawdown = drawdown.min()

    risk_df = pd.DataFrame(
        {
            "Ticker": data.columns,
            "Asset": [get_asset_name(t) for t in data.columns],
            "Total Return %": total_return.values,
            "Annual Return %": annual_return.values,
            "Annual Volatility %": annual_volatility.values,
            "Max Drawdown %": max_drawdown.values,
        }
    )

    risk_df["Sharpe Ratio"] = risk_df.apply(
        lambda row: row["Annual Return %"] / row["Annual Volatility %"]
        if row["Annual Volatility %"] != 0
        else 0,
        axis=1,
    )

    risk_df = risk_df.sort_values("Annual Return %", ascending=False)
    st.dataframe(risk_df.round(2), use_container_width=True)

    st.subheader("Risk vs Return")

    risk_return_fig = px.scatter(
        risk_df,
        x="Annual Volatility %",
        y="Annual Return %",
        text="Ticker",
        hover_name="Asset",
        size=risk_df["Total Return %"].abs() + 10,
        title="Risk vs Return Matrix",
    )

    risk_return_fig.update_traces(textposition="top center")
    risk_return_fig.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(risk_return_fig, use_container_width=True)

    st.subheader("Drawdown Chart")

    drawdown_fig = go.Figure()
    for ticker in drawdown.columns:
        drawdown_fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown[ticker],
                mode="lines",
                name=f"{ticker} - {get_asset_name(ticker)}",
            )
        )

    drawdown_fig.update_layout(
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        yaxis_title="Drawdown %",
    )
    st.plotly_chart(drawdown_fig, use_container_width=True)


# =========================================================
# Main Layout
# =========================================================

tab_wealth, tab_market = st.tabs(["My Wealth", "Market Analysis"])

with tab_wealth:
    st.header("My Wealth")

    portfolio_stats, portfolio_calc = render_portfolio_section()
    st.divider()

    bank_cash_value, banks = render_bank_section()
    st.divider()

    (
        property_value,
        property_debt,
        property_equity,
        real_estate,
        property_cashflow,
    ) = load_and_render_real_estate()

    st.divider()

    render_net_worth_summary(
        portfolio_stats=portfolio_stats,
        bank_cash_value=bank_cash_value,
        property_value=property_value,
        property_debt=property_debt,
        property_equity=property_equity,
    )

with tab_market:
    render_market_analysis()
