import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# App Config
# =========================================================

st.set_page_config(
    page_title="Investment Intelligence Dashboard",
    layout="wide"
)

st.title("Investment Intelligence Dashboard")


# =========================================================
# File Paths
# =========================================================

DATA_FILES = {
    "portfolio": "portfolio.csv",
    "bank_accounts": "bank_accounts.csv",
    "properties": "properties.csv",
    "mortgage": "mortgage.csv",
    "property_cashflow": "property_cashflow.csv",
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
# Helpers
# =========================================================

def clean_ticker(ticker):
    if pd.isna(ticker):
        return ""
    return str(ticker).strip().upper()


def to_number(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def ensure_columns(df, required_columns):
    for col, default_value in required_columns.items():
        if col not in df.columns:
            df[col] = default_value
    return df[list(required_columns.keys())]


def read_csv_or_template(path, required_columns):
    if not os.path.exists(path):
        return pd.DataFrame(required_columns)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(required_columns)

    return ensure_columns(df, required_columns)


def save_csv(df, path):
    df.to_csv(path, index=False)
    st.success(f"Saved {path}")
    st.cache_data.clear()
    st.rerun()


def get_asset_name(ticker, custom_names=None):
    ticker = clean_ticker(ticker)

    if custom_names and ticker in custom_names:
        return custom_names[ticker]

    return DEFAULT_MARKET_ASSETS.get(ticker, ticker)


@st.cache_data(ttl=300)
def download_prices(tickers, start=None, end=None, period=None):
    tickers = [clean_ticker(t) for t in tickers if clean_ticker(t)]

    if len(tickers) == 0:
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
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        else:
            close = data.xs("Close", level=1, axis=1)
    else:
        close = data["Close"] if "Close" in data.columns else data

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close.columns = [clean_ticker(c) for c in close.columns]
    return close


def get_current_prices(tickers):
    tickers = [clean_ticker(t) for t in tickers if clean_ticker(t) and clean_ticker(t) != "CASH"]

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    prices = download_prices(tickers, period="7d")

    if prices.empty:
        return pd.Series(dtype=float)

    latest = prices.ffill().iloc[-1]
    return latest


def render_editable_table(title, df, key, file_path, column_config=None):
    st.markdown(f"### {title}")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key=key,
        column_config=column_config,
    )

    if st.button(f"Save {title}", key=f"save_{key}"):
        save_csv(edited_df, file_path)

    return edited_df.copy()


# =========================================================
# Data Loaders
# =========================================================

def load_portfolio():
    columns = {
        "Broker": "",
        "Ticker": "",
        "Quantity": 0.0,
        "AvgCost": 0.0,
    }

    df = read_csv_or_template(DATA_FILES["portfolio"], columns)
    df["Broker"] = df["Broker"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].apply(clean_ticker)
    df["Quantity"] = to_number(df["Quantity"])
    df["AvgCost"] = to_number(df["AvgCost"])
    return df


def load_banks():
    columns = {
        "Bank": "",
        "Account": "",
        "Balance": 0.0,
        "Currency": "THB",
    }

    df = read_csv_or_template(DATA_FILES["bank_accounts"], columns)
    df["Balance"] = to_number(df["Balance"])
    return df


def load_properties():
    columns = {
        "Property": "",
        "Type": "",
        "EstimatedValue": 0.0,
        "Location": "",
    }

    df = read_csv_or_template(DATA_FILES["properties"], columns)
    df["EstimatedValue"] = to_number(df["EstimatedValue"])
    return df


def load_mortgage():
    columns = {
        "Property": "",
        "OutstandingDebt": 0.0,
        "MonthlyPayment": 0.0,
        "InterestRate": 0.0,
    }

    df = read_csv_or_template(DATA_FILES["mortgage"], columns)
    df["OutstandingDebt"] = to_number(df["OutstandingDebt"])
    df["MonthlyPayment"] = to_number(df["MonthlyPayment"])
    df["InterestRate"] = to_number(df["InterestRate"])
    return df


def load_property_cashflow():
    columns = {
        "Property": "",
        "Month": "",
        "Rent": 0.0,
        "Expense": 0.0,
        "ExtraPayment": 0.0,
    }

    df = read_csv_or_template(DATA_FILES["property_cashflow"], columns)
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

    tickers = (
        portfolio.loc[portfolio["Ticker"] != "CASH", "Ticker"]
        .dropna()
        .unique()
        .tolist()
    )

    current_prices = get_current_prices(tickers)

    portfolio["CurrentPrice"] = portfolio["Ticker"].map(current_prices)
    portfolio.loc[portfolio["Ticker"] == "CASH", "CurrentPrice"] = 1
    portfolio["CurrentPrice"] = to_number(portfolio["CurrentPrice"])

    portfolio["CostBasis"] = portfolio["Quantity"] * portfolio["AvgCost"]
    portfolio["MarketValue"] = portfolio["Quantity"] * portfolio["CurrentPrice"]
    portfolio["PnL"] = portfolio["MarketValue"] - portfolio["CostBasis"]

    portfolio["ReturnPct"] = 0.0
    mask = portfolio["CostBasis"] != 0
    portfolio.loc[mask, "ReturnPct"] = (
        portfolio.loc[mask, "PnL"] / portfolio.loc[mask, "CostBasis"] * 100
    )

    return portfolio


def render_portfolio_section():
    st.subheader("Investment Portfolio")

    portfolio = load_portfolio()

    edited_portfolio = render_editable_table(
        title="Portfolio",
        df=portfolio,
        key="portfolio_editor",
        file_path=DATA_FILES["portfolio"],
        column_config={
            "Broker": st.column_config.TextColumn("Broker"),
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="เช่น AAPL, MSFT, NVDA, BTC-USD หรือ CASH"
            ),
            "Quantity": st.column_config.NumberColumn("Quantity", step=0.01),
            "AvgCost": st.column_config.NumberColumn("AvgCost", step=0.01),
        },
    )

    portfolio_calc = calculate_portfolio(edited_portfolio)

    if portfolio_calc.empty:
        st.info("ยังไม่มีข้อมูล Portfolio")
        return 0.0, pd.DataFrame()

    total_cost = portfolio_calc["CostBasis"].sum()
    investment_value = portfolio_calc["MarketValue"].sum()
    total_pnl = investment_value - total_cost
    total_return_pct = total_pnl / total_cost * 100 if total_cost else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Investment Value", f"{investment_value:,.2f}")
    col2.metric("Investment Gain/Loss", f"{total_pnl:,.2f}")
    col3.metric("Investment Return %", f"{total_return_pct:.2f}%")

    st.dataframe(portfolio_calc.round(2), use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)

    broker_summary = (
        portfolio_calc.groupby("Broker", dropna=False)["MarketValue"]
        .sum()
        .reset_index()
    )

    ticker_summary = (
        portfolio_calc.groupby("Ticker", dropna=False)["MarketValue"]
        .sum()
        .reset_index()
    )

    with chart_col1:
        if not broker_summary.empty:
            st.plotly_chart(
                px.pie(
                    broker_summary,
                    names="Broker",
                    values="MarketValue",
                    title="Allocation by Broker",
                ),
                use_container_width=True,
            )

    with chart_col2:
        if not ticker_summary.empty:
            st.plotly_chart(
                px.pie(
                    ticker_summary,
                    names="Ticker",
                    values="MarketValue",
                    title="Allocation by Ticker",
                ),
                use_container_width=True,
            )

    return investment_value, portfolio_calc


# =========================================================
# Bank Logic
# =========================================================

def render_bank_section():
    st.subheader("Bank Accounts")

    banks = load_banks()

    edited_banks = render_editable_table(
        title="Bank Accounts",
        df=banks,
        key="bank_editor",
        file_path=DATA_FILES["bank_accounts"],
        column_config={
            "Balance": st.column_config.NumberColumn("Balance", step=100.0),
        },
    )

    edited_banks["Balance"] = to_number(edited_banks["Balance"])
    bank_cash_value = edited_banks["Balance"].sum()

    st.metric("Total Bank Cash", f"{bank_cash_value:,.2f}")
    st.dataframe(edited_banks, use_container_width=True)

    if not edited_banks.empty and edited_banks["Balance"].sum() != 0:
        st.plotly_chart(
            px.pie(
                edited_banks,
                names="Bank",
                values="Balance",
                title="Bank Cash Allocation",
            ),
            use_container_width=True,
        )

    return bank_cash_value, edited_banks


# =========================================================
# Real Estate Logic
# =========================================================

def render_real_estate_section():
    st.subheader("Real Estate")

    properties = load_properties()
    mortgage = load_mortgage()
    cashflow = load_property_cashflow()

    prop_col, mortgage_col = st.columns(2)

    with prop_col:
        edited_properties = render_editable_table(
            title="Properties",
            df=properties,
            key="properties_editor",
            file_path=DATA_FILES["properties"],
            column_config={
                "EstimatedValue": st.column_config.NumberColumn(
                    "EstimatedValue",
                    step=10000.0,
                ),
            },
        )

    with mortgage_col:
        edited_mortgage = render_editable_table(
            title="Mortgage",
            df=mortgage,
            key="mortgage_editor",
            file_path=DATA_FILES["mortgage"],
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

    edited_cashflow = render_editable_table(
        title="Property Cash Flow",
        df=cashflow,
        key="cashflow_editor",
        file_path=DATA_FILES["property_cashflow"],
        column_config={
            "Rent": st.column_config.NumberColumn("Rent", step=1000.0),
            "Expense": st.column_config.NumberColumn("Expense", step=1000.0),
            "ExtraPayment": st.column_config.NumberColumn("ExtraPayment", step=1000.0),
        },
    )

    edited_properties["EstimatedValue"] = to_number(edited_properties["EstimatedValue"])
    edited_mortgage["OutstandingDebt"] = to_number(edited_mortgage["OutstandingDebt"])
    edited_cashflow["Rent"] = to_number(edited_cashflow["Rent"])
    edited_cashflow["Expense"] = to_number(edited_cashflow["Expense"])
    edited_cashflow["ExtraPayment"] = to_number(edited_cashflow["ExtraPayment"])

    real_estate = edited_properties.merge(
        edited_mortgage[["Property", "OutstandingDebt", "MonthlyPayment", "InterestRate"]],
        on="Property",
        how="left",
    )

    real_estate["OutstandingDebt"] = to_number(real_estate["OutstandingDebt"])
    real_estate["MonthlyPayment"] = to_number(real_estate["MonthlyPayment"])
    real_estate["InterestRate"] = to_number(real_estate["InterestRate"])
    real_estate["Equity"] = real_estate["EstimatedValue"] - real_estate["OutstandingDebt"]

    property_value = real_estate["EstimatedValue"].sum()
    property_debt = real_estate["OutstandingDebt"].sum()
    property_equity = real_estate["Equity"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Property Value", f"{property_value:,.2f}")
    col2.metric("Property Debt", f"{property_debt:,.2f}")
    col3.metric("Property Equity", f"{property_equity:,.2f}")

    st.markdown("### Real Estate Summary")
    st.dataframe(real_estate.round(2), use_container_width=True)

    if not real_estate.empty and real_estate["Equity"].sum() != 0:
        st.plotly_chart(
            px.pie(
                real_estate,
                names="Property",
                values="Equity",
                title="Real Estate Equity Allocation",
            ),
            use_container_width=True,
        )

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

def render_net_worth_summary(investment_value, bank_cash_value, property_value, property_debt, property_equity):
    st.header("Net Worth Summary")

    net_worth = investment_value + bank_cash_value + property_equity

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Investment Portfolio", f"{investment_value:,.2f}")
    col2.metric("Bank Cash", f"{bank_cash_value:,.2f}")
    col3.metric("Real Estate Equity", f"{property_equity:,.2f}")
    col4.metric("Net Worth", f"{net_worth:,.2f}")

    networth_df = pd.DataFrame(
        {
            "Category": [
                "Investment Portfolio",
                "Bank Cash",
                "Real Estate Equity",
            ],
            "Value": [
                investment_value,
                bank_cash_value,
                property_equity,
            ],
        }
    )

    st.dataframe(networth_df.round(2), use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if networth_df["Value"].sum() != 0:
            st.plotly_chart(
                px.pie(
                    networth_df,
                    names="Category",
                    values="Value",
                    title="Net Worth Allocation",
                ),
                use_container_width=True,
            )

    debt_df = pd.DataFrame(
        {
            "Category": [
                "Investment Portfolio",
                "Bank Cash",
                "Real Estate Gross Value",
                "Real Estate Debt",
                "Net Worth",
            ],
            "Value": [
                investment_value,
                bank_cash_value,
                property_value,
                -property_debt,
                net_worth,
            ],
        }
    )

    with chart_col2:
        st.plotly_chart(
            px.bar(
                debt_df,
                x="Category",
                y="Value",
                title="Assets, Debt, and Net Worth",
            ),
            use_container_width=True,
        )


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
    tickers = [t for t in tickers if t != "CASH"]
    return tickers


def render_market_analysis():
    st.header("Market Analysis")

    portfolio_tickers = get_portfolio_tickers_for_market()
    default_tickers = list(dict.fromkeys(list(DEFAULT_MARKET_ASSETS.keys()) + portfolio_tickers))

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
        clean_ticker(ticker)
        for ticker in ticker_input.split(",")
        if clean_ticker(ticker)
    ]

    selected_assets = list(dict.fromkeys(selected_assets))

    if start_date >= end_date:
        st.warning("วันที่เริ่มต้นต้องมาก่อนวันที่สิ้นสุด")
        return

    if len(selected_assets) == 0:
        st.warning("กรุณาเลือกสินทรัพย์อย่างน้อย 1 ตัว")
        return

    data = download_prices(
        selected_assets,
        start=start_date,
        end=end_date,
    )

    if data.empty:
        st.warning("ไม่พบข้อมูลราคาจาก yfinance สำหรับ ticker ที่เลือก")
        return

    data = data.ffill().bfill().dropna(axis=1, how="all")

    if data.empty:
        st.warning("ข้อมูลราคาว่างหลังจากล้างข้อมูล")
        return

    missing_tickers = sorted(set(selected_assets) - set(data.columns))
    if missing_tickers:
        st.info("Ticker ที่ดึงราคาไม่ได้: " + ", ".join(missing_tickers))

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

    momentum_days_map = {
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }

    momentum_days = momentum_days_map[momentum_choice]

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
        correlation = returns.corr()

        heatmap = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
        )

        heatmap.update_layout(
            template="plotly_dark",
            height=650,
        )

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

    rolling_max = data.cummax()
    drawdown = (data / rolling_max - 1) * 100
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

    rank_col1, rank_col2, rank_col3 = st.columns(3)

    with rank_col1:
        st.markdown("### Volatility Ranking")
        st.dataframe(
            risk_df.sort_values("Annual Volatility %", ascending=False)[
                ["Ticker", "Asset", "Annual Volatility %"]
            ].round(2),
            use_container_width=True,
        )

    with rank_col2:
        st.markdown("### Max Drawdown Ranking")
        st.dataframe(
            risk_df.sort_values("Max Drawdown %", ascending=True)[
                ["Ticker", "Asset", "Max Drawdown %"]
            ].round(2),
            use_container_width=True,
        )

    with rank_col3:
        st.markdown("### Sharpe Ratio Ranking")
        st.dataframe(
            risk_df.sort_values("Sharpe Ratio", ascending=False)[
                ["Ticker", "Asset", "Annual Return %", "Annual Volatility %", "Sharpe Ratio"]
            ].round(2),
            use_container_width=True,
        )

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
    risk_return_fig.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_title="Annual Volatility % (Risk)",
        yaxis_title="Annual Return %",
    )

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

    investment_value, portfolio_calc = render_portfolio_section()
    st.divider()

    bank_cash_value, banks = render_bank_section()
    st.divider()

    (
        property_value,
        property_debt,
        property_equity,
        real_estate,
        property_cashflow,
    ) = render_real_estate_section()

    st.divider()

    render_net_worth_summary(
        investment_value=investment_value,
        bank_cash_value=bank_cash_value,
        property_value=property_value,
        property_debt=property_debt,
        property_equity=property_equity,
    )

with tab_market:
    render_market_analysis()
