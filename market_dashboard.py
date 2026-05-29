import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

assets = {
    # US Market / Themes
    "SPY": "US Market",
    "QQQ": "US Tech / AI",
    "SOXX": "Semiconductor",
    "XLV": "Healthcare",
    "ITA": "Aerospace",
    "XLE": "Energy",
    "BRK-B": "Berkshire",

    # Countries
    "INDA": "India",
    "MCHI": "China",
    "THD": "Thailand ETF",

    # Macro Assets
    "GLD": "Gold",
    "BTC-USD": "Bitcoin"
}

years = 5
tickers = list(assets.keys())

data = yf.download(
    tickers,
    period="10y",
    auto_adjust=True
)["Close"]

data = data.ffill()
data = data.tail(252 * years)

normalized = data.div(data.iloc[0]).mul(100)

fig = go.Figure()

for ticker in normalized.columns:
    fig.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[ticker],
            mode="lines",
            name=f"{ticker} - {assets[ticker]}"
        )
    )

fig.update_layout(
    title=f"Macro Dashboard - Last {years} Years",
    template="plotly_dark",
    height=850,
    width=1600,
    hovermode="x unified",
    yaxis=dict(
        type="log",
        title="Normalized Performance (Log Scale)"
    ),
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    )
)

fig.show()

# -------------------------
# Momentum Ranking
# -------------------------

momentum_days = 63

momentum = (data.iloc[-1] / data.iloc[-momentum_days] - 1) * 100
momentum = momentum.sort_values(ascending=False)

print("\n" + "=" * 50)
print("Momentum Ranking (Last 3 Months)")
print("=" * 50)

for ticker, value in momentum.items():
    print(f"{ticker:10s} {assets[ticker]:20s} {value:8.2f}%")

# -------------------------
# Correlation Heatmap
# -------------------------

returns = data.pct_change().dropna()
correlation = returns.corr()

heatmap = px.imshow(
    correlation,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title=f"Correlation Heatmap - Last {years} Years"
)

heatmap.update_layout(
    template="plotly_dark",
    height=900,
    width=1200
)

heatmap.show()