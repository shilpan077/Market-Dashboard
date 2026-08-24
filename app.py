import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Institutional Market Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "")
GOLDAPI_KEY = os.getenv("GOLDAPI_KEY", "")

try:
    if not MASSIVE_API_KEY:
        MASSIVE_API_KEY = st.secrets.get("MASSIVE_API_KEY", "")

    if not GOLDAPI_KEY:
        GOLDAPI_KEY = st.secrets.get("GOLDAPI_KEY", "")
except Exception:
    pass


MASSIVE_BASE_URL = "https://api.polygon.io"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def yfinance_symbol(symbol: str) -> str:
    mapping = {
        "GOLD": "GC=F",
        "XAU": "GC=F",
        "XAUUSD": "GC=F",
        "SILVER": "SI=F",
        "XAG": "SI=F",
        "XAGUSD": "SI=F",
    }
    return mapping.get(symbol, symbol)


def is_precious_metal_spot(symbol: str) -> bool:
    return symbol in {"GOLD", "XAU", "XAUUSD", "SILVER", "XAG", "XAGUSD"}


def metal_code(symbol: str) -> str:
    if symbol in {"GOLD", "XAU", "XAUUSD"}:
        return "XAU"
    return "XAG"


def safe_float(value, default=np.nan):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except Exception:
        return default


# ============================================================
# MASSIVE / POLYGON DATA
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def massive_latest_quote(symbol: str):
    if not MASSIVE_API_KEY:
        return None

    symbol = clean_symbol(symbol)

    url = f"{MASSIVE_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"

    try:
        response = requests.get(
            url,
            params={"apiKey": MASSIVE_API_KEY},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        ticker = payload.get("ticker", {})

        last_trade = ticker.get("lastTrade", {})
        day_data = ticker.get("day", {})
        previous_day = ticker.get("prevDay", {})

        price = (
            last_trade.get("p")
            or day_data.get("c")
            or ticker.get("lastQuote", {}).get("P")
        )

        previous_close = previous_day.get("c")
        price = safe_float(price)
        previous_close = safe_float(previous_close)

        if not np.isfinite(price):
            return None

        change = np.nan
        change_percent = np.nan

        if np.isfinite(previous_close) and previous_close != 0:
            change = price - previous_close
            change_percent = change / previous_close * 100

        return {
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "source": "Massive real-time snapshot",
            "delayed": False,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def massive_daily_history(symbol: str, years: int = 3):
    if not MASSIVE_API_KEY:
        return None

    symbol = clean_symbol(symbol)

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365 * years)

    url = (
        f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{start_date}/{end_date}"
    )

    try:
        response = requests.get(
            url,
            params={
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": MASSIVE_API_KEY,
            },
            timeout=30,
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            return None

        rows = []

        for item in results:
            rows.append(
                {
                    "date": pd.to_datetime(item["t"], unit="ms"),
                    "open": safe_float(item.get("o")),
                    "high": safe_float(item.get("h")),
                    "low": safe_float(item.get("l")),
                    "close": safe_float(item.get("c")),
                    "volume": safe_float(item.get("v"), 0),
                }
            )

        dataframe = pd.DataFrame(rows)
        dataframe = dataframe.set_index("date")
        return dataframe

    except Exception:
        return None


# ============================================================
# GOLDAPI SPOT DATA
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def goldapi_latest_quote(symbol: str):
    if not GOLDAPI_KEY:
        return None

    code = metal_code(symbol)
    url = f"https://app.goldapi.net/price/{code}/USD"

    try:
        response = requests.get(
            url,
            headers={"x-api-key": GOLDAPI_KEY},
            params={"x-api-key": GOLDAPI_KEY},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        price = safe_float(payload.get("price"))
        previous_close = safe_float(
            payload.get("prev_close") or payload.get("previous_close")
        )

        if not np.isfinite(price):
            return None

        change = safe_float(payload.get("ch"))
        change_percent = safe_float(payload.get("chp"))

        if not np.isfinite(change) and np.isfinite(previous_close):
            change = price - previous_close

        if not np.isfinite(change_percent) and np.isfinite(previous_close):
            change_percent = change / previous_close * 100

        return {
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "source": f"GoldAPI {code}/USD spot",
            "delayed": False,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception:
        return None


# ============================================================
# YFINANCE FALLBACK DATA
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def yfinance_history(symbol: str, years: int = 3):
    yf_symbol = yfinance_symbol(symbol)

    try:
        data = yf.download(
            yf_symbol,
            period=f"{years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [
            str(column).lower().replace(" ", "_")
            for column in data.columns
        ]

        rename_map = {
            "adj_close": "adjclose",
        }

        data = data.rename(columns=rename_map)

        required = ["open", "high", "low", "close", "volume"]

        for column in required:
            if column not in data.columns:
                return None

        data = data[required].copy()
        data.index = pd.to_datetime(data.index)
        data = data.dropna(subset=["open", "high", "low", "close"])

        return data

    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def yfinance_latest_quote(symbol: str):
    yf_symbol = yfinance_symbol(symbol)

    try:
        data = yf.download(
            yf_symbol,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(c).lower() for c in data.columns]
        data = data.dropna(subset=["close"])

        if len(data) == 0:
            return None

        price = safe_float(data["close"].iloc[-1])
        previous_close = (
            safe_float(data["close"].iloc[-2])
            if len(data) > 1
            else np.nan
        )

        change = price - previous_close
        change_percent = (
            change / previous_close * 100
            if np.isfinite(previous_close) and previous_close != 0
            else np.nan
        )

        return {
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "source": "Yahoo Finance fallback",
            "delayed": True,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception:
        return None


# ============================================================
# DATA LOADING
# ============================================================

def load_quote(symbol: str):
    symbol = clean_symbol(symbol)

    if is_precious_metal_spot(symbol):
        quote = goldapi_latest_quote(symbol)
        if quote is not None:
            return quote

    quote = massive_latest_quote(symbol)
    if quote is not None:
        return quote

    return yfinance_latest_quote(symbol)


def load_history(symbol: str):
    symbol = clean_symbol(symbol)

    if not is_precious_metal_spot(symbol):
        data = massive_daily_history(symbol)
        if data is not None and not data.empty:
            return data

    return yfinance_history(symbol)


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / 14, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / 14, adjust=False).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + relative_strength))
    df["rsi"] = df["rsi"].fillna(50)

    df["macd"] = (
        df["close"].ewm(span=12, adjust=False).mean()
        - df["close"].ewm(span=26, adjust=False).mean()
    )

    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    df["volume_average"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = (
        df["volume"] / df["volume_average"].replace(0, np.nan)
    )

    df["trend_score"] = (
        (df["close"] > df["ema21"]).astype(int)
        + (df["close"] > df["ema50"]).astype(int)
        + (df["close"] > df["ema200"]).astype(int)
    )

    df["recent_high"] = df["high"].rolling(20).max().shift(1)
    df["recent_low"] = df["low"].rolling(20).min().shift(1)

    df["bullish_bos"] = df["close"] > df["recent_high"]
    df["bearish_bos"] = df["close"] < df["recent_low"]

    df["buy_condition"] = (
        (df["trend_score"] >= 2)
        & (df["macd"] > df["macd_signal"])
        & (df["rsi"] > 50)
        & (df["rsi"] < 70)
        & (df["volume_ratio"] >= 1.5)
    )

    df["sell_condition"] = (
        (df["trend_score"] <= 1)
        & (df["macd"] < df["macd_signal"])
        & (df["rsi"] < 50)
        & (df["rsi"] > 30)
        & (df["volume_ratio"] >= 1.5)
    )

    return df


def get_analysis(df: pd.DataFrame):
    latest = df.iloc[-1]

    close = safe_float(latest["close"])
    ema21 = safe_float(latest["ema21"])
    ema50 = safe_float(latest["ema50"])
    ema200 = safe_float(latest["ema200"])
    rsi = safe_float(latest["rsi"])
    atr = safe_float(latest["atr"])
    volume_ratio = safe_float(latest["volume_ratio"], 0)
    trend_score = int(safe_float(latest["trend_score"], 0))

    bullish_stack = ema21 > ema50 > ema200
    bearish_stack = ema21 < ema50 < ema200

    macd_bullish = latest["macd"] > latest["macd_signal"]
    macd_bearish = latest["macd"] < latest["macd_signal"]

    if bullish_stack and macd_bullish:
        trend = "STRONG BULLISH"
    elif trend_score >= 2 and macd_bullish:
        trend = "BULLISH"
    elif bearish_stack and macd_bearish:
        trend = "STRONG BEARISH"
    elif trend_score <= 1 and macd_bearish:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    buy_signal = bool(latest["buy_condition"])
    sell_signal = bool(latest["sell_condition"])

    if buy_signal:
        verdict = "BUY BIAS"
    elif sell_signal:
        verdict = "SELL BIAS"
    else:
        verdict = "WAIT / NEUTRAL"

    if np.isfinite(atr):
        buy_stop = close - atr * 1.5
        buy_target = close + atr * 3.0
        sell_stop = close + atr * 1.5
        sell_target = close - atr * 3.0
    else:
        buy_stop = np.nan
        buy_target = np.nan
        sell_stop = np.nan
        sell_target = np.nan

    return {
        "close": close,
        "rsi": rsi,
        "atr": atr,
        "volume_ratio": volume_ratio,
        "trend_score": trend_score,
        "trend": trend,
        "verdict": verdict,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
        "buy_stop": buy_stop,
        "buy_target": buy_target,
        "sell_stop": sell_stop,
        "sell_target": sell_target,
        "bullish_bos": bool(latest["bullish_bos"]),
        "bearish_bos": bool(latest["bearish_bos"]),
    }


# ============================================================
# FRED MACRO DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fred_latest(series_id: str):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    try:
        data = pd.read_csv(
            url,
            params={"id": series_id},
            timeout=20,
        )

        if data.empty:
            return np.nan, None

        value_column = data.columns[-1]
        data[value_column] = pd.to_numeric(
            data[value_column],
            errors="coerce",
        )

        data = data.dropna(subset=[value_column])

        if data.empty:
            return np.nan, None

        latest_value = safe_float(data[value_column].iloc[-1])
        latest_date = data.iloc[-1].iloc[0]

        return latest_value, latest_date

    except Exception:
        return np.nan, None


# ============================================================
# CHART
# ============================================================

def build_chart(df: pd.DataFrame, symbol: str):
    chart = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=[
            f"{symbol} Daily Price",
            "Volume",
            "RSI",
        ],
    )

    chart.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema21"],
            name="EMA 21",
            line={"color": "yellow", "width": 1},
        ),
        row=1,
        col=1,
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema50"],
            name="EMA 50",
            line={"color": "orange", "width": 2},
        ),
        row=1,
        col=1,
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema200"],
            name="EMA 200",
            line={"color": "white", "width": 2},
        ),
        row=1,
        col=1,
    )

    volume_colors = np.where(
        df["close"] >= df["open"],
        "rgba(0,180,80,0.65)",
        "rgba(220,50,50,0.65)",
    )

    chart.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color=volume_colors,
        ),
        row=2,
        col=1,
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["volume_average"],
            name="Volume Average",
            line={"color": "orange", "width": 1},
        ),
        row=2,
        col=1,
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["rsi"],
            name="RSI",
            line={"color": "cyan", "width": 2},
        ),
        row=3,
        col=1,
    )

    chart.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        row=3,
        col=1,
    )

    chart.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        row=3,
        col=1,
    )

    buy_points = df[df["buy_condition"]]

    if not buy_points.empty:
        chart.add_trace(
            go.Scatter(
                x=buy_points.index,
                y=buy_points["low"],
                mode="markers",
                name="Buy Signal",
                marker={
                    "color": "lime",
                    "size": 10,
                    "symbol": "triangle-up",
                },
            ),
            row=1,
            col=1,
        )

    sell_points = df[df["sell_condition"]]

    if not sell_points.empty:
        chart.add_trace(
            go.Scatter(
                x=sell_points.index,
                y=sell_points["high"],
                mode="markers",
                name="Sell Signal",
                marker={
                    "color": "red",
                    "size": 10,
                    "symbol": "triangle-down",
                },
            ),
            row=1,
            col=1,
        )

    chart.update_layout(
        height=850,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )

    chart.update_yaxes(title_text="Price", row=1, col=1)
    chart.update_yaxes(title_text="Volume", row=2, col=1)
    chart.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    return chart


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Market Dashboard")

default_symbols = [
    "AAPL",
    "MSFT",
    "NVDA",
    "SPY",
    "QQQ",
    "SLV",
    "GLD",
    "XAGUSD",
    "XAUUSD",
    "SI=F",
    "GC=F",
]

selected_symbol = st.sidebar.selectbox(
    "Select asset",
    default_symbols,
    index=0,
)

custom_symbol = st.sidebar.text_input(
    "Or enter another symbol",
    placeholder="Example: TSLA",
)

symbol = clean_symbol(custom_symbol or selected_symbol)

auto_refresh = st.sidebar.checkbox(
    "Auto-refresh",
    value=True,
)

refresh_seconds = st.sidebar.slider(
    "Refresh interval in seconds",
    min_value=15,
    max_value=300,
    value=30,
    step=15,
)

if auto_refresh and st_autorefresh is not None:
    st_autorefresh(
        interval=refresh_seconds * 1000,
        key="market_refresh",
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("Institutional-Style Daily Market Dashboard")

st.caption(
    "Rule-based technical, volume, structure, risk, and macro analysis. "
    "This is decision-support software, not a guarantee of future returns."
)

quote = load_quote(symbol)
history = load_history(symbol)

if quote is None:
    st.error(
        "No quote data was returned. Check the symbol or configure a live "
        "market-data provider."
    )
    st.stop()

if history is None or history.empty:
    st.error(
        "No historical data was returned. Check the symbol and data-provider "
        "permissions."
    )
    st.stop()

df = calculate_indicators(history)
analysis = get_analysis(df)

if quote["delayed"]:
    st.warning(
        "The current quote is using fallback data and may be delayed. "
        "Add MASSIVE_API_KEY or GOLDAPI_KEY for live data."
    )
else:
    st.success(f"Live provider: {quote['source']}")


# ============================================================
# TOP METRICS
# ============================================================

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

metric_1.metric(
    "Last Price",
    f"{quote['price']:,.4f}",
    f"{quote['change_percent']:+.2f}%",
)

metric_2.metric(
    "Trend",
    analysis["trend"],
)

metric_3.metric(
    "Trend Score",
    f"{analysis['trend_score']}/3",
)

metric_4.metric(
    "RSI",
    f"{analysis['rsi']:.1f}",
)

metric_5.metric(
    "Volume Ratio",
    f"{analysis['volume_ratio']:.2f}x",
)


# ============================================================
# SIGNAL PANEL
# ============================================================

st.subheader("Daily Analysis")

panel_1, panel_2, panel_3, panel_4 = st.columns(4)

panel_1.metric(
    "Verdict",
    analysis["verdict"],
)

panel_2.metric(
    "ATR",
    f"{analysis['atr']:,.4f}",
)

panel_3.metric(
    "Buy Stop Guide",
    f"{analysis['buy_stop']:,.4f}",
)

panel_4.metric(
    "Buy Target Guide",
    f"{analysis['buy_target']:,.4f}",
)

if analysis["buy_signal"]:
    st.success(
        "BUY BIAS: trend, momentum, and volume conditions are aligned."
    )
elif analysis["sell_signal"]:
    st.error(
        "SELL BIAS: bearish trend, momentum, and volume conditions are aligned."
    )
else:
    st.info(
        "WAIT: the required confirmation conditions are not fully aligned."
    )


# ============================================================
# CHART
# ============================================================

st.plotly_chart(
    build_chart(df.tail(500), symbol),
    use_container_width=True,
)


# ============================================================
# MACRO PANEL
# ============================================================

st.subheader("Macro Reference Data")

macro_1, macro_2, macro_3 = st.columns(3)

ten_year, ten_year_date = fred_latest("DGS10")
real_yield, real_yield_date = fred_latest("DFII10")
fed_funds, fed_funds_date = fred_latest("FEDFUNDS")

macro_1.metric(
    "10Y Treasury Yield",
    f"{ten_year:.2f}%" if np.isfinite(ten_year) else "Unavailable",
)

macro_2.metric(
    "10Y Real Yield",
    f"{real_yield:.2f}%" if np.isfinite(real_yield) else "Unavailable",
)

macro_3.metric(
    "Effective Fed Funds",
    f"{fed_funds:.2f}%" if np.isfinite(fed_funds) else "Unavailable",
)

st.caption(
    "Macro values may be daily or monthly depending on the official series."
)


# ============================================================
# DATA TABLE
# ============================================================

with st.expander("View latest calculations"):
    output_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema21",
        "ema50",
        "ema200",
        "rsi",
        "atr",
        "volume_ratio",
        "trend_score",
        "macd",
        "macd_signal",
    ]

    available_columns = [
        column for column in output_columns if column in df.columns
    ]

    st.dataframe(
        df[available_columns].tail(20).round(4),
        use_container_width=True,
    )


st.caption(
    f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
