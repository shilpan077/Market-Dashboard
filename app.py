import os
from io import StringIO
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
    page_title="Market Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CONFIGURATION AND API KEYS
# ============================================================

def read_secret(name: str) -> str:
    """
    Reads a secret from Streamlit Cloud Secrets first,
    then from environment variables.
    """
    try:
        value = st.secrets[name]
    except Exception:
        value = os.getenv(name, "")

    return str(value or "").strip().strip('"').strip("'")


MASSIVE_API_KEY = read_secret("MASSIVE_API_KEY")
GOLDAPI_KEY = read_secret("GOLDAPI_KEY")

# Current Massive API base URL
MASSIVE_BASE_URL = "https://api.massive.com"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def safe_float(value, default=np.nan):
    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return default

    except Exception:
        return default


def format_percent(value) -> str:
    value = safe_float(value)

    if np.isfinite(value):
        return f"{value:+.2f}%"

    return "Unavailable"


def yfinance_symbol(symbol: str) -> str:
    """
    Converts friendly metal names into Yahoo Finance symbols.
    """
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
    return symbol in {
        "GOLD",
        "XAU",
        "XAUUSD",
        "SILVER",
        "XAG",
        "XAGUSD",
    }


def metal_code(symbol: str) -> str:
    if symbol in {"GOLD", "XAU", "XAUUSD"}:
        return "XAU"

    return "XAG"


def normalize_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Makes yfinance columns consistent when yfinance returns
    normal columns or MultiIndex columns.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).lower().replace(" ", "_")
        for column in data.columns
    ]

    return data


# ============================================================
# MASSIVE DATA
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def massive_latest_quote(symbol: str, api_key: str):
    """
    Gets the latest U.S. stock or ETF snapshot from Massive.
    """
    if not api_key:
        return None

    symbol = clean_symbol(symbol)

    url = (
        f"{MASSIVE_BASE_URL}/v2/snapshot/locale/us/"
        f"markets/stocks/tickers/{symbol}"
    )

    try:
        response = requests.get(
            url,
            params={"apiKey": api_key},
            timeout=20,
        )

        if response.status_code != 200:
            return None

        payload = response.json()
        ticker = payload.get("ticker") or {}

        last_trade = ticker.get("lastTrade") or {}
        last_quote = ticker.get("lastQuote") or {}
        day_data = ticker.get("day") or {}
        previous_day = ticker.get("prevDay") or {}

        price = (
            last_trade.get("p")
            or day_data.get("c")
            or last_quote.get("P")
            or last_quote.get("p")
        )

        previous_close = previous_day.get("c")

        price = safe_float(price)
        previous_close = safe_float(previous_close)

        if not np.isfinite(price):
            return None

        if np.isfinite(previous_close) and previous_close != 0:
            change = price - previous_close
            change_percent = change / previous_close * 100
        else:
            change = np.nan
            change_percent = np.nan

        return {
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "source": "Massive",
            "delayed": False,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def massive_daily_history(
    symbol: str,
    years: int,
    api_key: str,
):
    """
    Gets daily historical candles from Massive.
    """
    if not api_key:
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
                "apiKey": api_key,
            },
            timeout=30,
        )

        if response.status_code != 200:
            return None

        payload = response.json()
        results = payload.get("results") or []

        if not results:
            return None

        rows = []

        for item in results:
            rows.append(
                {
                    "date": pd.to_datetime(
                        item.get("t"),
                        unit="ms",
                        errors="coerce",
                    ),
                    "open": safe_float(item.get("o")),
                    "high": safe_float(item.get("h")),
                    "low": safe_float(item.get("l")),
                    "close": safe_float(item.get("c")),
                    "volume": safe_float(item.get("v"), 0),
                }
            )

        dataframe = pd.DataFrame(rows)

        if dataframe.empty:
            return None

        dataframe = dataframe.dropna(
            subset=["date", "open", "high", "low", "close"]
        )

        dataframe = dataframe.set_index("date")
        dataframe = dataframe.sort_index()

        return dataframe

    except Exception:
        return None


# ============================================================
# GOLDAPI DATA
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def goldapi_latest_quote(symbol: str, api_key: str):
    """
    Gets current gold or silver spot prices from GoldAPI.

    Gold:
        XAU/USD

    Silver:
        XAG/USD
    """
    if not api_key:
        return None

    code = metal_code(symbol)

    url = f"https://app.goldapi.net/price/{code}/USD"

    try:
        response = requests.get(
            url,
            params={"x-api-key": api_key},
            timeout=20,
        )

        if response.status_code != 200:
            return None

        payload = response.json()

        price = safe_float(payload.get("price"))

        previous_close = safe_float(
            payload.get("prev_close")
            or payload.get("previous_close")
        )

        if not np.isfinite(price):
            return None

        change = safe_float(payload.get("ch"))
        change_percent = safe_float(payload.get("chp"))

        if not np.isfinite(change) and np.isfinite(previous_close):
            change = price - previous_close

        if not np.isfinite(change_percent):
            if np.isfinite(previous_close) and previous_close != 0:
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
# YAHOO FINANCE FALLBACK DATA
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def yfinance_history(symbol: str, years: int = 3):
    """
    Fallback historical data source.
    Data can be delayed.
    """
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

        data = normalize_yfinance_columns(data)

        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in required_columns:
            if column not in data.columns:
                return None

        data = data[required_columns].copy()
        data.index = pd.to_datetime(data.index)

        data = data.dropna(
            subset=["open", "high", "low", "close"]
        )

        data = data.sort_index()

        if data.empty:
            return None

        return data

    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def yfinance_latest_quote(symbol: str):
    """
    Fallback latest quote.
    This data may be delayed.
    """
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

        data = normalize_yfinance_columns(data)

        if "close" not in data.columns:
            return None

        data = data.dropna(subset=["close"])

        if data.empty:
            return None

        price = safe_float(data["close"].iloc[-1])

        if len(data) > 1:
            previous_close = safe_float(data["close"].iloc[-2])
        else:
            previous_close = np.nan

        if (
            np.isfinite(price)
            and np.isfinite(previous_close)
            and previous_close != 0
        ):
            change = price - previous_close
            change_percent = change / previous_close * 100
        else:
            change = np.nan
            change_percent = np.nan

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

    # GoldAPI is used for gold/silver spot prices
    if is_precious_metal_spot(symbol):
        quote = goldapi_latest_quote(symbol, GOLDAPI_KEY)

        if quote is not None:
            return quote

    # Massive is used for U.S. stocks and ETFs
    quote = massive_latest_quote(symbol, MASSIVE_API_KEY)

    if quote is not None:
        return quote

    # Fallback
    return yfinance_latest_quote(symbol)


def load_history(symbol: str):
    symbol = clean_symbol(symbol)

    # Massive historical data for U.S. stocks and ETFs
    if not is_precious_metal_spot(symbol):
        data = massive_daily_history(
            symbol,
            years=3,
            api_key=MASSIVE_API_KEY,
        )

        if data is not None and not data.empty:
            return data

    # Fallback history, including gold and silver futures
    return yfinance_history(symbol, years=3)


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Moving averages
    df["ema21"] = df["close"].ewm(
        span=21,
        adjust=False,
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    # RSI
    delta = df["close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    df["rsi"] = 100 - (
        100 / (1 + relative_strength)
    )

    df["rsi"] = df["rsi"].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(50)

    # MACD
    fast_ema = df["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    slow_ema = df["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["macd"] = fast_ema - slow_ema

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    # ATR
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    # Volume
    df["volume_average"] = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"]
        / df["volume_average"].replace(0, np.nan)
    )

    # Trend score
    df["trend_score"] = (
        (df["close"] > df["ema21"]).astype(int)
        + (df["close"] > df["ema50"]).astype(int)
        + (df["close"] > df["ema200"]).astype(int)
    )

    # Breakout structure
    df["recent_high"] = (
        df["high"].rolling(20).max().shift(1)
    )

    df["recent_low"] = (
        df["low"].rolling(20).min().shift(1)
    )

    df["bullish_bos"] = (
        df["close"] > df["recent_high"]
    )

    df["bearish_bos"] = (
        df["close"] < df["recent_low"]
    )

    # Buy and sell conditions
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

    close = safe_float(latest.get("close"))
    ema21 = safe_float(latest.get("ema21"))
    ema50 = safe_float(latest.get("ema50"))
    ema200 = safe_float(latest.get("ema200"))
    rsi = safe_float(latest.get("rsi"), 50)
    atr = safe_float(latest.get("atr"))
    volume_ratio = safe_float(
        latest.get("volume_ratio"),
        0,
    )

    trend_score = int(
        safe_float(
            latest.get("trend_score"),
            0,
        )
    )

    macd = safe_float(latest.get("macd"))
    macd_signal = safe_float(
        latest.get("macd_signal")
    )

    bullish_stack = (
        np.isfinite(ema21)
        and np.isfinite(ema50)
        and np.isfinite(ema200)
        and ema21 > ema50 > ema200
    )

    bearish_stack = (
        np.isfinite(ema21)
        and np.isfinite(ema50)
        and np.isfinite(ema200)
        and ema21 < ema50 < ema200
    )

    macd_bullish = (
        np.isfinite(macd)
        and np.isfinite(macd_signal)
        and macd > macd_signal
    )

    macd_bearish = (
        np.isfinite(macd)
        and np.isfinite(macd_signal)
        and macd < macd_signal
    )

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

    buy_signal = bool(
        latest.get("buy_condition", False)
    )

    sell_signal = bool(
        latest.get("sell_condition", False)
    )

    if buy_signal:
        verdict = "BUY BIAS"
    elif sell_signal:
        verdict = "SELL BIAS"
    else:
        verdict = "WAIT / NEUTRAL"

    if np.isfinite(close) and np.isfinite(atr):
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
        "bullish_bos": bool(
            latest.get("bullish_bos", False)
        ),
        "bearish_bos": bool(
            latest.get("bearish_bos", False)
        ),
    }


# ============================================================
# FRED MACRO DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fred_latest(series_id: str):
    """
    Gets the latest value from FRED.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    try:
        response = requests.get(
            url,
            params={"id": series_id},
            timeout=20,
        )

        response.raise_for_status()

        data = pd.read_csv(
            StringIO(response.text)
        )

        if data.empty:
            return np.nan, None

        value_column = data.columns[-1]

        data[value_column] = pd.to_numeric(
            data[value_column],
            errors="coerce",
        )

        data = data.dropna(
            subset=[value_column]
        )

        if data.empty:
            return np.nan, None

        latest_value = safe_float(
            data[value_column].iloc[-1]
        )

        latest_date = data.iloc[-1].iloc[0]

        return latest_value, latest_date

    except Exception:
        return np.nan, None


# ============================================================
# CHART
# ============================================================

def build_chart(
    df: pd.DataFrame,
    symbol: str,
):
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

    # Candlestick chart
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

    # Moving averages
    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema21"],
            name="EMA 21",
            line={
                "color": "yellow",
                "width": 1,
            },
        ),
        row=1,
        col=1,
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema50"],
            name="EMA 50",
            line={
                "color": "orange",
                "width": 2,
            },
        ),
        row=1,
        col=1,
    )

    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema200"],
            name="EMA 200",
            line={
                "color": "white",
                "width": 2,
            },
        ),
        row=1,
        col=1,
    )

    # Volume colors
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
            line={
                "color": "orange",
                "width": 1,
            },
        ),
        row=2,
        col=1,
    )

    # RSI
    chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["rsi"],
            name="RSI",
            line={
                "color": "cyan",
                "width": 2,
            },
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

    # Buy signals
    buy_points = df[
        df["buy_condition"].fillna(False)
    ]

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

    # Sell signals
    sell_points = df[
        df["sell_condition"].fillna(False)
    ]

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
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
    )

    chart.update_yaxes(
        title_text="Price",
        row=1,
        col=1,
    )

    chart.update_yaxes(
        title_text="Volume",
        row=2,
        col=1,
    )

    chart.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        row=3,
        col=1,
    )

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
    "GLD",
    "SLV",
    "XAUUSD",
    "XAGUSD",
    "GC=F",
    "SI=F",
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

symbol = clean_symbol(
    custom_symbol or selected_symbol
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "API connection status:"
)

st.sidebar.caption(
    f"Massive key loaded: "
    f"{'YES' if MASSIVE_API_KEY else 'NO'}"
)

st.sidebar.caption(
    f"GoldAPI key loaded: "
    f"{'YES' if GOLDAPI_KEY else 'NO'}"
)

st.sidebar.markdown("---")

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

st.title(
    "Institutional-Style Market Dashboard"
)

st.caption(
    "Technical, volume, structure, risk, and macro analysis. "
    "This is decision-support software, not guaranteed financial advice."
)

quote = load_quote(symbol)
history = load_history(symbol)

if quote is None:
    st.error(
        "No quote data was returned. "
        "Check the symbol, API key, or data-provider access."
    )

    with st.expander("Connection checklist"):
        st.write(
            f"Selected symbol: `{symbol}`"
        )
        st.write(
            f"Massive key loaded: "
            f"`{'YES' if MASSIVE_API_KEY else 'NO'}`"
        )
        st.write(
            f"GoldAPI key loaded: "
            f"`{'YES' if GOLDAPI_KEY else 'NO'}`"
        )
        st.write(
            "For testing Massive, select `AAPL`."
        )
        st.write(
            "For testing GoldAPI, select `XAUUSD` or `XAGUSD`."
        )

    st.stop()

if history is None or history.empty:
    st.error(
        "No historical data was returned. "
        "Check the symbol or provider permissions."
    )
    st.stop()

# Calculate indicators
df = calculate_indicators(history)

if df.empty:
    st.error("The historical dataset is empty.")
    st.stop()

analysis = get_analysis(df)

# Provider notice
if quote["delayed"]:
    st.warning(
        "Fallback data is being used and may be delayed. "
        "Check your API keys and market-data permissions."
    )
else:
    st.success(
        f"Connected provider: {quote['source']}"
    )


# ============================================================
# TOP METRICS
# ============================================================

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

metric_1.metric(
    "Last Price",
    f"{quote['price']:,.4f}",
    format_percent(quote["change_percent"]),
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
# DAILY ANALYSIS
# ============================================================

st.subheader("Daily Analysis")

panel_1, panel_2, panel_3, panel_4 = st.columns(4)

panel_1.metric(
    "Verdict",
    analysis["verdict"],
)

panel_2.metric(
    "ATR",
    f"{analysis['atr']:,.4f}"
    if np.isfinite(analysis["atr"])
    else "Unavailable",
)

panel_3.metric(
    "Buy Stop Guide",
    f"{analysis['buy_stop']:,.4f}"
    if np.isfinite(analysis["buy_stop"])
    else "Unavailable",
)

panel_4.metric(
    "Buy Target Guide",
    f"{analysis['buy_target']:,.4f}"
    if np.isfinite(analysis["buy_target"])
    else "Unavailable",
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
    build_chart(
        df.tail(500),
        symbol,
    ),
    use_container_width=True,
)


# ============================================================
# MACRO DATA
# ============================================================

st.subheader("Macro Reference Data")

macro_1, macro_2, macro_3 = st.columns(3)

ten_year, ten_year_date = fred_latest("DGS10")
real_yield, real_yield_date = fred_latest("DFII10")
fed_funds, fed_funds_date = fred_latest("FEDFUNDS")

macro_1.metric(
    "10Y Treasury Yield",
    f"{ten_year:.2f}%"
    if np.isfinite(ten_year)
    else "Unavailable",
)

macro_2.metric(
    "10Y Real Yield",
    f"{real_yield:.2f}%"
    if np.isfinite(real_yield)
    else "Unavailable",
)

macro_3.metric(
    "Effective Fed Funds",
    f"{fed_funds:.2f}%"
    if np.isfinite(fed_funds)
    else "Unavailable",
)

st.caption(
    "Macro values can be daily or monthly depending on the official series."
)


# ============================================================
# LATEST CALCULATIONS TABLE
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
        "macd_histogram",
    ]

    available_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    st.dataframe(
        df[available_columns].tail(20).round(4),
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Last refresh: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
