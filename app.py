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
    page_title="Institutional Market Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# API CONFIGURATION
# ============================================================

def read_secret(name: str) -> str:
    """
    Reads a secret from Streamlit Cloud Secrets.
    Environment variables are used as a backup.
    """
    value = ""

    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if not value:
        value = os.getenv(name, "")

    return str(value or "").strip().strip('"').strip("'")


MASSIVE_API_KEY = read_secret("MASSIVE_API_KEY")
GOLDAPI_KEY = read_secret("GOLDAPI_KEY")

# Current Massive API URL
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


def format_delta(value) -> str:
    value = safe_float(value)

    if np.isfinite(value):
        return f"{value:+.2f}%"

    return "Unavailable"


def format_number(value, decimals=4) -> str:
    value = safe_float(value)

    if np.isfinite(value):
        return f"{value:,.{decimals}f}"

    return "Unavailable"


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


def yfinance_symbol(symbol: str) -> str:
    """
    Converts friendly metal symbols into Yahoo Finance futures symbols.
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


def normalize_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Makes yfinance column names consistent.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).lower().replace(" ", "_")
        for column in data.columns
    ]

    return data


# ============================================================
# MASSIVE API
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def massive_latest_quote(symbol: str, api_key: str):
    """
    Gets the latest stock or ETF quote from Massive.

    Returns:
        quote, status_message
    """
    if not api_key:
        return None, "Massive API key is empty."

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
            return None, f"Massive HTTP {response.status_code}"

        payload = response.json()
        ticker = payload.get("ticker") or {}

        if not ticker:
            return None, "Massive returned no ticker data."

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
            return None, "Massive returned no usable price."

        if np.isfinite(previous_close) and previous_close != 0:
            change = price - previous_close
            change_percent = change / previous_close * 100
        else:
            change = np.nan
            change_percent = np.nan

        quote = {
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "source": "Massive",
            "delayed": False,
            "timestamp": datetime.now(timezone.utc),
        }

        return quote, "Massive quote received successfully."

    except requests.RequestException:
        return None, "Could not connect to Massive."

    except Exception:
        return None, "Massive returned an unexpected response."


@st.cache_data(ttl=60, show_spinner=False)
def massive_daily_history(
    symbol: str,
    years: int,
    api_key: str,
):
    """
    Gets daily historical data from Massive.

    Returns:
        dataframe, status_message
    """
    if not api_key:
        return None, "Massive API key is empty."

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
            return None, f"Massive history HTTP {response.status_code}"

        payload = response.json()
        results = payload.get("results") or []

        if not results:
            return None, "Massive returned no historical data."

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
            return None, "Massive returned an empty dataset."

        dataframe = dataframe.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        dataframe = dataframe.set_index("date")
        dataframe = dataframe.sort_index()

        if dataframe.empty:
            return None, "No usable historical rows were found."

        return dataframe, "Massive history received successfully."

    except requests.RequestException:
        return None, "Could not connect to Massive history endpoint."

    except Exception:
        return None, "Unexpected Massive history response."


# ============================================================
# GOLDAPI
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def goldapi_latest_quote(symbol: str, api_key: str):
    """
    Gets gold or silver spot prices from GoldAPI.

    Gold:
        XAU/USD

    Silver:
        XAG/USD
    """
    if not api_key:
        return None, "GoldAPI key is empty."

    code = metal_code(symbol)
    url = f"https://app.goldapi.net/price/{code}/USD"

    try:
        response = requests.get(
            url,
            params={"x-api-key": api_key},
            timeout=20,
        )

        if response.status_code != 200:
            return None, f"GoldAPI HTTP {response.status_code}"

        payload = response.json()

        price = safe_float(payload.get("price"))

        previous_close = safe_float(
            payload.get("prev_close")
            or payload.get("previous_close")
            or payload.get("prev_close_price")
        )

        if not np.isfinite(price):
            return None, "GoldAPI returned no usable price."

        change = safe_float(
            payload.get("ch")
            or payload.get("change")
        )

        change_percent = safe_float(
            payload.get("chp")
            or payload.get("change_percent")
        )

        if not np.isfinite(change) and np.isfinite(previous_close):
            change = price - previous_close

        if (
            not np.isfinite(change_percent)
            and np.isfinite(previous_close)
            and previous_close != 0
        ):
            change_percent = change / previous_close * 100

        quote = {
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "source": f"GoldAPI {code}/USD spot",
            "delayed": False,
            "timestamp": datetime.now(timezone.utc),
        }

        return quote, "GoldAPI quote received successfully."

    except requests.RequestException:
        return None, "Could not connect to GoldAPI."

    except Exception:
        return None, "GoldAPI returned an unexpected response."


# ============================================================
# YAHOO FINANCE FALLBACK
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def yfinance_latest_quote(symbol: str):
    """
    Gets a fallback quote from Yahoo Finance.
    Data may be delayed.
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
            return None, "Yahoo Finance returned no quote."

        data = normalize_yfinance_columns(data)

        if "close" not in data.columns:
            return None, "Yahoo Finance returned no close price."

        data = data.dropna(subset=["close"])

        if data.empty:
            return None, "Yahoo Finance returned no usable price."

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

        quote = {
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "source": "Yahoo Finance fallback",
            "delayed": True,
            "timestamp": datetime.now(timezone.utc),
        }

        return quote, "Yahoo Finance fallback is being used."

    except Exception:
        return None, "Yahoo Finance request failed."


@st.cache_data(ttl=60, show_spinner=False)
def yfinance_history(symbol: str, years: int = 3):
    """
    Gets fallback historical data from Yahoo Finance.
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
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )

        data = data.sort_index()

        if data.empty:
            return None

        return data

    except Exception:
        return None


# ============================================================
# DATA LOADING
# ============================================================

def load_quote(symbol: str):
    symbol = clean_symbol(symbol)

    provider_messages = []

    # GoldAPI for gold and silver spot
    if is_precious_metal_spot(symbol):
        quote, message = goldapi_latest_quote(
            symbol,
            GOLDAPI_KEY,
        )

        if quote is not None:
            return quote, message

        provider_messages.append(message)

    # Massive for stocks and ETFs
    quote, message = massive_latest_quote(
        symbol,
        MASSIVE_API_KEY,
    )

    if quote is not None:
        return quote, message

    provider_messages.append(message)

    # Yahoo Finance fallback
    quote, yahoo_message = yfinance_latest_quote(symbol)

    if quote is not None:
        combined_message = (
            "Fallback data is being used. "
            + " | ".join(provider_messages)
        )
        return quote, combined_message

    provider_messages.append(yahoo_message)

    return None, " | ".join(provider_messages)


def load_history(symbol: str):
    symbol = clean_symbol(symbol)

    # Use Massive history for stocks and ETFs
    if not is_precious_metal_spot(symbol):
        dataframe, message = massive_daily_history(
            symbol,
            years=3,
            api_key=MASSIVE_API_KEY,
        )

        if dataframe is not None and not dataframe.empty:
            return dataframe, message

    # Use Yahoo Finance fallback history
    dataframe = yfinance_history(
        symbol,
        years=3,
    )

    if dataframe is not None and not dataframe.empty:
        return dataframe, "Yahoo Finance history is being used."

    return None, "No historical data was returned."


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

    # Recent structure
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

    # Buy condition
    df["buy_condition"] = (
        (df["trend_score"] >= 2)
        & (df["macd"] > df["macd_signal"])
        & (df["rsi"] > 50)
        & (df["rsi"] < 70)
        & (df["volume_ratio"] >= 1.5)
    )

    # Sell condition
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


@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series_id: str, years: int = 3):
    """
    Gets a full historical series from FRED as a dataframe
    with a single "value" column, indexed by date.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    try:
        response = requests.get(
            url,
            params={"id": series_id},
            timeout=20,
        )

        response.raise_for_status()

        data = pd.read_csv(StringIO(response.text))

        if data.empty:
            return None

        date_column = data.columns[0]
        value_column = data.columns[-1]

        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce",
        )

        data[value_column] = pd.to_numeric(
            data[value_column],
            errors="coerce",
        )

        data = data.dropna(subset=[date_column, value_column])

        if data.empty:
            return None

        cutoff = datetime.now() - timedelta(days=365 * years)
        data = data[data[date_column] >= cutoff]

        if data.empty:
            return None

        data = data.set_index(date_column)
        data = data.rename(columns={value_column: "value"})
        data = data.sort_index()

        return data

    except Exception:
        return None


# ============================================================
# GOLD & MACRO TERMINAL DATA
# ============================================================
# DXY is fetched via the existing yfinance_latest_quote / yfinance_history
# helpers above (symbol "DX-Y.NYB"), since those already handle any
# unmapped symbol generically.

CFTC_COT_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"


@st.cache_data(ttl=21600, show_spinner=False)
def load_cot_gold(weeks: int = 52):
    """
    Gets CFTC Disaggregated Commitments of Traders data for COMEX Gold
    futures (Managed Money positioning) from CFTC's public Socrata API.
    Published weekly, positions as of the prior Tuesday.
    """
    params = {
        "$where": "market_and_exchange_names='GOLD - COMMODITY EXCHANGE INC.'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": weeks,
    }

    try:
        response = requests.get(
            CFTC_COT_URL,
            params=params,
            timeout=20,
        )

        if response.status_code != 200:
            return None, f"CFTC COT HTTP {response.status_code}"

        payload = response.json()

        if not isinstance(payload, list) or not payload:
            return None, "CFTC COT returned no rows."

        rows = []

        for item in payload:
            rows.append(
                {
                    "date": pd.to_datetime(
                        item.get("report_date_as_yyyy_mm_dd"),
                        errors="coerce",
                    ),
                    "open_interest": safe_float(
                        item.get("open_interest_all")
                    ),
                    "mm_long": safe_float(
                        item.get("m_money_positions_long_all")
                    ),
                    "mm_short": safe_float(
                        item.get("m_money_positions_short_all")
                    ),
                }
            )

        cot_df = pd.DataFrame(rows).dropna(subset=["date"])
        cot_df = cot_df.sort_values("date")

        if cot_df.empty:
            return None, "CFTC COT rows had no usable dates."

        cot_df["mm_net"] = cot_df["mm_long"] - cot_df["mm_short"]

        return cot_df, "CFTC COT data received successfully."

    except requests.RequestException:
        return None, "Could not connect to the CFTC public reporting API."

    except ValueError:
        return None, "CFTC COT returned a non-JSON response."

    except Exception:
        return None, "Unexpected CFTC COT response."


@st.cache_data(ttl=900, show_spinner=False)
def load_geopolitical_headlines(query: str, max_records: int = 12):
    """
    Gets recent news headlines from GDELT's free Doc 2.0 API,
    filtered by the given search query.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "sort": "datedesc",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20,
        )

        if response.status_code != 200:
            return None, f"GDELT HTTP {response.status_code}"

        payload = response.json()
        articles = payload.get("articles") or []

        if not articles:
            return None, "GDELT returned no articles for this query."

        headlines = []

        for article in articles[:max_records]:
            headlines.append(
                {
                    "title": article.get("title", "Untitled"),
                    "url": article.get("url", ""),
                    "source": article.get("domain", "Unknown source"),
                    "date": article.get("seendate", ""),
                }
            )

        return headlines, "GDELT headlines received successfully."

    except requests.RequestException:
        return None, "Could not connect to GDELT."

    except ValueError:
        return None, "GDELT returned a non-JSON response."

    except Exception:
        return None, "Unexpected GDELT response."


def compute_etf_flow_proxy(symbol: str):
    """
    Approximates gold ETF flow pressure using price and dollar volume
    relative to the 20-day average. This is a proxy only: official
    creation/redemption unit flows are published by the ETF issuer
    (e.g., SPDR for GLD) and are not available through a free API.
    """
    quote, quote_status = load_quote(symbol)
    history, history_status = load_history(symbol)

    if quote is None or history is None or history.empty:
        return None, f"{symbol}: {quote_status or history_status}"

    working = history.copy()
    working["dollar_volume"] = working["close"] * working["volume"]
    working["dollar_volume_avg20"] = (
        working["dollar_volume"].rolling(20).mean()
    )

    latest = working.iloc[-1]

    dollar_volume_avg = safe_float(latest.get("dollar_volume_avg20"))
    dollar_volume = safe_float(latest.get("dollar_volume"))

    if np.isfinite(dollar_volume_avg) and dollar_volume_avg != 0:
        volume_ratio = dollar_volume / dollar_volume_avg
    else:
        volume_ratio = np.nan

    price_change = safe_float(quote.get("change_percent"))

    if np.isfinite(volume_ratio) and volume_ratio >= 1.3:
        if np.isfinite(price_change) and price_change > 0:
            signal = "Accumulation-like (heavy volume, price up)"
        elif np.isfinite(price_change) and price_change < 0:
            signal = "Distribution-like (heavy volume, price down)"
        else:
            signal = "Elevated activity, flat price"
    else:
        signal = "Normal activity"

    return (
        {
            "symbol": symbol,
            "price": quote.get("price"),
            "change_percent": price_change,
            "volume_ratio": volume_ratio,
            "signal": signal,
        },
        "OK",
    )


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


def build_macro_driver_chart(gold_df, dxy_df, real_yield_df, breakeven_df):
    """
    Indexes Gold, DXY, the 10Y real yield, and 5Y breakeven inflation
    to 100 over a shared date range so their trends can be compared
    on one chart regardless of their native units/scale.
    """
    def clean_index(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        index = pd.to_datetime(result.index)

        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)

        result.index = index.normalize()
        return result

    gold_df = clean_index(gold_df)
    dxy_df = clean_index(dxy_df)
    real_yield_df = clean_index(real_yield_df)
    breakeven_df = clean_index(breakeven_df)

    combined = pd.concat(
        {
            "gold": gold_df["close"],
            "dxy": dxy_df["close"],
            "real_yield": real_yield_df["value"],
            "breakeven": breakeven_df["value"],
        },
        axis=1,
        join="outer",
    )

    combined = combined.sort_index().ffill().dropna()

    if combined.empty:
        return None

    indexed = combined / combined.iloc[0] * 100

    chart = go.Figure()

    chart.add_trace(go.Scatter(
        x=indexed.index,
        y=indexed["gold"],
        name="Gold (indexed)",
        line={"color": "gold", "width": 2.5},
    ))

    chart.add_trace(go.Scatter(
        x=indexed.index,
        y=indexed["dxy"],
        name="DXY (indexed)",
        line={"color": "deepskyblue", "width": 2},
    ))

    chart.add_trace(go.Scatter(
        x=indexed.index,
        y=indexed["real_yield"],
        name="10Y Real Yield (indexed)",
        line={"color": "orangered", "width": 2},
    ))

    chart.add_trace(go.Scatter(
        x=indexed.index,
        y=indexed["breakeven"],
        name="5Y Breakeven Inflation (indexed)",
        line={"color": "mediumorchid", "width": 2},
    ))

    chart.update_layout(
        title="Gold Drivers: DXY, Real Yields & Breakeven Inflation (Indexed to 100)",
        height=480,
        template="plotly_dark",
        legend_orientation="h",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        yaxis_title="Indexed level (start = 100)",
    )

    return chart


def build_yield_curve_chart(dgs2_df, dgs10_df, spread_df):
    chart = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.65, 0.35],
        subplot_titles=["2Y vs 10Y Treasury Yield", "2Y-10Y Spread"],
    )

    chart.add_trace(
        go.Scatter(
            x=dgs2_df.index,
            y=dgs2_df["value"],
            name="2Y Yield",
            line={"color": "cyan", "width": 2},
        ),
        row=1,
        col=1,
    )

    chart.add_trace(
        go.Scatter(
            x=dgs10_df.index,
            y=dgs10_df["value"],
            name="10Y Yield",
            line={"color": "gold", "width": 2},
        ),
        row=1,
        col=1,
    )

    spread_colors = np.where(
        spread_df["value"] < 0,
        "rgba(220,50,50,0.75)",
        "rgba(0,180,80,0.75)",
    )

    chart.add_trace(
        go.Bar(
            x=spread_df.index,
            y=spread_df["value"],
            name="2Y-10Y Spread",
            marker_color=spread_colors,
        ),
        row=2,
        col=1,
    )

    chart.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
        row=2,
        col=1,
    )

    chart.update_layout(
        height=520,
        template="plotly_dark",
        legend_orientation="h",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )

    chart.update_yaxes(title_text="Yield (%)", row=1, col=1)
    chart.update_yaxes(title_text="Spread (pp)", row=2, col=1)

    return chart


def build_cot_chart(cot_df: pd.DataFrame):
    chart = make_subplots(specs=[[{"secondary_y": True}]])

    net_colors = np.where(
        cot_df["mm_net"] < 0,
        "rgba(220,50,50,0.75)",
        "rgba(0,180,80,0.75)",
    )

    chart.add_trace(
        go.Bar(
            x=cot_df["date"],
            y=cot_df["mm_net"],
            name="Managed Money Net Position",
            marker_color=net_colors,
        ),
        secondary_y=False,
    )

    chart.add_trace(
        go.Scatter(
            x=cot_df["date"],
            y=cot_df["open_interest"],
            name="Open Interest",
            line={"color": "white", "width": 1.5},
        ),
        secondary_y=True,
    )

    chart.update_layout(
        title="CFTC Gold Futures: Managed Money Net Positioning vs Open Interest",
        height=420,
        template="plotly_dark",
        legend_orientation="h",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )

    chart.update_yaxes(
        title_text="Net contracts (Managed Money)",
        secondary_y=False,
    )
    chart.update_yaxes(
        title_text="Open Interest",
        secondary_y=True,
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

st.sidebar.caption("API connection status")

st.sidebar.write(
    f"Massive key loaded: "
    f"{'YES' if MASSIVE_API_KEY else 'NO'}"
)

st.sidebar.write(
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

st.sidebar.markdown("---")
st.sidebar.caption("Gold & Macro Terminal settings")

cot_weeks = st.sidebar.slider(
    "CFTC COT history (weeks)",
    min_value=12,
    max_value=104,
    value=52,
    step=4,
)

headline_query = st.sidebar.text_input(
    "Geopolitical headline search",
    value='gold OR "central bank" OR sanctions OR geopolitical OR tariffs',
)

show_headlines = st.sidebar.checkbox(
    "Show geopolitical headlines",
    value=True,
)


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("Institutional-Style Market Dashboard")

st.caption(
    "Technical, volume, structure, risk, and macro analysis. "
    "This is decision-support software, not guaranteed financial advice."
)



# ============================================================
# GOLD & MACRO TERMINAL
# ============================================================

st.divider()
st.header("🥇 Gold & US Market Macro Terminal")

st.caption(
    "Every value below is fetched live from its official public source "
    "(FRED, Yahoo Finance, CFTC, GDELT) with the series ID and report "
    "date shown wherever possible. Where no free, real-time source "
    "exists (official ETF creation/redemption flows, central bank "
    "tonnage, the FOMC dot plot), the panel says so explicitly instead "
    "of estimating a number."
)

gold_quote, gold_quote_status = load_quote("XAUUSD")
gold_history, gold_history_status = load_history("XAUUSD")

dxy_quote, dxy_quote_status = yfinance_latest_quote("DX-Y.NYB")
dxy_history = yfinance_history("DX-Y.NYB", years=3)

real_yield_latest, real_yield_date = fred_latest("DFII10")
breakeven_latest, breakeven_date = fred_latest("T5YIE")
spread_latest, spread_date = fred_latest("T10Y2Y")
fed_upper, fed_upper_date = fred_latest("DFEDTARU")
fed_lower, fed_lower_date = fred_latest("DFEDTARL")
fed_effective, fed_effective_date = fred_latest("DFF")

# ---- Top metric row ----

term_m1, term_m2, term_m3, term_m4, term_m5 = st.columns(5)

term_m1.metric(
    "Gold Spot (XAU/USD)",
    format_number(gold_quote["price"], 2) if gold_quote else "Unavailable",
    format_delta(gold_quote["change_percent"]) if gold_quote else None,
)

term_m2.metric(
    "DXY (US Dollar Index)",
    format_number(dxy_quote["price"], 2) if dxy_quote else "Unavailable",
    format_delta(dxy_quote["change_percent"]) if dxy_quote else None,
)

term_m3.metric(
    "10Y TIPS Real Yield",
    f"{real_yield_latest:.2f}%"
    if np.isfinite(real_yield_latest)
    else "Unavailable",
    help=f"FRED DFII10, as of {real_yield_date}",
)

term_m4.metric(
    "5Y Breakeven Inflation",
    f"{breakeven_latest:.2f}%"
    if np.isfinite(breakeven_latest)
    else "Unavailable",
    help=f"FRED T5YIE, as of {breakeven_date}",
)

term_m5.metric(
    "2Y-10Y Spread",
    f"{spread_latest:+.2f} pp" if np.isfinite(spread_latest) else "Unavailable",
    help=f"FRED T10Y2Y, as of {spread_date}",
)

if not dxy_quote:
    st.caption(f"DXY: {dxy_quote_status}")

# ---- Gold drivers combined chart ----

st.subheader("DXY + Real Yields + Breakeven Inflation vs Gold")

real_yield_series = fred_series("DFII10", years=3)
breakeven_series = fred_series("T5YIE", years=3)

if (
    gold_history is not None and not gold_history.empty
    and dxy_history is not None and not dxy_history.empty
    and real_yield_series is not None
    and breakeven_series is not None
):
    driver_chart = build_macro_driver_chart(
        gold_history,
        dxy_history,
        real_yield_series,
        breakeven_series,
    )

    if driver_chart is not None:
        st.plotly_chart(driver_chart, use_container_width=True)
        st.caption(
            "Rising DXY and rising real yields are historically headwinds "
            "for gold; rising breakeven inflation combined with falling "
            "real yields has historically been a tailwind. Each series is "
            "indexed to 100 at the start of the visible window so trend "
            "direction, not scale, drives the comparison."
        )
    else:
        st.info(
            "Not enough overlapping data to build the combined driver "
            "chart yet."
        )
else:
    st.info(
        "One or more inputs (Gold, DXY, real yield, breakeven) are "
        "unavailable right now."
    )

# ---- Yield curve ----

st.subheader("Yield Curve (2Y-10Y)")

dgs2_series = fred_series("DGS2", years=3)
dgs10_series = fred_series("DGS10", years=3)
spread_series = fred_series("T10Y2Y", years=3)

if (
    dgs2_series is not None
    and dgs10_series is not None
    and spread_series is not None
):
    st.plotly_chart(
        build_yield_curve_chart(dgs2_series, dgs10_series, spread_series),
        use_container_width=True,
    )
    st.caption(
        "A negative spread (curve inversion) has historically preceded "
        "US recessions and often supports gold as a hedge; a sharp "
        "steepening back above zero after an inversion has historically "
        "coincided with risk-off periods too. Source: FRED DGS2, DGS10, "
        "T10Y2Y."
    )
else:
    st.info("Yield curve data is unavailable right now.")

# ---- Fed policy ----

st.subheader("Fed Rate Decision / Dots")

fed_c1, fed_c2, fed_c3 = st.columns(3)

if np.isfinite(fed_upper) and np.isfinite(fed_lower):
    fed_c1.metric(
        "Fed Funds Target Range",
        f"{fed_lower:.2f}% - {fed_upper:.2f}%",
        help=f"FRED DFEDTARL / DFEDTARU, as of {fed_upper_date}",
    )
else:
    fed_c1.metric("Fed Funds Target Range", "Unavailable")

fed_c2.metric(
    "Effective Fed Funds Rate",
    f"{fed_effective:.2f}%" if np.isfinite(fed_effective) else "Unavailable",
    help=f"FRED DFF, as of {fed_effective_date}",
)

if np.isfinite(fed_effective) and np.isfinite(breakeven_latest):
    fed_c3.metric(
        "Real Policy Rate (EFFR - Breakeven)",
        f"{(fed_effective - breakeven_latest):+.2f}%",
    )
else:
    fed_c3.metric("Real Policy Rate (EFFR - Breakeven)", "Unavailable")

st.info(
    "The FOMC dot plot (Summary of Economic Projections) is published "
    "quarterly by the Fed and has no free real-time API. For the latest "
    "dot plot and the next meeting date, see "
    "[federalreserve.gov/monetarypolicy/fomccalendars.htm]"
    "(https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)."
)

# ---- Gold ETF flows (proxy) ----

st.subheader("Gold ETF Flows (Volume Proxy)")

st.caption(
    "Official creation/redemption unit flows are published by each ETF "
    "issuer (e.g., SPDR for GLD, iShares for IAU) and are not available "
    "through a free real-time API. The figures below approximate flow "
    "pressure using price and dollar volume relative to the 20-day "
    "average — a genuine proxy, not official flow data."
)

etf_columns = st.columns(2)

for etf_column, etf_symbol in zip(etf_columns, ["GLD", "IAU"]):
    flow, flow_status = compute_etf_flow_proxy(etf_symbol)

    with etf_column:
        if flow is None:
            st.warning(f"{etf_symbol}: {flow_status}")
            continue

        st.metric(
            f"{etf_symbol} Price",
            format_number(flow["price"], 2),
            format_delta(flow["change_percent"]),
        )

        if np.isfinite(flow["volume_ratio"]):
            st.write(
                f"Volume vs 20D avg: **{flow['volume_ratio']:.2f}x**"
            )
        else:
            st.write("Volume vs 20D avg: Unavailable")

        st.write(f"Signal: **{flow['signal']}**")

# ---- Central bank buying ----

st.subheader("Central Bank Buying")

st.info(
    "Central bank gold reserve changes are reported quarterly by the "
    "World Gold Council and monthly (with a lag) by the IMF's "
    "International Financial Statistics — neither publishes a free "
    "real-time API. Central banks have been net gold buyers on a "
    "sustained multi-year basis. For the latest tonnage by country, see "
    "[World Gold Council - Gold Demand Trends]"
    "(https://www.gold.org/goldhub/data/gold-demand-by-country) and "
    "[IMF Data - International Financial Statistics](https://data.imf.org)."
)

# ---- CFTC COT positioning ----

st.subheader("CFTC COT (Positioning) - Gold Futures")

cot_df, cot_status = load_cot_gold(weeks=cot_weeks)

if cot_df is not None and not cot_df.empty:
    latest_cot = cot_df.iloc[-1]

    cot_c1, cot_c2, cot_c3 = st.columns(3)

    cot_c1.metric(
        "Managed Money Net Position",
        f"{latest_cot['mm_net']:,.0f}",
        help=f"Report date: {latest_cot['date'].date()}",
    )
    cot_c2.metric("Managed Money Long", f"{latest_cot['mm_long']:,.0f}")
    cot_c3.metric("Managed Money Short", f"{latest_cot['mm_short']:,.0f}")

    st.plotly_chart(build_cot_chart(cot_df), use_container_width=True)
    st.caption(
        "Source: CFTC Disaggregated Commitments of Traders report "
        "(COMEX Gold futures), published weekly, positions as of the "
        "prior Tuesday. Extreme net-long positioning by Managed Money "
        "has historically preceded pullbacks; extreme net-short "
        "positioning has historically preceded rallies."
    )
else:
    st.warning(f"CFTC COT data is unavailable right now: {cot_status}")
    st.caption(
        "You can check the official report directly at "
        "[cftc.gov Commitments of Traders]"
        "(https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)."
    )

# ---- Geopolitical headlines ----

if show_headlines:
    st.subheader("Geopolitical Headlines")

    headlines, headlines_status = load_geopolitical_headlines(
        headline_query,
        max_records=12,
    )

    if headlines:
        for item in headlines:
            st.markdown(
                f"**[{item['title']}]({item['url']})**  \n"
                f"{item['source']} · {item['date']}"
            )
    else:
        st.warning(f"Headlines are unavailable right now: {headlines_status}")


st.caption(
    "Last refresh: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)


quote, quote_status = load_quote(symbol)
history, history_status = load_history(symbol)

if quote is None:
    st.error(
        "No quote data was returned."
    )

    st.info(quote_status)

    with st.expander("Connection details"):
        st.write(f"Selected symbol: `{symbol}`")
        st.write(
            f"Massive key loaded: "
            f"`{'YES' if MASSIVE_API_KEY else 'NO'}`"
        )
        st.write(
            f"GoldAPI key loaded: "
            f"`{'YES' if GOLDAPI_KEY else 'NO'}`"
        )
        st.write(f"Quote status: `{quote_status}`")
        st.write(f"History status: `{history_status}`")

    st.stop()

if history is None or history.empty:
    st.error(
        "No historical data was returned."
    )

    st.info(history_status)

    with st.expander("Connection details"):
        st.write(f"Selected symbol: `{symbol}`")
        st.write(f"Quote status: `{quote_status}`")
        st.write(f"History status: `{history_status}`")

    st.stop()

df = calculate_indicators(history)

if df.empty:
    st.error("The historical dataset is empty.")
    st.stop()

analysis = get_analysis(df)

if quote["delayed"]:
    st.warning(quote_status)
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
    format_number(quote["price"]),
    format_delta(quote["change_percent"]),
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
    format_number(analysis["rsi"], 1),
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
    format_number(analysis["atr"]),
)

panel_3.metric(
    "Buy Stop Guide",
    format_number(analysis["buy_stop"]),
)

panel_4.metric(
    "Buy Target Guide",
    format_number(analysis["buy_target"]),
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
# CALCULATIONS TABLE
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
