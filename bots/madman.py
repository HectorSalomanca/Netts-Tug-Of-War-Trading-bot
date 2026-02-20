import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID = os.getenv("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
PUMP_THRESHOLD = 0.05
VOLUME_SPIKE_MULTIPLIER = 2.0


def get_intraday_bars(symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=500,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"[MADMAN] Intraday bars error for {symbol}: {e}")
        return None


def get_daily_bars(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"[MADMAN] Daily bars error for {symbol}: {e}")
        return None


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def detect_pump(df_1min: pd.DataFrame) -> tuple[bool, float]:
    if df_1min is None or len(df_1min) < 10:
        return False, 0.0
    recent = df_1min.tail(10)
    price_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]
    return price_change >= PUMP_THRESHOLD, round(price_change, 4)


def detect_volume_spike(df_daily: pd.DataFrame) -> tuple[bool, float]:
    if df_daily is None or len(df_daily) < 10:
        return False, 1.0
    avg_volume = df_daily["volume"].iloc[:-1].mean()
    latest_volume = df_daily["volume"].iloc[-1]
    ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
    return ratio >= VOLUME_SPIKE_MULTIPLIER, round(ratio, 2)


def get_recent_retail_sentiment(symbol: str) -> tuple[str, float]:
    try:
        result = (
            supabase.table("signals")
            .select("direction, confidence")
            .eq("symbol", symbol)
            .eq("bot", "madman")
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        rows = result.data
        if not rows:
            return "neutral", 0.5

        buy_conf = [r["confidence"] for r in rows if r["direction"] == "buy"]
        sell_conf = [r["confidence"] for r in rows if r["direction"] == "sell"]

        avg_buy = np.mean(buy_conf) if buy_conf else 0
        avg_sell = np.mean(sell_conf) if sell_conf else 0

        if avg_buy > avg_sell and avg_buy > 0.55:
            return "buy", float(avg_buy)
        elif avg_sell > avg_buy and avg_sell > 0.55:
            return "sell", float(avg_sell)
        return "neutral", 0.5
    except Exception as e:
        print(f"[MADMAN] Sentiment fetch error: {e}")
        return "neutral", 0.5


def analyze(symbol: str) -> dict:
    print(f"[MADMAN] Analyzing {symbol}...")

    df_1min = get_intraday_bars(symbol, days=2)
    df_daily = get_daily_bars(symbol, days=30)

    if df_daily is None or len(df_daily) < 5:
        return {
            "symbol": symbol,
            "direction": "neutral",
            "confidence": 0.5,
            "signal_type": "insufficient_data",
            "raw_data": {},
        }

    closes_daily = df_daily["close"].values
    rsi = compute_rsi(closes_daily)

    is_pump, pump_pct = detect_pump(df_1min)
    is_vol_spike, vol_ratio = detect_volume_spike(df_daily)

    sentiment_dir, sentiment_conf = get_recent_retail_sentiment(symbol)

    fomo_score = 0
    fear_score = 0
    factor_details = {}

    if rsi >= RSI_OVERBOUGHT:
        fomo_score += 2
        factor_details["rsi_overbought"] = True
        factor_details["rsi"] = rsi
    elif rsi <= RSI_OVERSOLD:
        fear_score += 2
        factor_details["rsi_oversold"] = True
        factor_details["rsi"] = rsi
    else:
        factor_details["rsi"] = rsi

    if is_pump:
        fomo_score += 3
        factor_details["pump_detected"] = True
        factor_details["pump_pct"] = pump_pct
    else:
        factor_details["pump_detected"] = False
        factor_details["pump_pct"] = pump_pct

    if is_vol_spike:
        fomo_score += 1
        factor_details["volume_spike"] = True
        factor_details["volume_ratio"] = vol_ratio
    else:
        factor_details["volume_spike"] = False
        factor_details["volume_ratio"] = vol_ratio

    if sentiment_dir == "buy":
        fomo_score += 2
    elif sentiment_dir == "sell":
        fear_score += 2
    factor_details["retail_sentiment"] = sentiment_dir

    total = fomo_score + fear_score
    if total == 0:
        direction = "neutral"
        confidence = 0.5
    elif fomo_score > fear_score:
        direction = "buy"
        confidence = round(min(0.5 + (fomo_score / total) * 0.45, 0.95), 4)
    else:
        direction = "sell"
        confidence = round(min(0.5 + (fear_score / total) * 0.45, 0.95), 4)

    raw_data = {
        **factor_details,
        "current_price": float(closes_daily[-1]),
        "fomo_score": fomo_score,
        "fear_score": fear_score,
        "retail_sentiment_confidence": sentiment_conf,
    }

    print(f"[MADMAN] {symbol}: {direction.upper()} @ {confidence:.2%} (RSI: {rsi}, Pump: {is_pump})")

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "signal_type": "rsi_pump_fomo",
        "raw_data": raw_data,
    }


def log_signal(result: dict):
    if not USER_ID:
        print(f"[MADMAN] USER_ID not set — skipping Supabase write")
        return
    record = {
        "symbol": result["symbol"],
        "bot": "madman",
        "direction": result["direction"],
        "confidence": result["confidence"],
        "signal_type": result["signal_type"],
        "raw_data": result["raw_data"],
        "source_url": None,
        "user_id": USER_ID,
    }
    try:
        supabase.table("signals").insert(record).execute()
    except Exception as e:
        print(f"[MADMAN] Supabase error: {e}")


def run(symbols: list[str]) -> list[dict]:
    results = []
    for symbol in symbols:
        result = analyze(symbol)
        log_signal(result)
        results.append(result)
    return results


if __name__ == "__main__":
    test_symbols = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
    results = run(test_symbols)
    for r in results:
        print(r)
