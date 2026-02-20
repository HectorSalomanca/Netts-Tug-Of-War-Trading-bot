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


def get_price_history(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
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
        print(f"[SOVEREIGN] Price history error for {symbol}: {e}")
        return None


def compute_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b
    half_kelly = kelly * 0.5
    return max(0.0, min(half_kelly, 0.25))


def compute_signals(df: pd.DataFrame) -> dict:
    closes = df["close"].values
    if len(closes) < 20:
        return {"sma20": None, "sma50": None, "momentum": None, "volatility": None}

    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else None
    momentum_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns[-20:]) * np.sqrt(252)

    return {
        "sma20": sma20,
        "sma50": sma50,
        "momentum_5d": momentum_5d,
        "volatility": volatility,
        "current_price": closes[-1],
    }


def get_recent_sentiment(symbol: str) -> tuple[str, float]:
    try:
        result = (
            supabase.table("signals")
            .select("direction, confidence")
            .eq("symbol", symbol)
            .eq("bot", "sovereign")
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
            return "buy", avg_buy
        elif avg_sell > avg_buy and avg_sell > 0.55:
            return "sell", avg_sell
        return "neutral", 0.5
    except Exception as e:
        print(f"[SOVEREIGN] Sentiment fetch error: {e}")
        return "neutral", 0.5


def analyze(symbol: str) -> dict:
    print(f"[SOVEREIGN] Analyzing {symbol}...")
    df = get_price_history(symbol)

    if df is None or len(df) < 10:
        return {
            "symbol": symbol,
            "direction": "neutral",
            "confidence": 0.5,
            "signal_type": "insufficient_data",
            "raw_data": {},
        }

    signals = compute_signals(df)
    sentiment_dir, sentiment_conf = get_recent_sentiment(symbol)

    bullish_factors = 0
    bearish_factors = 0
    factor_details = {}

    current = signals.get("current_price", 0)
    sma20 = signals.get("sma20")
    sma50 = signals.get("sma50")
    momentum = signals.get("momentum_5d", 0)
    volatility = signals.get("volatility", 0.3)

    if sma20 and current > sma20:
        bullish_factors += 1
        factor_details["price_above_sma20"] = True
    elif sma20:
        bearish_factors += 1
        factor_details["price_above_sma20"] = False

    if sma50 and sma20 and sma20 > sma50:
        bullish_factors += 2
        factor_details["golden_cross"] = True
    elif sma50 and sma20:
        bearish_factors += 2
        factor_details["golden_cross"] = False

    if momentum > 0.02:
        bullish_factors += 1
        factor_details["momentum_positive"] = True
    elif momentum < -0.02:
        bearish_factors += 1
        factor_details["momentum_positive"] = False

    if sentiment_dir == "buy":
        bullish_factors += 2
    elif sentiment_dir == "sell":
        bearish_factors += 2
    factor_details["sentiment"] = sentiment_dir

    total_factors = bullish_factors + bearish_factors
    if total_factors == 0:
        direction = "neutral"
        base_confidence = 0.5
    elif bullish_factors > bearish_factors:
        direction = "buy"
        base_confidence = bullish_factors / total_factors
    else:
        direction = "sell"
        base_confidence = bearish_factors / total_factors

    vol_penalty = min(volatility / 2.0, 0.2)
    confidence = round(max(0.5, min(base_confidence - vol_penalty, 0.95)), 4)

    closes = df["close"].values
    returns = np.diff(closes) / closes[:-1]
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
    avg_win = np.mean(wins) if len(wins) > 0 else 0.01
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.01
    kelly_fraction = compute_kelly_criterion(win_rate, avg_win, avg_loss)

    raw_data = {
        **factor_details,
        "current_price": current,
        "sma20": sma20,
        "sma50": sma50,
        "momentum_5d": momentum,
        "volatility": volatility,
        "kelly_fraction": kelly_fraction,
        "win_rate": win_rate,
        "bullish_factors": bullish_factors,
        "bearish_factors": bearish_factors,
        "sentiment_direction": sentiment_dir,
        "sentiment_confidence": sentiment_conf,
    }

    print(f"[SOVEREIGN] {symbol}: {direction.upper()} @ {confidence:.2%} (Kelly: {kelly_fraction:.2%})")

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "signal_type": "kelly_multi_factor",
        "raw_data": raw_data,
    }


def log_signal(result: dict):
    if not USER_ID:
        print(f"[SOVEREIGN] USER_ID not set — skipping Supabase write")
        return
    record = {
        "symbol": result["symbol"],
        "bot": "sovereign",
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
        print(f"[SOVEREIGN] Supabase error: {e}")


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
