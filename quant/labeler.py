"""
Triple Barrier Labeler V3 — Institutional-Grade Trade Labeling

Labels historical trades based on which barrier is hit first:
  - Upper Barrier (+4%): Take Profit → label = 1 (win)
  - Lower Barrier (-2%): Stop Loss   → label = -1 (loss)
  - Vertical Barrier (3:45 PM ET):   → label = 0 (time stop)

Runs nightly on all trades in Supabase that lack a tbl_label.
Labels are used as training targets for Stockformer weekly retraining.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional

import pytz
from dateutil import parser as dateutil_parser
from dotenv import load_dotenv
from supabase import create_client, Client
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID           = os.getenv("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

ET = pytz.timezone("America/New_York")

# Barrier thresholds
UPPER_BARRIER_PCT = 0.04   # +4% take profit
LOWER_BARRIER_PCT = 0.02   # -2% stop loss
VERTICAL_HOUR     = 15     # 3:45 PM ET
VERTICAL_MINUTE   = 45


def get_intraday_bars(symbol: str, date: datetime) -> Optional[pd.DataFrame]:
    """Fetch 1-min bars for a specific trading day."""
    try:
        start = date.replace(hour=9, minute=30, second=0, microsecond=0, tzinfo=ET)
        end   = date.replace(hour=16, minute=0, second=0, microsecond=0, tzinfo=ET)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start.astimezone(timezone.utc),
            end=end.astimezone(timezone.utc),
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df.sort_index()
    except Exception as e:
        print(f"[LABELER] Bar fetch error for {symbol} on {date.date()}: {e}")
        return None


def label_trade(
    symbol: str,
    entry_price: float,
    entry_time: datetime,
    side: str,
) -> dict:
    """
    Apply Triple Barrier Labeling to a single trade.

    Returns:
      - tbl_label: 1 (win), -1 (loss), 0 (time stop)
      - barrier_hit: "upper", "lower", "vertical"
      - exit_price: price at barrier hit
      - exit_time: timestamp of barrier hit
      - max_favorable: maximum favorable excursion (%)
      - max_adverse: maximum adverse excursion (%)
    """
    trade_date = entry_time.astimezone(ET)
    df = get_intraday_bars(symbol, trade_date)

    if df is None or len(df) < 10:
        return {"tbl_label": 0, "barrier_hit": "no_data", "exit_price": entry_price}

    # Filter to bars after entry
    entry_utc = entry_time.astimezone(timezone.utc) if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
    df_after = df[df.index >= entry_utc]
    if len(df_after) == 0:
        return {"tbl_label": 0, "barrier_hit": "no_bars_after_entry", "exit_price": entry_price}

    # Vertical barrier: 3:45 PM ET
    vertical_time = trade_date.replace(hour=VERTICAL_HOUR, minute=VERTICAL_MINUTE, second=0)
    vertical_utc = vertical_time.astimezone(timezone.utc)

    max_favorable = 0.0
    max_adverse = 0.0

    for idx, row in df_after.iterrows():
        current_price = float(row["close"])
        bar_time = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx

        if side == "buy":
            pct_change = (current_price - entry_price) / entry_price
        else:
            pct_change = (entry_price - current_price) / entry_price

        max_favorable = max(max_favorable, pct_change)
        max_adverse = min(max_adverse, pct_change)

        # Upper barrier (take profit)
        if pct_change >= UPPER_BARRIER_PCT:
            return {
                "tbl_label": 1,
                "barrier_hit": "upper",
                "exit_price": round(current_price, 4),
                "exit_time": str(bar_time),
                "max_favorable": round(max_favorable, 6),
                "max_adverse": round(max_adverse, 6),
                "bars_to_exit": len(df_after.loc[:idx]),
            }

        # Lower barrier (stop loss)
        if pct_change <= -LOWER_BARRIER_PCT:
            return {
                "tbl_label": -1,
                "barrier_hit": "lower",
                "exit_price": round(current_price, 4),
                "exit_time": str(bar_time),
                "max_favorable": round(max_favorable, 6),
                "max_adverse": round(max_adverse, 6),
                "bars_to_exit": len(df_after.loc[:idx]),
            }

        # Vertical barrier (time stop)
        if hasattr(bar_time, 'astimezone'):
            bar_et = bar_time.astimezone(ET)
        else:
            bar_et = bar_time
        if hasattr(bar_et, 'hour') and (bar_et.hour > VERTICAL_HOUR or
            (bar_et.hour == VERTICAL_HOUR and bar_et.minute >= VERTICAL_MINUTE)):
            return {
                "tbl_label": 0,
                "barrier_hit": "vertical",
                "exit_price": round(current_price, 4),
                "exit_time": str(bar_time),
                "max_favorable": round(max_favorable, 6),
                "max_adverse": round(max_adverse, 6),
                "bars_to_exit": len(df_after.loc[:idx]),
            }

    # End of data without hitting any barrier
    last_price = float(df_after["close"].iloc[-1])
    return {
        "tbl_label": 0,
        "barrier_hit": "eod",
        "exit_price": round(last_price, 4),
        "max_favorable": round(max_favorable, 6),
        "max_adverse": round(max_adverse, 6),
    }


def run_nightly_labeling():
    """
    Fetch all unlabeled trades from Supabase and apply Triple Barrier Labeling.
    Updates each trade with tbl_label.
    """
    print(f"[LABELER] Starting nightly labeling at {datetime.now(timezone.utc).isoformat()}")

    try:
        result = (
            supabase.table("trades")
            .select("id, symbol, side, limit_price, created_at, status")
            .is_("tbl_label", "null")
            .eq("status", "filled")
            .order("created_at", desc=False)
            .limit(100)
            .execute()
        )
    except Exception as e:
        print(f"[LABELER] Fetch error: {e}")
        return

    trades = result.data if result.data else []
    if not trades:
        print("[LABELER] No unlabeled trades found")
        return

    print(f"[LABELER] Found {len(trades)} unlabeled trades")

    labeled = 0
    for trade in trades:
        entry_time = dateutil_parser.parse(trade["created_at"])
        raw_price = trade.get("limit_price") or trade.get("fill_price")
        if raw_price is None:
            continue
        entry_price = float(raw_price)
        if entry_price <= 0:
            continue

        label_result = label_trade(
            symbol=trade["symbol"],
            entry_price=entry_price,
            entry_time=entry_time,
            side=trade["side"],
        )

        label_str = {1: "WIN", -1: "LOSS", 0: "TIME_STOP"}.get(label_result["tbl_label"], "UNKNOWN")
        print(f"[LABELER] {trade['symbol']}: {label_str} via {label_result['barrier_hit']} "
              f"(MFE={label_result.get('max_favorable', 0):.2%}, MAE={label_result.get('max_adverse', 0):.2%})")

        try:
            supabase.table("trades").update({
                "tbl_label": label_result["tbl_label"],
            }).eq("id", trade["id"]).execute()
            labeled += 1
        except Exception as e:
            print(f"[LABELER] Update error for {trade['id']}: {e}")

    print(f"[LABELER] Labeled {labeled}/{len(trades)} trades")


if __name__ == "__main__":
    run_nightly_labeling()
