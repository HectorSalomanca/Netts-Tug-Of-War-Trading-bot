"""
Position Manager — handles all exit logic for open trades.

Exit rules (in priority order):
1. STOP LOSS     : position down > 2% → exit immediately (hard risk limit)
2. TAKE PROFIT   : position up  > 4% → exit (lock in 2:1 reward/risk)
3. SIGNAL FLIP   : Sovereign flips direction on an open position → exit
4. CROWDED EXIT  : Madman joins our direction (trade became crowded) → exit
5. EOD CLOSE     : 15 min before market close → close all day trades

This is what separates a real quant bot from a signal generator.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

import pytz
from referee.secret_vault import get_secret
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import ClosePositionRequest, GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from supabase import create_client, Client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.ticker_profiles import get_behavior
from referee.net_guard import with_retry

ALPACA_API_KEY   = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
SUPABASE_URL     = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID          = get_secret("USER_ID")
PAPER_TRADE      = (get_secret("ALPACA_PAPER_TRADE") or "True").lower() == "true"

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADE)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

ET = pytz.timezone("America/New_York")

# Default stops (trend regime)
STOP_LOSS_PCT   = 0.019
TAKE_PROFIT_PCT = 0.038
EOD_CLOSE_MINS  = 15

# Off-round stops: trigger slightly before institutional round-number clusters
# (e.g. 1.9% fires before the crowd's 2% stops get swept)
REGIME_STOPS = {
    "trend":       {"stop": 0.019, "take": 0.038},
    "trend_bull":  {"stop": 0.024, "take": 0.048},
    "trend_bear":  {"stop": 0.014, "take": 0.029},
    "chop":        {"stop": 0.009, "take": 0.018},
    "crisis":      {"stop": 0.00,  "take": 0.00},
}

# P4: Trailing stop — tracks high-water mark per position
# Only for momentum_breakout tickers in trend/trend_bull
TRAILING_STOP_PCT = 0.014  # trail 1.4% below high-water (off-round)
_high_water_marks = defaultdict(float)  # {symbol: max unrealized_pct seen}


@with_retry(max_attempts=3, base_delay=5, default=[], label="get_open_positions")
def get_open_positions() -> list:
    return trading_client.get_all_positions()


def is_near_close() -> bool:
    now_et = datetime.now(ET)
    close_time = now_et.replace(hour=15, minute=45, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return close_time <= now_et <= market_close


def _normalize_order_status(status) -> str:
    s = str(status).lower()
    if "partial" in s:
        return "partial"
    if "fill" in s:
        return "filled"
    if "reject" in s:
        return "rejected"
    if "expire" in s:
        return "expired"
    if "cancel" in s:
        return "cancelled"
    if "accept" in s or "new" in s or "pending" in s:
        return "pending"
    return s


def _cancel_open_orders(symbol: str) -> int:
    cancelled = 0
    try:
        orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        for order in orders:
            trading_client.cancel_order_by_id(order.id)
            cancelled += 1
    except Exception as e:
        print(f"[POS_MGR] Open-order cancel error for {symbol}: {e}")
    return cancelled


def _wait_for_order_release(symbol: str, wait_seconds: int = 6) -> bool:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            orders = trading_client.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            )
            if not orders:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _wait_for_order_fill(order_id: Optional[str], wait_seconds: int = 8) -> dict:
    if not order_id:
        return {"status": "pending", "filled_qty": 0.0, "fill_price": None}
    deadline = time.time() + wait_seconds
    last_status = "pending"
    while time.time() < deadline:
        try:
            order = trading_client.get_order_by_id(order_id)
            status = _normalize_order_status(order.status)
            last_status = status
            filled_qty = float(order.filled_qty or 0)
            fill_price = float(order.filled_avg_price or 0) or None
            if status in ("filled", "cancelled", "expired", "rejected") or filled_qty > 0:
                return {
                    "status": status,
                    "filled_qty": filled_qty,
                    "fill_price": fill_price,
                }
        except Exception:
            pass
        time.sleep(1)
    return {"status": last_status, "filled_qty": 0.0, "fill_price": None}


def close_position(symbol: str, reason: str, qty: Optional[float] = None):
    try:
        position = trading_client.get_open_position(symbol)
    except Exception as e:
        print(f"[POS_MGR] {symbol}: no open position to close ({e})")
        return None

    signed_qty = float(position.qty)
    available_qty = abs(signed_qty)
    if available_qty <= 0:
        print(f"[POS_MGR] {symbol}: zero available quantity, skipping close")
        return None

    close_qty = int(min(abs(qty), available_qty)) if qty is not None else int(available_qty)
    close_qty = max(close_qty, 1)
    entry_side = "buy" if signed_qty > 0 else "sell"
    exit_side = "sell" if signed_qty > 0 else "buy"
    avg_entry_price = float(position.avg_entry_price or 0)

    cancelled = _cancel_open_orders(symbol)
    if cancelled:
        print(f"[POS_MGR] {symbol}: cancelled {cancelled} open order(s) before close")
        if not _wait_for_order_release(symbol):
            print(f"[POS_MGR] {symbol}: open orders still releasing — attempting close anyway")

    @with_retry(max_attempts=3, base_delay=5, default=None, label=f"close_position({symbol})")
    def _do_close():
        if close_qty < int(available_qty):
            req = ClosePositionRequest(qty=str(close_qty))
            return trading_client.close_position(symbol, close_options=req)
        return trading_client.close_position(symbol)

    order = _do_close()
    if order is None:
        print(f"[POS_MGR] CLOSE FAILED {symbol} — reason: {reason}")
        return None

    order_id = str(getattr(order, "id", "")) or None
    fill = _wait_for_order_fill(order_id)
    fill_qty = float(fill.get("filled_qty") or 0)
    fill_price = fill.get("fill_price")
    status = fill.get("status", "pending")

    realized_pnl = None
    if fill_price is not None and avg_entry_price > 0 and fill_qty > 0:
        if signed_qty > 0:
            realized_pnl = (fill_price - avg_entry_price) * fill_qty
        else:
            realized_pnl = (avg_entry_price - fill_price) * fill_qty
        realized_pnl = round(realized_pnl, 2)

    _log_close_trade(
        symbol=symbol,
        side=exit_side,
        qty=fill_qty or close_qty,
        alpaca_order_id=order_id,
        status=status,
        fill_price=fill_price,
        pnl=realized_pnl,
    )
    if status == "filled":
        _update_trade_record(symbol, entry_side, realized_pnl)

    if status == "filled":
        pnl_tag = f" | pnl={realized_pnl:+.2f}" if realized_pnl is not None else ""
        print(f"[POS_MGR] CLOSED {symbol} — reason: {reason}{pnl_tag}")
    else:
        print(f"[POS_MGR] CLOSE SUBMITTED {symbol} — reason: {reason} | status={status}")

    return {
        "action": "exit",
        "status": status,
        "alpaca_order_id": order_id,
        "qty": int(fill_qty or close_qty),
        "side": exit_side,
        "fill_price": fill_price,
        "pnl": realized_pnl,
    }


def _log_close_trade(
    symbol: str,
    side: str,
    qty: float,
    alpaca_order_id: Optional[str],
    status: str,
    fill_price: Optional[float],
    pnl: Optional[float],
):
    if not USER_ID:
        return
    db_status = "filled" if status == "filled" else "pending"
    try:
        supabase.table("trades").insert({
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "fill_qty": int(qty),
            "order_type": "market",
            "order_type_detail": "position_close",
            "alpaca_order_id": alpaca_order_id,
            "status": db_status,
            "fill_price": fill_price,
            "pnl": pnl,
            "user_id": USER_ID,
        }).execute()
    except Exception as e:
        print(f"[POS_MGR] Close trade insert error: {e}")


def _update_trade_record(symbol: str, entry_side: str, realized_pnl: Optional[float]):
    if not USER_ID:
        return
    try:
        if realized_pnl is None:
            return
        latest = (
            supabase.table("trades")
            .select("id")
            .eq("symbol", symbol)
            .eq("user_id", USER_ID)
            .eq("side", entry_side)
            .eq("status", "filled")
            .is_("pnl", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if latest.data:
            supabase.table("trades").update({
                "pnl": realized_pnl,
            }).eq("id", latest.data[0]["id"]).execute()
    except Exception as e:
        print(f"[POS_MGR] Supabase update error: {e}")


def check_stop_take(position, regime: str = "trend") -> Optional[str]:
    try:
        stops = REGIME_STOPS.get(regime, REGIME_STOPS["trend"])
        symbol = position.symbol
        unrealized_pct = float(position.unrealized_plpc)

        # Fixed stop loss (off-round)
        if unrealized_pct <= -stops["stop"]:
            _high_water_marks.pop(symbol, None)
            return f"STOP_LOSS ({unrealized_pct:.2%}) [{regime}]"

        # Fixed take profit (off-round)
        if unrealized_pct >= stops["take"]:
            _high_water_marks.pop(symbol, None)
            return f"TAKE_PROFIT ({unrealized_pct:.2%}) [{regime}]"

        # P4: Trailing stop for momentum tickers in bullish regimes
        behavior = get_behavior(symbol)
        if behavior == "momentum_breakout" and regime in ("trend", "trend_bull"):
            # Update high-water mark
            if unrealized_pct > _high_water_marks[symbol]:
                _high_water_marks[symbol] = unrealized_pct

            hwm = _high_water_marks[symbol]
            # Only activate trailing stop once position is >1% profitable
            if hwm > 0.01 and unrealized_pct < (hwm - TRAILING_STOP_PCT):
                _high_water_marks.pop(symbol, None)
                return f"TRAILING_STOP ({unrealized_pct:.2%}, HWM={hwm:.2%}) [{regime}]"

    except Exception:
        pass
    return None


def check_signal_exit(position, sovereign_results: dict, madman_results: dict) -> Optional[str]:
    symbol = position.symbol
    side   = "buy" if float(position.qty) > 0 else "sell"

    s = sovereign_results.get(symbol)
    m = madman_results.get(symbol)
    if not s or not m:
        return None

    s_dir = s["direction"]
    m_dir = m["direction"]

    # Sovereign flipped against our position
    if side == "buy" and s_dir == "sell":
        return f"SIGNAL_FLIP (Sovereign now SELL)"
    if side == "sell" and s_dir == "buy":
        return f"SIGNAL_FLIP (Sovereign now BUY)"

    # Trade became crowded — Madman joined our direction (edge gone)
    if side == "buy" and m_dir == "buy":
        return f"CROWDED_EXIT (Madman now BUY — retail piled in)"
    if side == "sell" and m_dir == "sell":
        return f"CROWDED_EXIT (Madman now SELL — retail piled in)"

    return None


def run_exit_checks(
    sovereign_results: dict,
    madman_results: dict,
    regime: str = "trend",
    force_close: bool = False,
):
    positions = get_open_positions()

    if not positions:
        print("[POS_MGR] No open positions to check")
        return

    print(f"[POS_MGR] Checking {len(positions)} open position(s) | regime={regime}...")

    # Force close (crisis regime) — skip SQQQ hedge and any shorts (they're intentional)
    if force_close:
        print("[POS_MGR] Force close — crisis regime, closing long positions")
        for pos in positions:
            qty = float(pos.qty)
            if pos.symbol == "SQQQ":
                print(f"[POS_MGR] Keeping SQQQ crisis hedge — skipping")
                continue
            if qty < 0:
                print(f"[POS_MGR] Keeping short {pos.symbol} ({qty}) — crisis short, skipping")
                continue
            close_position(pos.symbol, "CRISIS_HALT")
        return

    # EOD close — 15 min before close, exit everything
    if is_near_close():
        print("[POS_MGR] Near market close — closing all positions (EOD rule)")
        for pos in positions:
            close_position(pos.symbol, "EOD_CLOSE")
        return

    for pos in positions:
        symbol = pos.symbol
        unrealized_pct = float(pos.unrealized_plpc)
        unrealized_pnl = float(pos.unrealized_pl)

        print(f"[POS_MGR] {symbol}: P&L={unrealized_pnl:+.2f} ({unrealized_pct:+.2%})")

        # 1. Stop loss / take profit (regime-adjusted)
        exit_reason = check_stop_take(pos, regime)
        if exit_reason:
            close_position(symbol, exit_reason)
            continue

        # 2. Signal-based exit
        exit_reason = check_signal_exit(pos, sovereign_results, madman_results)
        if exit_reason:
            close_position(symbol, exit_reason)
            continue

        print(f"[POS_MGR] {symbol}: holding — no exit trigger")
