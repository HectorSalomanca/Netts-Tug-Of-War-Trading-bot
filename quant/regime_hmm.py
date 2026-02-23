"""
Regime HMM V3 — 4-State Gaussian Hidden Markov Model

States:
  0 = Chop       (low volatility, mean-reverting, range-bound)
  1 = Trend-Bull (directional up momentum, normal vol)
  2 = Trend-Bear (directional down momentum, normal vol)
  3 = Crisis     (volatility spike, fat tails, all trading halted)

V3 Upgrades:
  - 4 emission features: realized vol, momentum, OFI cross-asset avg, VIX proxy
  - 4 states (split Trend into Bull/Bear for directional regime)
  - pomegranate backend for M1 ARM performance (hmmlearn fallback)
  - get_latest_regime_full() returns confidence + all metadata

Trained on SPY daily returns. Updates every 60 minutes.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
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

N_STATES = 4
STATE_NAMES_V3 = {0: "chop", 1: "trend_bull", 2: "trend_bear", 3: "crisis"}
# Backward compat: map V3 states to V2 names for engine
STATE_TO_V2 = {"chop": "chop", "trend_bull": "trend", "trend_bear": "trend", "crisis": "crisis"}


def get_spy_history(days: int = 252) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs("SPY", level=0)
        return df.sort_index()
    except Exception as e:
        print(f"[HMM] SPY fetch error: {e}")
        return None


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Four features per day (V3):
      - Realized volatility (20-day rolling std of returns, annualized)
      - 5-day momentum (return over last 5 days)
      - Volatility ratio (VIX proxy): 5-day vol / 20-day vol
      - Momentum acceleration: 5-day mom minus 20-day mom
    """
    closes = df["close"].values
    returns = np.diff(closes) / closes[:-1]

    vol_window = 20
    mom_window = 5
    n = len(returns)

    features = []
    for i in range(vol_window, n):
        vol_20 = np.std(returns[i - vol_window:i]) * np.sqrt(252)
        vol_5 = np.std(returns[max(i - mom_window, 0):i]) * np.sqrt(252) if i >= mom_window else vol_20
        mom_5 = (closes[i + 1] - closes[i + 1 - mom_window]) / closes[i + 1 - mom_window]
        mom_20 = (closes[i + 1] - closes[i + 1 - vol_window]) / closes[i + 1 - vol_window]

        # VIX proxy: short-term vol / long-term vol (>1 = vol expanding)
        vix_proxy = vol_5 / vol_20 if vol_20 > 0 else 1.0
        # Momentum acceleration
        mom_accel = mom_5 - mom_20

        features.append([vol_20, mom_5, vix_proxy, mom_accel])

    return np.array(features)


def fit_hmm(features: np.ndarray) -> object:
    """Fit a 4-state Gaussian HMM. Tries hmmlearn, no external fallback needed."""
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=300,
            random_state=42,
            tol=1e-4,
        )
        model.fit(X)
        return model, scaler
    except Exception as e:
        print(f"[HMM] Fit error: {e}")
        return None, None


def _assign_state_labels(model, scaler) -> dict:
    """
    Map HMM hidden states 0-3 to chop/trend_bull/trend_bear/crisis.
    Highest vol = crisis, lowest vol = chop.
    Of the two middle states: positive momentum = trend_bull, negative = trend_bear.
    """
    try:
        means = scaler.inverse_transform(model.means_)
        vol_by_state = {i: means[i][0] for i in range(N_STATES)}
        mom_by_state = {i: means[i][1] for i in range(N_STATES)}
        sorted_by_vol = sorted(vol_by_state, key=vol_by_state.get)

        # Lowest vol = chop, highest vol = crisis
        chop_state = sorted_by_vol[0]
        crisis_state = sorted_by_vol[-1]

        # Middle two: split by momentum direction
        middle = [s for s in sorted_by_vol if s not in (chop_state, crisis_state)]
        if len(middle) == 2:
            if mom_by_state[middle[0]] >= mom_by_state[middle[1]]:
                bull_state, bear_state = middle[0], middle[1]
            else:
                bull_state, bear_state = middle[1], middle[0]
        elif len(middle) == 1:
            # Fallback: only 1 middle state
            bull_state = middle[0]
            bear_state = middle[0]
        else:
            bull_state = sorted_by_vol[1]
            bear_state = sorted_by_vol[1]

        return {
            chop_state: "chop",
            bull_state: "trend_bull",
            bear_state: "trend_bear",
            crisis_state: "crisis",
        }
    except Exception:
        return {0: "chop", 1: "trend_bull", 2: "trend_bear", 3: "crisis"}


def infer_current_regime() -> dict:
    """
    Full pipeline: fetch SPY → build features → fit HMM → decode current state.
    Returns dict with state name, confidence, and raw features.
    """
    df = get_spy_history(days=252)
    if df is None or len(df) < 60:
        print("[HMM] Insufficient data — defaulting to trend")
        return {"state": "trend", "confidence": 0.5, "spy_volatility": 0.15, "spy_momentum": 0.0}

    features = build_feature_matrix(df)
    if len(features) < 30:
        return {"state": "trend", "confidence": 0.5, "spy_volatility": 0.15, "spy_momentum": 0.0}

    model, scaler = fit_hmm(features)
    if model is None:
        return {"state": "trend", "confidence": 0.5, "spy_volatility": 0.15, "spy_momentum": 0.0}

    state_map = _assign_state_labels(model, scaler)

    X_scaled = scaler.transform(features)
    hidden_states = model.predict(X_scaled)
    posteriors = model.predict_proba(X_scaled)

    current_hidden = int(hidden_states[-1])
    current_state = state_map.get(current_hidden, "trend_bull")
    confidence = float(posteriors[-1][current_hidden])

    current_vol  = float(features[-1][0])
    current_mom  = float(features[-1][1])
    current_vix  = float(features[-1][2]) if features.shape[1] > 2 else 1.0
    current_accel = float(features[-1][3]) if features.shape[1] > 3 else 0.0

    # FIX: Crisis guard — only label as crisis if vol is genuinely elevated
    # Annualized vol must be >35% (SPY crisis threshold) to call it crisis.
    # Prevents mislabeling normal trending days as crisis.
    CRISIS_VOL_FLOOR = 0.35
    if current_state == "crisis" and current_vol < CRISIS_VOL_FLOOR:
        # Downgrade to trend_bear (high vol but not crisis-level)
        current_state = "trend_bear"
        print(f"[HMM] Crisis downgraded to trend_bear (vol={current_vol:.3f} < {CRISIS_VOL_FLOOR})")

    # Map V3 state to V2 for backward compat with engine
    v2_state = STATE_TO_V2.get(current_state, "trend")

    print(f"[HMM] Regime: {current_state.upper()} (v2={v2_state}) (conf={confidence:.2%}, vol={current_vol:.3f}, mom={current_mom:.4f}, vix_proxy={current_vix:.2f})")

    return {
        "state": v2_state,
        "state_v3": current_state,
        "confidence": round(confidence, 4),
        "spy_volatility": round(current_vol, 4),
        "spy_momentum": round(current_mom, 6),
        "vix_proxy": round(current_vix, 4),
        "momentum_accel": round(current_accel, 6),
    }


def log_regime(regime: dict):
    """Write current regime state to Supabase."""
    if not USER_ID:
        return
    try:
        supabase.table("regime_states").insert({
            "state": regime["state"],
            "confidence": regime["confidence"],
            "spy_volatility": regime["spy_volatility"],
            "spy_momentum": regime["spy_momentum"],
            "user_id": USER_ID,
        }).execute()
    except Exception as e:
        print(f"[HMM] Supabase log error: {e}")


def get_latest_regime() -> str:
    """
    Fast path: read the most recent regime from Supabase.
    Falls back to 'trend' if no record found.
    """
    r = get_latest_regime_full()
    return r["state"]


def get_latest_regime_full() -> dict:
    """
    Returns full regime dict: {state, state_v3, confidence, spy_volatility, spy_momentum}.
    Falls back to trend/0.5 defaults if no record found.
    """
    try:
        result = (
            supabase.table("regime_states")
            .select("state, confidence, spy_volatility, spy_momentum, created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            age_seconds = (
                datetime.now(timezone.utc) -
                datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            ).total_seconds()
            if age_seconds < 1800:
                state = row["state"]
                vol = row.get("spy_volatility", 0.0)
                mom = row.get("spy_momentum", 0.0)
                # Reconstruct state_v3 from v2 state + momentum direction
                if state == "trend":
                    state_v3 = "trend_bull" if mom >= 0 else "trend_bear"
                else:
                    state_v3 = state
                return {
                    "state":          state,
                    "state_v3":       state_v3,
                    "confidence":     row.get("confidence", 0.5),
                    "spy_volatility": vol,
                    "spy_momentum":   mom,
                }
    except Exception:
        pass
    return {"state": "trend", "state_v3": "trend_bull", "confidence": 0.5, "spy_volatility": 0.0, "spy_momentum": 0.0}


def run_and_log() -> dict:
    """Infer regime and log it. Called by the scheduler."""
    regime = infer_current_regime()
    log_regime(regime)
    return regime


if __name__ == "__main__":
    result = run_and_log()
    print(result)
