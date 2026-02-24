"""
Alpha Factory — WorldQuant 101 Formulaic Alphas for Tug-Of-War

Generates 20 uncorrelated micro-alphas from price/volume data.
Each alpha is a simple mathematical formula that captures a specific
market microstructure pattern. Fed directly into Ridge regression
alongside Stockformer and HMM for a dense, uncorrelated feature vector.

Reference: Kakushadze, Z. (2016) "101 Formulaic Alphas"
           https://arxiv.org/abs/1601.00991

Design:
  - Each alpha function takes (close, open, high, low, volume, vwap) arrays
  - Returns a single float score for the current bar
  - All alphas are normalized to roughly [-1, 1] via rank-percentile
  - compute_all_alphas() returns {alpha_name: score} for one symbol
  - compute_watchlist_alphas() returns cross-sectional ranked alphas
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Helper Functions ─────────────────────────────────────────────────────────

def _rank(x: np.ndarray) -> np.ndarray:
    """Rank values and normalize to [0, 1]."""
    temp = x.argsort().argsort()
    return temp / (len(temp) - 1 + 1e-8)


def _delta(x: np.ndarray, d: int = 1) -> np.ndarray:
    """x[t] - x[t-d]."""
    result = np.zeros_like(x)
    result[d:] = x[d:] - x[:-d]
    return result


def _ts_rank(x: np.ndarray, window: int = 10) -> float:
    """Rank of current value within last `window` values, normalized to [0, 1]."""
    if len(x) < window:
        return 0.5
    recent = x[-window:]
    return float(np.searchsorted(np.sort(recent), recent[-1]) / (window - 1 + 1e-8))


def _ts_min(x: np.ndarray, window: int = 10) -> float:
    if len(x) < window:
        return float(x[-1]) if len(x) > 0 else 0.0
    return float(np.min(x[-window:]))


def _ts_max(x: np.ndarray, window: int = 10) -> float:
    if len(x) < window:
        return float(x[-1]) if len(x) > 0 else 0.0
    return float(np.max(x[-window:]))


def _ts_argmax(x: np.ndarray, window: int = 10) -> float:
    """Position of max within last window, normalized to [0, 1]."""
    if len(x) < window:
        return 0.5
    return float(np.argmax(x[-window:]) / (window - 1 + 1e-8))


def _ts_argmin(x: np.ndarray, window: int = 10) -> float:
    if len(x) < window:
        return 0.5
    return float(np.argmin(x[-window:]) / (window - 1 + 1e-8))


def _stddev(x: np.ndarray, window: int = 10) -> float:
    if len(x) < window:
        return 0.0
    return float(np.std(x[-window:]))


def _correlation(x: np.ndarray, y: np.ndarray, window: int = 10) -> float:
    if len(x) < window or len(y) < window:
        return 0.0
    xw, yw = x[-window:], y[-window:]
    if np.std(xw) < 1e-8 or np.std(yw) < 1e-8:
        return 0.0
    return float(np.corrcoef(xw, yw)[0, 1])


def _covariance(x: np.ndarray, y: np.ndarray, window: int = 10) -> float:
    if len(x) < window or len(y) < window:
        return 0.0
    return float(np.cov(x[-window:], y[-window:])[0, 1])


def _decay_linear(x: np.ndarray, window: int = 10) -> float:
    """Linearly-weighted moving average (recent values weighted more)."""
    if len(x) < window:
        window = len(x)
    if window == 0:
        return 0.0
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return float(np.dot(x[-window:], weights))


def _sign_delta(x: np.ndarray, d: int = 1) -> float:
    if len(x) < d + 1:
        return 0.0
    return float(np.sign(x[-1] - x[-1 - d]))


# ── WorldQuant 101 Alphas (Selected 20) ─────────────────────────────────────

def alpha_001(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#1: (rank(ts_argmax(signed_power(returns, 2), 5)) - 0.5)"""
    if len(close) < 7:
        return 0.0
    returns = np.diff(close) / (close[:-1] + 1e-8)
    signed_power = np.sign(returns) * returns**2
    return _ts_argmax(signed_power, 5) - 0.5


def alpha_002(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#2: -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/close), 6)"""
    open_ = kw.get("open", close)
    if len(close) < 8:
        return 0.0
    log_vol = np.log(volume + 1)
    d_log_vol = _delta(log_vol, 2)
    co_ratio = (close - open_) / (close + 1e-8)
    return -1.0 * _correlation(_rank(d_log_vol), _rank(co_ratio), 6)


def alpha_006(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#6: -1 * correlation(open, volume, 10)"""
    open_ = kw.get("open", close)
    return -1.0 * _correlation(open_, volume, 10)


def alpha_012(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
    return _sign_delta(volume, 1) * (-1.0 * _sign_delta(close, 1))


def alpha_015(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#15: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
    high = kw.get("high", close)
    if len(close) < 6:
        return 0.0
    corrs = []
    for i in range(3):
        offset = len(close) - 3 + i
        if offset >= 3:
            c = _correlation(_rank(high[:offset]), _rank(volume[:offset]), 3)
            corrs.append(c)
    return -1.0 * sum(corrs) / (len(corrs) + 1e-8)


def alpha_017(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#17: -1 * rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume/adv20, 5))"""
    if len(close) < 22:
        return 0.0
    adv20 = np.mean(volume[-20:])
    vol_ratio = volume / (adv20 + 1e-8)
    dd_close = _delta(_delta(close, 1), 1)
    return -1.0 * _ts_rank(close, 10) * float(np.sign(dd_close[-1])) * _ts_rank(vol_ratio, 5)


def alpha_020(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#20: -1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))"""
    open_ = kw.get("open", close)
    high = kw.get("high", close)
    low = kw.get("low", close)
    if len(close) < 2:
        return 0.0
    a = open_[-1] - high[-2]
    b = open_[-1] - close[-2]
    c = open_[-1] - low[-2]
    return -1.0 * np.sign(a) * np.sign(b) * np.sign(c)


def alpha_023(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#23: if mean(high, 20) < high then -1 * delta(high, 2) else 0"""
    high = kw.get("high", close)
    if len(high) < 22:
        return 0.0
    if np.mean(high[-20:]) < high[-1]:
        return -1.0 * (high[-1] - high[-3]) / (high[-3] + 1e-8)
    return 0.0


def alpha_026(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#26: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)"""
    high = kw.get("high", close)
    if len(close) < 13:
        return 0.0
    corrs = []
    for i in range(3):
        offset = len(close) - 3 + i + 1
        if offset >= 10:
            c = _correlation(
                np.array([_ts_rank(volume[:j], 5) for j in range(offset-5, offset)]),
                np.array([_ts_rank(high[:j], 5) for j in range(offset-5, offset)]),
                5
            )
            corrs.append(c)
    return -1.0 * max(corrs) if corrs else 0.0


def alpha_028(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#28: scale(correlation(adv20, low, 5) + (high+low)/2 - close)"""
    high = kw.get("high", close)
    low = kw.get("low", close)
    if len(close) < 22:
        return 0.0
    adv20 = np.convolve(volume, np.ones(20)/20, mode='same')
    corr = _correlation(adv20, low, 5)
    raw = corr + (high[-1] + low[-1]) / 2 - close[-1]
    return float(np.clip(raw / (abs(raw) + 1e-8), -1, 1))


def alpha_033(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#33: rank(-1 + (open/close))"""
    open_ = kw.get("open", close)
    if len(close) < 1:
        return 0.0
    return float(np.clip(-1 + open_[-1] / (close[-1] + 1e-8), -1, 1))


def alpha_034(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#34: rank((1 - rank(stddev(returns, 2)/stddev(returns, 5))) + (1 - rank(delta(close, 1))))"""
    if len(close) < 7:
        return 0.0
    returns = np.diff(close) / (close[:-1] + 1e-8)
    std2 = _stddev(returns, 2)
    std5 = _stddev(returns, 5)
    ratio = std2 / (std5 + 1e-8)
    dc = (close[-1] - close[-2]) / (close[-2] + 1e-8)
    return float(np.clip((1 - ratio) + (1 - dc), -1, 1))


def alpha_038(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#38: -1 * rank(ts_rank(close, 10)) * rank(close / open)"""
    open_ = kw.get("open", close)
    if len(close) < 10:
        return 0.0
    return -1.0 * _ts_rank(close, 10) * (close[-1] / (open_[-1] + 1e-8) - 1.0)


def alpha_041(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#41: power(high * low, 0.5) - vwap"""
    high = kw.get("high", close)
    low = kw.get("low", close)
    vwap = kw.get("vwap", close)
    if len(close) < 1:
        return 0.0
    raw = np.sqrt(high[-1] * low[-1]) - vwap[-1]
    return float(np.clip(raw / (vwap[-1] + 1e-8), -1, 1))


def alpha_044(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#44: -1 * correlation(high, rank(volume), 5)"""
    high = kw.get("high", close)
    if len(close) < 5:
        return 0.0
    return -1.0 * _correlation(high, _rank(volume), 5)


def alpha_049(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#49: if delta(delay(close,10),10)/10 - delta(close,10)/10 < -0.1*close then 1 else -delta(close,1)"""
    if len(close) < 21:
        return 0.0
    cond = (close[-11] - close[-21]) / 10 - (close[-1] - close[-11]) / 10
    if cond < -0.1 * close[-1]:
        return 1.0
    return -1.0 * _sign_delta(close, 1)


def alpha_052(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#52: (-ts_min(low, 5) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20))/220)) * ts_rank(volume, 5)"""
    low = kw.get("low", close)
    if len(close) < 22:
        return 0.0
    returns = np.diff(close) / (close[:-1] + 1e-8)
    ts_min_5 = _ts_min(low, 5)
    ts_min_5_lag = _ts_min(low[:-5], 5) if len(low) > 10 else ts_min_5
    mom_diff = (np.sum(returns[-20:]) - np.sum(returns[-20:])) / 20 if len(returns) >= 20 else 0
    return float(np.clip((-ts_min_5 + ts_min_5_lag) * mom_diff * _ts_rank(volume, 5), -1, 1))


def alpha_053(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#53: -1 * delta((close - low) / (high - low + 1e-8), 9)"""
    high = kw.get("high", close)
    low = kw.get("low", close)
    if len(close) < 10:
        return 0.0
    hl_ratio = (close - low) / (high - low + 1e-8)
    return float(np.clip(-1.0 * (hl_ratio[-1] - hl_ratio[-10]), -1, 1))


def alpha_054(close: np.ndarray, volume: np.ndarray, **kw) -> float:
    """Alpha#54: -1 * (low - close) * power(open, 5) / ((low - high) * power(close, 5) + 1e-8)"""
    open_ = kw.get("open", close)
    high = kw.get("high", close)
    low = kw.get("low", close)
    if len(close) < 1:
        return 0.0
    num = -1.0 * (low[-1] - close[-1]) * (open_[-1] ** 5)
    den = (low[-1] - high[-1]) * (close[-1] ** 5) + 1e-8
    return float(np.clip(num / den, -1, 1))


# ── Alpha Registry ──────────────────────────────────────────────────────────

ALPHA_FUNCTIONS = {
    "alpha_001": alpha_001,
    "alpha_002": alpha_002,
    "alpha_006": alpha_006,
    "alpha_012": alpha_012,
    "alpha_015": alpha_015,
    "alpha_017": alpha_017,
    "alpha_020": alpha_020,
    "alpha_023": alpha_023,
    "alpha_026": alpha_026,
    "alpha_028": alpha_028,
    "alpha_033": alpha_033,
    "alpha_034": alpha_034,
    "alpha_038": alpha_038,
    "alpha_041": alpha_041,
    "alpha_044": alpha_044,
    "alpha_049": alpha_049,
    "alpha_052": alpha_052,
    "alpha_053": alpha_053,
    "alpha_054": alpha_054,
}


# ── Public API ──────────────────────────────────────────────────────────────

def compute_all_alphas(
    close: np.ndarray,
    volume: np.ndarray,
    open_: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    vwap: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all 19 formulaic alphas for a single symbol.
    Returns {alpha_name: score} where score is roughly in [-1, 1].
    """
    kw = {
        "open": open_ if open_ is not None else close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "vwap": vwap if vwap is not None else close,
    }
    results = {}
    for name, fn in ALPHA_FUNCTIONS.items():
        try:
            val = fn(close, volume, **kw)
            results[name] = round(float(np.clip(val, -1.0, 1.0)), 6)
        except Exception:
            results[name] = 0.0
    return results


def compute_watchlist_alphas(watchlist_data: dict) -> dict:
    """
    Compute alphas for all symbols, then cross-sectionally rank them.
    This is the WorldQuant Alphathon approach: rank-normalize across
    the universe so alphas are comparable across symbols.

    Args:
        watchlist_data: {symbol: {close, volume, open, high, low, vwap} as np.arrays}
    Returns:
        {symbol: {alpha_name: cross_sectional_rank_score, ...}}
    """
    # Step 1: compute raw alphas per symbol
    raw_alphas = {}
    for sym, data in watchlist_data.items():
        raw_alphas[sym] = compute_all_alphas(
            close=data["close"],
            volume=data["volume"],
            open_=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            vwap=data.get("vwap"),
        )

    if len(raw_alphas) < 3:
        return raw_alphas

    # Step 2: cross-sectional rank each alpha
    alpha_names = list(ALPHA_FUNCTIONS.keys())
    ranked = {sym: {} for sym in raw_alphas}

    for alpha_name in alpha_names:
        values = {sym: raw_alphas[sym].get(alpha_name, 0.0) for sym in raw_alphas}
        vals_arr = np.array(list(values.values()))
        # Rank-percentile: 0 = worst, 1 = best
        if np.std(vals_arr) > 1e-8:
            ranks = vals_arr.argsort().argsort()
            norm_ranks = ranks / (len(ranks) - 1 + 1e-8)
            # Center around 0: [-0.5, 0.5]
            norm_ranks = norm_ranks - 0.5
        else:
            norm_ranks = np.zeros(len(vals_arr))

        for i, sym in enumerate(values.keys()):
            ranked[sym][alpha_name] = round(float(norm_ranks[i]), 4)

    return ranked


def compute_alpha_composite(ranked_alphas: dict) -> float:
    """
    Combine all ranked alphas into a single composite score.
    Simple equal-weight average of all alpha rank scores.
    Returns score in roughly [-0.5, 0.5].
    """
    if not ranked_alphas:
        return 0.0
    values = [v for v in ranked_alphas.values() if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return round(float(np.mean(values)), 4)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 60
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volume = np.random.randint(1000, 10000, n).astype(float)
    open_ = close + np.random.randn(n) * 0.2
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)

    alphas = compute_all_alphas(close, volume, open_, high, low)
    print(f"Computed {len(alphas)} alphas:")
    for name, val in sorted(alphas.items()):
        print(f"  {name}: {val:+.4f}")
