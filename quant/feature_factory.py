"""
Feature Factory — Adaptive Fractional Differencing + SPY/QQQ Neutralization

Two jobs:
1. AFD: finds minimum d in [0,1] that makes a price series stationary
   while preserving as much trend memory as possible.
2. Feature Neutralization: strips SPY + QQQ beta from each ticker's
   returns, leaving the "Pure Alpha" residual.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Adaptive Fractional Differencing ─────────────────────────────────────────

def _get_weights(d: float, size: int) -> np.ndarray:
    """Compute fracDiff weights for a given d and window size."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def frac_diff(series: np.ndarray, d: float, threshold: float = 1e-5) -> np.ndarray:
    """Apply fractional differencing with weight cutoff threshold."""
    weights = _get_weights(d, len(series))
    weights_cumsum = np.cumsum(abs(weights))
    weights_cumsum /= weights_cumsum[-1]
    skip = len(weights_cumsum[weights_cumsum > threshold])
    result = np.full(len(series), np.nan)
    for i in range(skip, len(series)):
        w = weights[-(i + 1):]
        result[i] = np.dot(w, series[i - len(w) + 1: i + 1])
    return result


def _adf_pvalue(series: np.ndarray) -> float:
    """ADF test p-value — returns 1.0 if statsmodels unavailable."""
    try:
        from statsmodels.tsa.stattools import adfuller
        clean = series[~np.isnan(series)]
        if len(clean) < 20:
            return 1.0
        result = adfuller(clean, maxlag=1, regression="c", autolag=None)
        return float(result[1])
    except Exception:
        return 1.0


def adaptive_frac_diff(prices: np.ndarray, p_threshold: float = 0.05) -> tuple:
    """
    Find minimum d in [0,1] that achieves stationarity (ADF p < p_threshold).
    Returns (differenced_series, d_value).
    """
    for d in np.arange(0.0, 1.05, 0.1):
        diff = frac_diff(prices, round(d, 2))
        p = _adf_pvalue(diff)
        if p < p_threshold:
            return diff, round(d, 2)
    diff = frac_diff(prices, 1.0)
    return diff, 1.0


def compute_afd_momentum(prices: np.ndarray, lookback: int = 5) -> float:
    """
    Returns the AFD-based momentum: mean of last `lookback` AFD values.
    Positive = upward trend memory, negative = downward.
    """
    if len(prices) < 30:
        return 0.0
    diff, d = adaptive_frac_diff(prices)
    clean = diff[~np.isnan(diff)]
    if len(clean) < lookback:
        return 0.0
    return float(np.mean(clean[-lookback:]))


# ── Feature Neutralization ────────────────────────────────────────────────────

def compute_returns(prices: np.ndarray) -> np.ndarray:
    if len(prices) < 2:
        return np.array([])
    return np.diff(prices) / prices[:-1]


def neutralize(
    ticker_prices: np.ndarray,
    spy_prices: np.ndarray,
    qqq_prices: np.ndarray,
) -> np.ndarray:
    """
    Strip SPY + QQQ beta from ticker returns via OLS regression.
    Returns the residual (Pure Alpha) return series.
    """
    min_len = min(len(ticker_prices), len(spy_prices), len(qqq_prices))
    if min_len < 10:
        return compute_returns(ticker_prices)

    r_ticker = compute_returns(ticker_prices[-min_len:])
    r_spy    = compute_returns(spy_prices[-min_len:])
    r_qqq    = compute_returns(qqq_prices[-min_len:])

    n = min(len(r_ticker), len(r_spy), len(r_qqq))
    r_ticker = r_ticker[-n:]
    r_spy    = r_spy[-n:]
    r_qqq    = r_qqq[-n:]

    X = np.column_stack([np.ones(n), r_spy, r_qqq])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, r_ticker, rcond=None)
        residuals = r_ticker - (beta[0] + beta[1] * r_spy + beta[2] * r_qqq)
        return residuals
    except Exception:
        return r_ticker


def compute_pure_alpha_momentum(
    ticker_prices: np.ndarray,
    spy_prices: np.ndarray,
    qqq_prices: np.ndarray,
    lookback: int = 5,
) -> float:
    """
    Returns mean of last `lookback` pure-alpha residual returns.
    Positive = ticker outperforming market on its own merit.
    """
    residuals = neutralize(ticker_prices, spy_prices, qqq_prices)
    if len(residuals) < lookback:
        return 0.0
    return float(np.mean(residuals[-lookback:]))


# ── Combined Feature Vector ───────────────────────────────────────────────────

def build_features(
    ticker_prices: np.ndarray,
    spy_prices: np.ndarray,
    qqq_prices: np.ndarray,
) -> dict:
    """
    Full feature vector for a single ticker.
    Returns dict with afd_momentum, pure_alpha, d_value, neutralized_vol.
    """
    afd_series, d_val = adaptive_frac_diff(ticker_prices)
    clean_afd = afd_series[~np.isnan(afd_series)]
    afd_mom = float(np.mean(clean_afd[-5:])) if len(clean_afd) >= 5 else 0.0

    residuals = neutralize(ticker_prices, spy_prices, qqq_prices)
    pure_alpha = float(np.mean(residuals[-5:])) if len(residuals) >= 5 else 0.0
    neutralized_vol = float(np.std(residuals[-20:])) * np.sqrt(252) if len(residuals) >= 20 else 0.3

    return {
        "afd_momentum": round(afd_mom, 6),
        "pure_alpha": round(pure_alpha, 6),
        "d_value": d_val,
        "neutralized_vol": round(neutralized_vol, 4),
    }
