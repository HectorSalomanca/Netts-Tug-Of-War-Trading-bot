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


# ── Cross-Sectional Deviation (Optiver 1st Place Strategy) ───────────────────

def compute_cross_sectional_deviation(symbol_values: dict) -> dict:
    """
    Optiver Kaggle winning technique: subtract the cross-sectional median
    from each symbol's feature value. This strips macro-market noise.

    If the whole market's OFI spikes, NVDA's OFI spike isn't special.
    Only buy NVDA if its order flow is exceptionally aggressive vs peers.

    Args:
        symbol_values: {symbol: raw_value} for a single feature across all symbols
    Returns:
        {symbol: deviation_from_median}
    """
    if not symbol_values:
        return {}
    values = np.array(list(symbol_values.values()))
    median_val = float(np.median(values))
    return {sym: round(float(val - median_val), 6) for sym, val in symbol_values.items()}


def compute_all_deviations(watchlist_features: dict) -> dict:
    """
    Compute cross-sectional deviations for all features across the watchlist.

    Args:
        watchlist_features: {symbol: {feature_name: value, ...}, ...}
    Returns:
        {symbol: {feature_name_dev: deviation, ...}, ...}
    """
    if not watchlist_features:
        return {}

    # Collect all feature names from first symbol
    first_sym = next(iter(watchlist_features))
    feature_names = list(watchlist_features[first_sym].keys())

    # For each feature, compute cross-sectional deviation
    deviations = {sym: {} for sym in watchlist_features}
    for feat in feature_names:
        raw_values = {}
        for sym, feats in watchlist_features.items():
            val = feats.get(feat, 0.0)
            if val is not None and np.isfinite(val):
                raw_values[sym] = val

        if len(raw_values) >= 3:  # need at least 3 symbols for meaningful median
            devs = compute_cross_sectional_deviation(raw_values)
            for sym, dev in devs.items():
                deviations[sym][f"{feat}_dev"] = dev

    return deviations


def zero_sum_allocation(scores: dict) -> dict:
    """
    Optiver zero-sum post-processing: adjust scores so they sum to zero.
    This forces the portfolio to be beta-neutral within the micro-portfolio.

    If the bot wants to buy 3 tech stocks, this dynamically sizes them
    so you are actively hedging beta within the portfolio itself.

    Args:
        scores: {symbol: raw_score}
    Returns:
        {symbol: zero_sum_adjusted_score}
    """
    if not scores:
        return {}
    values = np.array(list(scores.values()))
    mean_score = float(np.mean(values))
    return {sym: round(float(val - mean_score), 6) for sym, val in scores.items()}


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
