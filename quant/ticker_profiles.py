"""
Ticker Personality Profiles — per-symbol behavioral configs for the Madman agent.

Each ticker has:
  - behavior: "momentum_breakout" or "mean_reversion" (9:45 AM micro-regime)
  - sigma_method: "implied" (ATM straddle proxy) or "historical"
  - pump_mult: multiplier on σ to define a "pump" (higher = less sensitive)
  - spread_pct_90: historical 90th percentile bid-ask spread at 9:45 AM
  - sector: for correlation guard cross-reference
  - alt_data_sources: list of high-alpha alternative data sources
  - alt_keywords: trigger keywords for alt data scoring
"""

from typing import Optional
import numpy as np


PROFILES = {
    "NVDA": {
        "behavior": "momentum_breakout",
        "sigma_method": "implied",
        "pump_mult": 2.0,       # 2σ move = pump (high beta)
        "spread_pct_90": 0.03,  # 3 bps typical 9:45 spread
        "sector": "tech_growth",
        "alt_data_sources": ["tsmc_guidance", "supply_chain"],
        "alt_keywords": ["CoWoS", "HBM", "wafer yield", "AI chip", "data center"],
    },
    "CRWD": {
        "behavior": "momentum_breakout",
        "sigma_method": "historical",
        "pump_mult": 2.0,
        "spread_pct_90": 0.05,
        "sector": "tech_growth",
        "alt_data_sources": ["cve_database", "github_commits"],
        "alt_keywords": ["zero-day", "CVE", "breach", "patch", "ransomware", "Falcon"],
    },
    "LLY": {
        "behavior": "event_driven",
        "sigma_method": "historical",
        "pump_mult": 2.5,
        "spread_pct_90": 0.02,
        "sector": "healthcare",
        "alt_data_sources": ["fda_rss", "clinicaltrials"],
        "alt_keywords": ["Phase 3", "fast-track", "FDA approval", "efficacy", "GLP-1", "tirzepatide"],
    },
    "TSMC": {
        "behavior": "mean_reversion",
        "sigma_method": "historical",
        "pump_mult": 2.5,
        "spread_pct_90": 0.04,
        "sector": "international",
        "alt_data_sources": ["earnings_call", "supply_chain"],
        "alt_keywords": ["N3", "N2", "CoWoS", "capacity", "utilization", "wafer"],
    },
    "JPM": {
        "behavior": "mean_reversion",
        "sigma_method": "historical",
        "pump_mult": 3.0,       # low beta, needs bigger move
        "spread_pct_90": 0.02,
        "sector": "financials",
        "alt_data_sources": ["fed_minutes", "13f_filings"],
        "alt_keywords": ["NII", "provisions", "buyback", "dividend", "Basel"],
    },
    "NEE": {
        "behavior": "mean_reversion",
        "sigma_method": "historical",
        "pump_mult": 3.0,
        "spread_pct_90": 0.03,
        "sector": "utilities",
        "alt_data_sources": ["energy_policy"],
        "alt_keywords": ["renewable", "solar", "rate case", "PPA", "IRA"],
    },
    "CAT": {
        "behavior": "mean_reversion",
        "sigma_method": "historical",
        "pump_mult": 2.5,
        "spread_pct_90": 0.03,
        "sector": "industrials",
        "alt_data_sources": ["infrastructure_policy"],
        "alt_keywords": ["infrastructure", "backlog", "mining", "construction", "tariff"],
    },
    "SONY": {
        "behavior": "mean_reversion",
        "sigma_method": "historical",
        "pump_mult": 2.5,
        "spread_pct_90": 0.05,
        "sector": "international",
        "alt_data_sources": ["gaming_data", "entertainment"],
        "alt_keywords": ["PS5", "PlayStation", "sensor", "music", "streaming"],
    },
    "PLTR": {
        "behavior": "momentum_breakout",
        "sigma_method": "historical",
        "pump_mult": 1.8,       # very high beta
        "spread_pct_90": 0.04,
        "sector": "tech_growth",
        "alt_data_sources": ["gov_contracts", "13f_filings"],
        "alt_keywords": ["AIP", "contract", "DoD", "Army", "government", "Gotham"],
    },
    "MSTR": {
        "behavior": "momentum_breakout",
        "sigma_method": "implied",
        "pump_mult": 1.5,       # extreme beta, crypto proxy
        "spread_pct_90": 0.08,
        "sector": "crypto_proxy",
        "alt_data_sources": ["bitcoin_price", "treasury_filings"],
        "alt_keywords": ["BTC", "bitcoin", "treasury", "convertible", "Saylor"],
    },
}

# Default profile for unknown tickers
DEFAULT_PROFILE = {
    "behavior": "mean_reversion",
    "sigma_method": "historical",
    "pump_mult": 2.5,
    "spread_pct_90": 0.05,
    "sector": "unknown",
    "alt_data_sources": [],
    "alt_keywords": [],
}


def get_profile(symbol: str) -> dict:
    """Return the personality profile for a symbol."""
    return PROFILES.get(symbol, DEFAULT_PROFILE)


def compute_dynamic_pump_threshold(
    symbol: str,
    recent_closes: np.ndarray,
    window: int = 20,
) -> float:
    """
    Compute a dynamic pump threshold based on ticker personality and recent volatility.
    Returns the minimum % move to classify as a "pump".

    pump_threshold = realized_sigma * pump_mult
    """
    profile = get_profile(symbol)
    if len(recent_closes) < window + 1:
        return 0.05  # fallback 5%

    returns = np.diff(recent_closes[-window - 1:]) / recent_closes[-window - 1:-1]
    sigma = float(np.std(returns))

    # Annualize then scale to 15-min (intraday)
    # Daily sigma → 15-min sigma ≈ daily / sqrt(26 bars per day)
    sigma_15m = sigma / np.sqrt(26)

    threshold = sigma_15m * profile["pump_mult"]
    return max(threshold, 0.005)  # floor at 0.5%


def is_spread_too_wide(
    symbol: str,
    current_spread_pct: float,
) -> bool:
    """
    Check if current bid-ask spread exceeds the 90th percentile for this ticker.
    If True, the Madman agent should pause limit IOC execution.
    """
    profile = get_profile(symbol)
    return current_spread_pct > profile["spread_pct_90"]


def get_behavior(symbol: str) -> str:
    """Return 'momentum_breakout', 'mean_reversion', or 'event_driven'."""
    return get_profile(symbol)["behavior"]


def get_alt_keywords(symbol: str) -> list:
    """Return high-alpha trigger keywords for this ticker."""
    return get_profile(symbol)["alt_keywords"]


if __name__ == "__main__":
    import json
    for sym, prof in PROFILES.items():
        print(f"{sym}: {prof['behavior']} | pump_mult={prof['pump_mult']} | sector={prof['sector']}")
