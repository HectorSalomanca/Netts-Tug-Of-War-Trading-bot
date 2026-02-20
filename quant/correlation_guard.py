"""
Correlation Guard — prevents over-concentration in correlated sectors.

Problem: NVDA, PLTR, MSTR are all high-beta tech/crypto plays.
Holding all three simultaneously = 3x exposure to the same macro move.

Rules:
  - Max 2 positions from the same sector group at once
  - If a new trade would exceed sector limit, skip it
  - Sector groups defined below (can be tuned)
"""

from typing import Optional

# Sector groupings — symbols that move together
SECTOR_GROUPS = {
    "tech_growth":   ["NVDA", "CRWD", "PLTR"],
    "crypto_proxy":  ["MSTR", "PLTR"],
    "healthcare":    ["LLY"],
    "financials":    ["JPM"],
    "industrials":   ["CAT"],
    "utilities":     ["NEE"],
    "international": ["TSMC", "SONY"],
}

MAX_PER_SECTOR = 2   # never hold more than 2 from same group


def get_symbol_sectors(symbol: str) -> list:
    """Return all sector groups a symbol belongs to."""
    return [group for group, members in SECTOR_GROUPS.items() if symbol in members]


def check_correlation_risk(
    symbol: str,
    open_positions: list,  # list of symbol strings currently held
) -> tuple:
    """
    Returns (is_safe: bool, reason: str).
    is_safe=False means skip this trade due to sector concentration.
    """
    symbol_sectors = get_symbol_sectors(symbol)

    if not symbol_sectors:
        return True, "no sector group"

    for sector in symbol_sectors:
        members = SECTOR_GROUPS[sector]
        held_in_sector = [p for p in open_positions if p in members]

        if len(held_in_sector) >= MAX_PER_SECTOR:
            reason = f"sector '{sector}' already has {len(held_in_sector)} positions ({held_in_sector})"
            print(f"[CORR_GUARD] {symbol}: BLOCKED — {reason}")
            return False, reason

    return True, "ok"


def filter_by_correlation(
    candidates: list,       # list of (symbol, verdict_dict) tuples to consider
    open_positions: list,   # currently held symbols
) -> list:
    """
    Filter a list of candidate trades, removing those that would
    create sector concentration risk.
    Returns filtered list of (symbol, verdict_dict).
    """
    approved = []
    simulated_positions = list(open_positions)

    for symbol, verdict in candidates:
        is_safe, reason = check_correlation_risk(symbol, simulated_positions)
        if is_safe:
            approved.append((symbol, verdict))
            simulated_positions.append(symbol)  # simulate holding it
        else:
            print(f"[CORR_GUARD] Skipping {symbol}: {reason}")

    return approved


if __name__ == "__main__":
    # Test
    open_pos = ["NVDA", "CRWD"]
    candidates = [
        ("PLTR", {"verdict": "execute"}),
        ("JPM",  {"verdict": "execute"}),
        ("MSTR", {"verdict": "execute"}),
        ("CAT",  {"verdict": "execute"}),
    ]
    result = filter_by_correlation(candidates, open_pos)
    print(f"\nApproved trades: {[s for s, _ in result]}")
