"""
Earnings Filter — skip any symbol with earnings within 48 hours.

Uses Yahoo Finance earnings calendar (no API key needed).
Called by engine_v2.py before executing any trade.
"""

import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional
from bs4 import BeautifulSoup

EARNINGS_WINDOW_HOURS = 48


def _fetch_yahoo_earnings_date(symbol: str) -> Optional[datetime]:
    """Scrape Yahoo Finance for next earnings date."""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            r = client.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")

        # Look for "Earnings Date" in the summary table
        for row in soup.find_all("li"):
            label = row.find("span", class_=lambda c: c and "label" in c.lower())
            value = row.find("span", class_=lambda c: c and "value" in c.lower())
            if label and value and "earnings" in label.get_text(strip=True).lower():
                raw = value.get_text(strip=True)
                # Parse formats like "Feb 26, 2026" or "Feb 26 - Mar 2, 2026"
                raw = raw.split(" - ")[0].strip()
                try:
                    dt = datetime.strptime(raw, "%b %d, %Y")
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

        # Fallback: search for any date-like text near "Earnings"
        text = soup.get_text()
        import re
        matches = re.findall(r"Earnings Date[^\n]*?(\w+ \d+, \d{4})", text)
        if matches:
            try:
                dt = datetime.strptime(matches[0], "%b %d, %Y")
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

    except Exception:
        pass
    return None


def is_earnings_soon(symbol: str, window_hours: int = EARNINGS_WINDOW_HOURS) -> bool:
    """
    Returns True if earnings are within `window_hours` from now.
    Defaults to False (allow trade) if we can't determine the date.
    """
    earnings_date = _fetch_yahoo_earnings_date(symbol)
    if earnings_date is None:
        return False  # can't confirm → allow trade (conservative)

    now = datetime.now(timezone.utc)
    delta = earnings_date - now
    hours_away = delta.total_seconds() / 3600

    if 0 <= hours_away <= window_hours:
        print(f"[EARNINGS] {symbol}: earnings in {hours_away:.1f}h — SKIP")
        return True

    return False


# Cache to avoid hammering Yahoo on every cycle
_earnings_cache: dict = {}
_cache_ttl_seconds = 3600  # refresh every hour


def check_earnings_risk(symbol: str) -> bool:
    """Cached version of is_earnings_soon."""
    now = datetime.now(timezone.utc)
    cached = _earnings_cache.get(symbol)
    if cached:
        result, fetched_at = cached
        if (now - fetched_at).total_seconds() < _cache_ttl_seconds:
            return result

    result = is_earnings_soon(symbol)
    _earnings_cache[symbol] = (result, now)
    return result


def filter_watchlist(symbols: list) -> tuple:
    """
    Returns (safe_symbols, blocked_symbols).
    safe_symbols = no earnings risk, ok to trade.
    blocked_symbols = earnings within 48h, skip.
    """
    safe, blocked = [], []
    for symbol in symbols:
        if check_earnings_risk(symbol):
            blocked.append(symbol)
        else:
            safe.append(symbol)

    if blocked:
        print(f"[EARNINGS] Blocked {len(blocked)} symbol(s): {blocked}")
    return safe, blocked


if __name__ == "__main__":
    test = ["NVDA", "CRWD", "LLY", "JPM", "PLTR"]
    safe, blocked = filter_watchlist(test)
    print(f"Safe: {safe}")
    print(f"Blocked: {blocked}")
