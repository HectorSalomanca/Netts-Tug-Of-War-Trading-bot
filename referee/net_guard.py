"""
net_guard.py — Offline-resilient API wrapper for Tug-Of-War

Provides:
  - @with_retry: decorator that retries on network errors with exponential backoff
  - is_online(): fast connectivity check (pings Alpaca domain)
  - safe_call(): inline retry for one-off calls

Design: all failures return a safe default so run_cycle() continues
rather than crashing. Bracket orders on Alpaca protect capital when offline.
"""

import time
import socket
import functools
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Errors that indicate a transient network problem (not a logic bug)
_NETWORK_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
    Exception,  # broad catch — Alpaca SDK wraps errors in generic Exception
)

ALPACA_HOST = "api.alpaca.markets"
CONNECTIVITY_TIMEOUT = 3  # seconds for the ping check


def is_online() -> bool:
    """Return True if we can reach Alpaca's API host."""
    try:
        socket.setdefaulttimeout(CONNECTIVITY_TIMEOUT)
        socket.getaddrinfo(ALPACA_HOST, 443)
        return True
    except OSError:
        return False


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 5.0,
    default: Any = None,
    label: str = "",
):
    """
    Decorator: retry on any exception with exponential backoff.

    Args:
        max_attempts: total tries (1 = no retry)
        base_delay:   seconds before first retry; doubles each attempt
        default:      value returned after all attempts fail
        label:        tag for log messages (e.g. 'get_mid_price')
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            tag = label or fn.__name__
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except _NETWORK_ERRORS as e:
                    err_str = str(e)
                    # Don't retry logic errors (e.g. bad symbol, auth failure)
                    if any(code in err_str for code in ("40", "42", "403", "401", "404")):
                        print(f"[NET] {tag}: non-retryable error — {err_str[:120]}")
                        return default
                    if attempt < max_attempts:
                        print(f"[OFFLINE] {tag}: attempt {attempt}/{max_attempts} failed — retrying in {delay:.0f}s | {err_str[:80]}")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        print(f"[OFFLINE] {tag}: all {max_attempts} attempts failed — returning default | {err_str[:80]}")
            return default
        return wrapper  # type: ignore
    return decorator


def safe_call(fn: Callable, *args, default: Any = None, label: str = "", **kwargs) -> Any:
    """
    One-off retry wrapper for inline use (no decorator needed).

    Example:
        equity = safe_call(trading_client.get_account, default=None, label="get_account")
    """
    wrapped = with_retry(max_attempts=3, base_delay=5, default=default, label=label)(fn)
    return wrapped(*args, **kwargs)
