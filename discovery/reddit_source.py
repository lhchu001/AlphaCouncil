# ============================================================
# discovery/reddit_source.py
# ============================================================

"""Reddit source adapter. Requires PRAW credentials in environment variables.

REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
Failures return [].


Reddit source adapter for US and Hong Kong equities.

Public interface:
    get_reddit_us_symbols()
    get_reddit_hk_symbols()
    get_reddit_symbols(market="US" | "HK" | "ALL")

All functions return Yahoo Finance-compatible equity symbols. Any exception
returns an empty list, as required by the source-adapter contract.

Reddit is much stronger for US equities than Hong Kong equities. HK coverage
is attempted through HK-focused subreddits and explicit HK ticker forms, but
results may legitimately be empty.
"""

from __future__ import annotations

from collections import Counter
import os
import re
from typing import Iterable

try:
    import praw
except ImportError:
    praw = None


US_SUBREDDITS = ("wallstreetbets", "stocks", "StockMarket")
HK_SUBREDDITS = ("HongKong", "HongKongFinance", "HK_Stocks", "AsiaTrading")

# Common English words that can appear after a dollar sign but are not tickers.
IGNORED_SYMBOLS = {
    "A", "AM", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "FOR", "FROM",
    "HAS", "HAVE", "I", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "NOT",
    "OF", "ON", "OR", "OUR", "SO", "THE", "TO", "UP", "US", "WAS", "WE",
    "WHAT", "WHEN", "WHO", "WHY", "WITH", "YOU", "YOUR",
}

CASHTAG_PATTERN = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z]{1,6})(?![A-Za-z])")
US_BARE_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5})(?![A-Za-z0-9])")
HK_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:HKEX[:.\s-]*)?([0-9]{1,5})(?:\.HK)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _reddit_client():
    if praw is None:
        raise RuntimeError("praw is not installed")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all((client_id, client_secret, user_agent)):
        raise RuntimeError(
            "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def _titles(reddit, subreddits: Iterable[str], posts_per_subreddit: int):
    for subreddit_name in subreddits:
        for submission in reddit.subreddit(subreddit_name).hot(
            limit=posts_per_subreddit
        ):
            title = str(getattr(submission, "title", ""))
            if title:
                yield title


def _normalise_us_symbol(raw: str) -> str | None:
    symbol = raw.upper().strip()
    if symbol in IGNORED_SYMBOLS:
        return None
    if not 1 <= len(symbol) <= 5:
        return None
    return symbol


def _normalise_hk_symbol(raw: str) -> str | None:
    digits = "".join(character for character in raw if character.isdigit())
    if not digits:
        return None
    return f"{int(digits):04d}.HK"


def get_reddit_us_symbols(
    limit: int = 50,
    posts_per_subreddit: int = 50,
) -> list[str]:
    """Return US symbols from US-focused Reddit communities."""
    try:
        reddit = _reddit_client()
        counter: Counter[str] = Counter()

        for title in _titles(reddit, US_SUBREDDITS, posts_per_subreddit):
            # Require cashtags for the main result to reduce false positives.
            for raw_symbol in CASHTAG_PATTERN.findall(title):
                symbol = _normalise_us_symbol(raw_symbol)
                if symbol:
                    counter[symbol] += 1

        return [symbol for symbol, _ in counter.most_common(limit)]
    except Exception as exc:
        print(f"Reddit US warning: {type(exc).__name__}: {exc}")
        return []


def get_reddit_hk_symbols(
    limit: int = 50,
    posts_per_subreddit: int = 50,
) -> list[str]:
    """Return HK symbols from HK/Asia-focused Reddit communities."""
    try:
        reddit = _reddit_client()
        counter: Counter[str] = Counter()

        for title in _titles(reddit, HK_SUBREDDITS, posts_per_subreddit):
            for raw_symbol in HK_PATTERN.findall(title):
                symbol = _normalise_hk_symbol(raw_symbol)
                if symbol:
                    counter[symbol] += 1

        return [symbol for symbol, _ in counter.most_common(limit)]
    except Exception as exc:
        print(f"Reddit HK warning: {type(exc).__name__}: {exc}")
        return []


def get_reddit_symbols(
    market: str = "ALL",
    limit: int = 50,
    posts_per_subreddit: int = 50,
) -> list[str]:
    """Return US, HK, or combined Reddit symbols."""
    try:
        market = market.upper()
        if market == "US":
            return get_reddit_us_symbols(limit, posts_per_subreddit)
        if market == "HK":
            return get_reddit_hk_symbols(limit, posts_per_subreddit)
        if market == "ALL":
            us = get_reddit_us_symbols(limit, posts_per_subreddit)
            hk = get_reddit_hk_symbols(limit, posts_per_subreddit)
            return list(dict.fromkeys(us + hk))[:limit]
        raise ValueError("market must be US, HK, or ALL")
    except Exception as exc:
        print(f"Reddit warning: {type(exc).__name__}: {exc}")
        return []