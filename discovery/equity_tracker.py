# ============================================================
# discovery/equity_tracker.py
# ============================================================

"""Main query program.

All source adapters return lists of yfinance-compatible symbols. The main
program merges duplicates, counts independent source confirmations, and returns
an ordered list of symbols. It does not return DataFrame rows or provider-
specific objects.
"""

from __future__ import annotations

from collections import defaultdict

from .akshare_source import (
    get_akshare_hk_symbols,
    get_akshare_us_symbols,
)
from .reddit_source import (
    get_reddit_hk_symbols,
    get_reddit_us_symbols,
)
from .stocktwits_source import get_stocktwits_symbols
from .xueqiu_source import (
    get_xueqiu_hk_symbols,
    get_xueqiu_us_symbols,
)


def is_valid_yfinance_equity_symbol(
    symbol: object,
) -> bool:
    if not isinstance(symbol, str):
        return False

    value = symbol.strip().upper()

    if (
        value.isascii()
        and value.isalpha()
        and 1 <= len(value) <= 5
    ):
        return True

    return (
        len(value) == 7
        and value[:4].isdigit()
        and value[4:] == ".HK"
    )


def _merge_ranked_sources(
    source_results: list[tuple[str, list[str]]],
    top_n: int,
) -> list[str]:
    scores: defaultdict[str, float] = defaultdict(float)
    source_count: defaultdict[str, int] = defaultdict(int)

    for _, symbols in source_results:
        unique_symbols = list(
            dict.fromkeys(symbols)
        )

        for rank, raw_symbol in enumerate(
            unique_symbols,
            start=1,
        ):
            symbol = str(raw_symbol).strip().upper()

            if not is_valid_yfinance_equity_symbol(
                symbol
            ):
                continue

            scores[symbol] += max(
                1.0,
                len(unique_symbols) - rank + 1,
            )
            source_count[symbol] += 1

    ordered = sorted(
        scores,
        key=lambda symbol: (
            source_count[symbol],
            scores[symbol],
            symbol,
        ),
        reverse=True,
    )

    return ordered[:max(0, top_n)]


def find_hot_equity_topics(
    *,
    xueqiu_us: bool = False,
    xueqiu_hk: bool = False,
    akshare_us: bool = False,
    akshare_hk: bool = False,
    stocktwits: bool = False,
    reddit_us: bool = False,
    reddit_hk: bool = False,
    top_n: int = 20,
) -> list[str]:
    """Return ranked US/HK Yahoo Finance symbols."""
    source_results: list[tuple[str, list[str]]] = []

    if xueqiu_us:
        source_results.append(
            (
                "xueqiu_us",
                get_xueqiu_us_symbols(),
            )
        )

    if xueqiu_hk:
        source_results.append(
            (
                "xueqiu_hk",
                get_xueqiu_hk_symbols(),
            )
        )

    if akshare_us:
        source_results.append(
            (
                "akshare_us",
                get_akshare_us_symbols(),
            )
        )

    if akshare_hk:
        source_results.append(
            (
                "akshare_hk",
                get_akshare_hk_symbols(),
            )
        )

    if stocktwits:
        source_results.append(
            (
                "stocktwits",
                get_stocktwits_symbols(),
            )
        )

    if reddit_us:
        source_results.append(
            (
                "reddit_us",
                get_reddit_us_symbols(),
            )
        )

    if reddit_hk:
        source_results.append(
            (
                "reddit_hk",
                get_reddit_hk_symbols(),
            )
        )

    return _merge_ranked_sources(
        source_results,
        top_n,
    )
