# ============================================================
# discovery/stocktwits_source.py
# ============================================================

"""StockTwits source adapter.

Returns valid yfinance-compatible symbols. StockTwits is primarily a US source.
Failures return [].
"""

from __future__ import annotations

import requests


STOCKTWITS_URL = (
    "https://api.stocktwits.com/api/2/trending/symbols/equities.json"
)


def _valid_us_equity_symbol(value: object) -> str | None:
    symbol = str(value or "").strip().upper()

    # US equity symbols for this project must be plain alphabetic tickers.
    # Reject .X, .TO, .TSX, -USD, and other qualified symbols.
    if (
        not symbol
        or not symbol.isascii()
        or not symbol.isalpha()
        or not 1 <= len(symbol) <= 5
    ):
        return None

    return symbol


def get_stocktwits_symbols(limit: int = 30) -> list[str]:
    try:
        response = requests.get(
            STOCKTWITS_URL,
            headers={
                "User-Agent": "HotEquityTopics/1.0",
                "Accept": "application/json",
            },
            timeout=(10, 30),
        )
        response.raise_for_status()

        payload = response.json()
        output: list[str] = []

        for item in payload.get("symbols", []):
            symbol = _valid_us_equity_symbol(item.get("symbol"))

            if symbol and symbol not in output:
                output.append(symbol)

            if len(output) >= limit:
                break

        return output

    except Exception as exc:
        print(
            f"StockTwits warning: "
            f"{type(exc).__name__}: {exc}"
        )
        return []