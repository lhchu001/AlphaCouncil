# ============================================================
# discovery/xueqiu_source.py
# ============================================================

"""Xueqiu hot-stock source adapter.

Xueqiu uses market-specific identifiers. US identifiers are converted to
Yahoo/yfinance symbols; HK identifiers are converted to 4-digit .HK symbols.
Failures return [].
"""

from __future__ import annotations

import requests

XUEQIU_URL = "https://stock.xueqiu.com/v5/stock/hotstocklist.json"
XUEQIU_LIST_TYPES = {"US": 11, "HK": 13}


def _to_yfinance_symbol(raw: str, market: str) -> str | None:
    value = str(raw or "").strip().upper()
    if not value:
        return None

    if market == "US":
        # Xueqiu may return US.AAPL or just AAPL.
        value = value.removeprefix("US.")
        value = value.replace("/", "-")
        if value.replace(".", "").replace("-", "").isalnum():
            return value
        return None

    if market == "HK":
        # Examples: HK.00700, 00700, 700.HK.
        value = value.removeprefix("HK.").removesuffix(".HK")
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return None
        return f"{int(digits):04d}.HK"

    return None


def get_xueqiu_symbols(market: str, limit: int = 50) -> list[str]:
    try:
        market = market.upper()
        if market not in XUEQIU_LIST_TYPES:
            raise ValueError("market must be US or HK")

        response = requests.get(
            XUEQIU_URL,
            params={
                "size": min(max(limit, 1), 50),
                "hotListType": XUEQIU_LIST_TYPES[market],
            },
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://xueqiu.com/",
            },
            timeout=(10, 30),
        )
        response.raise_for_status()
        payload = response.json()

        output: list[str] = []
        for item in payload.get("data", []):
            symbol = _to_yfinance_symbol(item.get("symbol"), market)
            if symbol:
                output.append(symbol)
            if len(output) >= limit:
                break

        return list(dict.fromkeys(output))
    except Exception as exc:
        print(f"Xueqiu {market} warning: {type(exc).__name__}: {exc}")
        return []


def get_xueqiu_us_symbols(limit: int = 50) -> list[str]:
    return get_xueqiu_symbols("US", limit)


def get_xueqiu_hk_symbols(limit: int = 50) -> list[str]:
    return get_xueqiu_symbols("HK", limit)
