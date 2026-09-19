# ============================================================
# discovery/akshare_source.py
# ============================================================

"""AkShare quote source adapter.

The functions return yfinance-compatible symbols only. Any provider/network/
parsing exception returns [].
"""

from __future__ import annotations

import random
import time

try:
    import akshare as ak
except ImportError:
    ak = None


def _to_yfinance_symbol(raw: object, market: str) -> str | None:
    value = str(raw or "").strip().upper()
    if not value:
        return None

    if market == "US":
        value = value.removeprefix("US.")
        # Reject crypto, exchange-qualified, and non-US formats.
        if (
            "." in value
            or "-" in value
            or "/" in value
            or not value.isascii()
            or not value.isalpha()
            or not 1 <= len(value) <= 5
        ):
            return None
        return value

    if market == "HK":
        value = value.removeprefix("HK.").removesuffix(".HK")
        digits = "".join(character for character in value if character.isdigit())
        if not digits:
            return None
        return f"{int(digits):04d}.HK"

    return None


def _numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _fetch(function, market: str, limit: int, attempts: int = 3) -> list[str]:
    if ak is None:
        raise RuntimeError("akshare is not installed")

    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            frame = function()
            if frame is None or frame.empty:
                return []

            symbol_column = "代码" if "代码" in frame.columns else "symbol"
            change_column = "涨跌幅" if "涨跌幅" in frame.columns else "changepercent"

            if symbol_column not in frame.columns:
                raise RuntimeError(f"Missing symbol column in AkShare {market} data")

            if change_column in frame.columns:
                frame = frame.copy()
                frame["__change"] = frame[change_column].map(_numeric)
                frame = frame.sort_values("__change", ascending=False)

            output: list[str] = []
            for raw in frame[symbol_column].tolist():
                symbol = _to_yfinance_symbol(raw, market)
                if symbol and symbol not in output:
                    output.append(symbol)
                if len(output) >= limit:
                    break

            return output

        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                delay = 4 * (2**attempt) + random.uniform(0, 2)
                time.sleep(delay)

    raise RuntimeError(f"AkShare {market} request failed") from last_error


def get_akshare_us_symbols(limit: int = 50) -> list[str]:
    try:
        return _fetch(ak.stock_us_spot_em, "US", limit)
    except Exception as exc:
        print(f"AkShare US warning: {type(exc).__name__}: {exc}")
        return []


def get_akshare_hk_symbols(limit: int = 50) -> list[str]:
    try:
        return _fetch(ak.stock_hk_spot_em, "HK", limit)
    except Exception as exc:
        print(f"AkShare HK warning: {type(exc).__name__}: {exc}")
        return []
