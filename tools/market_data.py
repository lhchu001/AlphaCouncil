# tools/market_data.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import yfinance as yf

from infrastructure.cache import JsonCache
from infrastructure.provenance import build_evidence


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    cache: JsonCache,
    max_age_seconds: float = 3600.0,
) -> Dict[str, Any]:
    """
    Retrieve OHLCV data with cache and provenance metadata.
    """
    cache_key = (
        f"ohlcv:{ticker}:{start}:{end}:{interval}"
    )

    cached = cache.get(cache_key)

    if cached is not None:
        return {
            "data": cached,
            "evidence": build_evidence(
                source="yfinance",
                ticker=ticker,
                facts=cached,
                retrieved_at=datetime.now(timezone.utc),
                market_date=end,
                period=interval,
                cache_hit=True,
                max_age_seconds=max_age_seconds,
            ),
        }

    data_frame = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval=interval,
    )

    if data_frame.empty:
        raise ValueError(
            f"No OHLCV data returned for {ticker}."
        )

    data = data_frame.reset_index().to_dict(
        orient="list"
    )

    cache.set(cache_key, data)

    return {
        "data": data,
        "evidence": build_evidence(
            source="yfinance",
            ticker=ticker,
            facts=data,
            retrieved_at=datetime.now(timezone.utc),
            market_date=end,
            period=interval,
            cache_hit=False,
            max_age_seconds=max_age_seconds,
        ),
    }