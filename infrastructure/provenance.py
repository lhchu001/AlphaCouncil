# infrastructure/provenance.py
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict

from agent.schemas import DataQuality, Evidence, Provenance


def compute_data_hash(data: Dict[str, Any]) -> str:
    normalized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_stale(
    retrieved_at: datetime,
    max_age_seconds: float = 3600.0,
) -> bool:
    """
    Check if data is stale based on retrieval time.
    
    Args:
        retrieved_at: When the data was retrieved
        max_age_seconds: Maximum acceptable age in seconds (default: 1 hour)
    
    Returns:
        True if data is stale, False otherwise
    """
    now = datetime.now(timezone.utc)
    age_seconds = (now - retrieved_at).total_seconds()
    return age_seconds > max_age_seconds


def build_evidence(
    source: str,
    ticker: str,
    facts: Dict[str, Any],
    retrieved_at: datetime,
    market_date: str | None = None,
    period: str | None = None,
    cache_hit: bool = False,
    max_age_seconds: float = 3600.0,
) -> Evidence:
    now = datetime.now(timezone.utc)
    age_seconds = (now - retrieved_at).total_seconds()

    quality = DataQuality(
        stale=age_seconds > max_age_seconds,
        freshness_seconds=age_seconds,
        max_age_seconds=max_age_seconds,
        status="ok" if age_seconds <= max_age_seconds else "degraded",
    )

    return Evidence(
        facts=facts,
        provenance=Provenance(
            source=source,
            retrieved_at=retrieved_at,
            market_date=market_date,
            period=period,
            ticker=ticker,
            cache_hit=cache_hit,
            data_hash=compute_data_hash(facts),
        ),
        quality=quality,
    )