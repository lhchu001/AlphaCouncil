# infrastructure/config.py
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    indicator_provider: str
    indicator_model: str
    fundamental_provider: str
    fundamental_model: str
    council_provider: str
    council_models: tuple[str, ...]
    chairman_model: str
    llm_timeout: float
    llm_retries: int
    cache_ttl: int
    technical_weight: float
    fundamental_weight: float
    low_confidence_threshold: float
    weak_signal_threshold: float
    checkpoint_db: str
    max_age_seconds: float
    start_date: str
    end_date: str
    price_interval: str
    fundamental_period: str
    max_stocks: int


def _positive_int(
    name: str,
    default: int,
) -> int:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return value if value > 0 else default


def load_settings() -> Settings:
    council_models = tuple(
        item.strip()
        for item in os.getenv(
            "COUNCIL_MODELS",
            "",
        ).split(",")
        if item.strip()
    )

    return Settings(
        indicator_provider=os.getenv(
            "INDICATOR_LLM_PROVIDER",
            "Nvidia_API",
        ),
        indicator_model=os.getenv(
            "INDICATOR_LLM_MODEL",
            "nvidia/nemotron-3-super-120b-a12b",
        ),
        fundamental_provider=os.getenv(
            "FUNDAMENTAL_LLM_PROVIDER",
            "Nvidia_API",
        ),
        fundamental_model=os.getenv(
            "FUNDAMENTAL_LLM_MODEL",
            "nvidia/nemotron-3-super-120b-a12b",
        ),
        council_provider=os.getenv(
            "COUNCIL_PROVIDER",
            "Nvidia_API",
        ),
        council_models=council_models,
        chairman_model=os.getenv(
            "COUNCIL_CHAIRMAN_MODEL",
            council_models[0]
            if council_models
            else "",
        ),
        llm_timeout=float(
            os.getenv("LLM_TIMEOUT_SECONDS", "90")
        ),
        llm_retries=int(
            os.getenv("LLM_MAX_RETRIES", "2")
        ),
        cache_ttl=int(
            os.getenv(
                "DATA_CACHE_TTL_SECONDS",
                "900",
            )
        ),
        technical_weight=float(
            os.getenv("TECHNICAL_WEIGHT", "0.45")
        ),
        fundamental_weight=float(
            os.getenv("FUNDAMENTAL_WEIGHT", "0.55")
        ),
        low_confidence_threshold=float(
            os.getenv(
                "LOW_CONFIDENCE_THRESHOLD",
                "0.60",
            )
        ),
        weak_signal_threshold=float(
            os.getenv(
                "WEAK_SIGNAL_THRESHOLD",
                "0.35",
            )
        ),
        checkpoint_db=os.getenv(
            "LANGGRAPH_CHECKPOINT_DB",
            "data/checkpoints.sqlite",
        ),
        max_age_seconds=float(
            os.getenv(
                "DATA_MAX_AGE_SECONDS",
                "3600",
            )
        ),
        start_date=os.getenv(
            "ALPHACOUNCIL_START_DATE",
            "",
        ).strip(),
        end_date=os.getenv(
            "ALPHACOUNCIL_END_DATE",
            "",
        ).strip(),
        price_interval=os.getenv(
            "ALPHACOUNCIL_PRICE_INTERVAL",
            "1d",
        ).strip()
        or "1d",
        fundamental_period=os.getenv(
            "ALPHACOUNCIL_FUNDAMENTAL_PERIOD",
            "yearly",
        ).strip().lower()
        or "yearly",
        max_stocks=_positive_int(
            "ALPHACOUNCIL_MAX_STOCKS",
            10,
        ),
    )