# main.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from langchain_core.messages import HumanMessage

from agent.recommendation_agent import (
    build_recommendation_agent,
)
from infrastructure.config import load_settings
from infrastructure.history import SignalHistory
from infrastructure.persistence import checkpoint_saver
from llm import load_config
from discovery.equity_tracker import (
    find_hot_equity_topics,
)


DEFAULT_PRICE_INTERVAL = "1d"
DEFAULT_FUNDAMENTAL_PERIOD = "yearly"
DEFAULT_MAX_STOCKS = 10


def _env_text(
    name: str,
    default: str,
) -> str:
    import os

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


def _env_int(
    name: str,
    default: int,
    minimum: int = 1,
) -> int:
    import os

    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw)
    except ValueError:
        print(
            f"Invalid {name}={raw!r}; "
            f"using {default}."
        )
        return default

    if value < minimum:
        print(
            f"Invalid {name}={raw!r}; "
            f"using {default}."
        )
        return default

    return value


def _valid_symbol(symbol: Any) -> bool:
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


def _analysis_dates() -> tuple[str, str]:
    import os

    yesterday = date.today() - timedelta(days=1)
    default_start = yesterday - timedelta(days=365)

    start = os.getenv(
        "ALPHACOUNCIL_START_DATE",
        default_start.isoformat(),
    ).strip()
    end = os.getenv(
        "ALPHACOUNCIL_END_DATE",
        yesterday.isoformat(),
    ).strip()

    return start or default_start.isoformat(), end or yesterday.isoformat()


def _get_symbols(max_stocks: int) -> list[str]:
    try:
        raw_symbols = find_hot_equity_topics(
            xueqiu_us=False,
            xueqiu_hk=False,
            akshare_us=False,
            akshare_hk=False,
            stocktwits=True,
            reddit_us=False,
            reddit_hk=False,
            top_n=max_stocks,
        )
    except Exception as exc:
        print(
            "Equity_Tracker failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return []

    symbols: list[str] = []

    for raw_symbol in raw_symbols or []:
        symbol = str(raw_symbol).strip().upper()

        if not _valid_symbol(symbol):
            print(
                "Skipping invalid Equity_Tracker symbol: "
                f"{raw_symbol!r}"
            )
            continue

        if symbol not in symbols:
            symbols.append(symbol)

        if len(symbols) >= max_stocks:
            break

    return symbols


def _run_one(
    app: Any,
    history: SignalHistory,
    symbol: str,
    start: str,
    end: str,
    interval: str,
    period: str,
) -> None:
    if not _valid_symbol(symbol):
        print(f"Skipping invalid symbol: {symbol!r}")
        return

    query = (
        f"Analyze {symbol} using technical indicators and "
        "fundamentals, then provide a buy, hold, or sell "
        "suggestion."
    )

    state = {
        "messages": [HumanMessage(content=query)],
        "ticker": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "period": period,
        "query": query,
    }

    config = {
        "configurable": {
            "thread_id": (
                f"{symbol}:{start}:{end}:"
                f"{interval}:{period}"
            )
        }
    }

    try:
        result = app.invoke(state, config=config)
    except Exception as exc:
        print(
            f"Evaluation failed for {symbol}: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    decision = result.get("decision")

    if decision is None:
        print(
            f"No decision returned for {symbol}."
        )
        return

    print("\n=== Recommendation ===")
    print(f"Ticker: {symbol}")
    print(f"Signal: {decision.signal}")
    print(f"Score: {decision.score:.2f}")
    print(
        "Human review required: "
        f"{decision.human_review_required}"
    )
    print(f"Council used: {decision.council_used}")
    print(f"Risk flags: {decision.risk_flags}")
    print("\n--- Final Analysis ---")
    print(result.get("final_analysis", ""))

    history.append(
        ticker=symbol,
        request={
            "start": start,
            "end": end,
            "interval": interval,
            "period": period,
        },
        result=decision.model_dump(),
    )


def main() -> None:
    load_config()
    settings = load_settings()
    history = SignalHistory()

    start, end = _analysis_dates()
    interval = _env_text(
        "ALPHACOUNCIL_PRICE_INTERVAL",
        DEFAULT_PRICE_INTERVAL,
    )
    period = _env_text(
        "ALPHACOUNCIL_FUNDAMENTAL_PERIOD",
        DEFAULT_FUNDAMENTAL_PERIOD,
    ).lower()
    max_stocks = _env_int(
        "ALPHACOUNCIL_MAX_STOCKS",
        DEFAULT_MAX_STOCKS,
    )

    if period not in {"yearly", "quarterly"}:
        print(
            f"Invalid fundamental period {period!r}; "
            f"using {DEFAULT_FUNDAMENTAL_PERIOD}."
        )
        period = DEFAULT_FUNDAMENTAL_PERIOD

    print(f"Council provider: {settings.council_provider}")
    print("Equity source: Equity_Tracker")
    print(f"Start date: {start}")
    print(f"End date: {end}")
    print(f"Price interval: {interval}")
    print(f"Fundamental period: {period}")
    print(f"Maximum stocks: {max_stocks}")
    print("\nLoading hot equities...\n")

    symbols = _get_symbols(max_stocks)

    if not symbols:
        print("No valid equities were returned.")
        return

    print(
        "Selected equities: "
        + ", ".join(symbols)
    )

    with checkpoint_saver(
        settings.checkpoint_db
    ) as checkpointer:
        app = build_recommendation_agent(
            checkpointer=checkpointer
        )

        for symbol in symbols:
            _run_one(
                app=app,
                history=history,
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
                period=period,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting. Bye.")