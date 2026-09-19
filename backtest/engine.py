# backtest/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd

from backtest.costs import calculate_cost


@dataclass
class Trade:
    date: str
    signal: str
    price: float
    quantity: float
    cost: float
    cash_after: float


def run_backtest(
    data: pd.DataFrame,
    signal_function: Callable[[pd.Series], str],
    initial_cash: float = 100_000.0,
    position_fraction: float = 0.95,
) -> Dict[str, object]:
    required = {"Close"}

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    cash = initial_cash
    shares = 0.0
    trades: List[Trade] = []
    equity_curve: List[Dict[str, float]] = []

    for timestamp, row in data.iterrows():
        price = float(row["Close"])
        signal = signal_function(row)

        if signal == "buy" and shares <= 0:
            allocation = cash * position_fraction
            quantity = allocation / price
            cost = calculate_cost(price, quantity)

            cash -= quantity * price + cost
            shares += quantity

            trades.append(
                Trade(
                    date=str(timestamp),
                    signal="buy",
                    price=price,
                    quantity=quantity,
                    cost=cost,
                    cash_after=cash,
                )
            )

        elif signal == "sell" and shares > 0:
            cost = calculate_cost(price, shares)

            cash += shares * price - cost
            shares = 0.0

            trades.append(
                Trade(
                    date=str(timestamp),
                    signal="sell",
                    price=price,
                    quantity=0.0,
                    cost=cost,
                    cash_after=cash,
                )
            )

        equity = cash + shares * price

        equity_curve.append(
            {
                "date": str(timestamp),
                "equity": equity,
                "cash": cash,
                "shares": shares,
            }
        )

    curve = pd.DataFrame(equity_curve)

    total_return = (
        curve["equity"].iloc[-1] / initial_cash
        - 1.0
    )

    running_max = curve["equity"].cummax()
    drawdown = curve["equity"] / running_max - 1.0

    return {
        "initial_cash": initial_cash,
        "final_equity": float(curve["equity"].iloc[-1]),
        "total_return": float(total_return),
        "max_drawdown": float(drawdown.min()),
        "trade_count": len(trades),
        "trades": trades,
        "equity_curve": curve,
    }