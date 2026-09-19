# backtest/costs.py
from __future__ import annotations


def calculate_cost(
    price: float,
    quantity: float,
    commission_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
) -> float:
    notional = abs(price * quantity)

    return notional * (
        commission_rate + slippage_rate
    )