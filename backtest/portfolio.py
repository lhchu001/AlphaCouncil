# backtest/portfolio.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    entry_date: str


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, any]] = []

    def buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        date: str,
        cost: float = 0.0,
    ) -> bool:
        total_cost = shares * price + cost

        if total_cost > self.cash:
            return False

        self.cash -= total_cost

        if ticker in self.positions:
            existing = self.positions[ticker]
            total_shares = existing.shares + shares
            avg_price = (
                (existing.shares * existing.entry_price + shares * price)
                / total_shares
            )
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=total_shares,
                entry_price=avg_price,
                entry_date=date,
            )
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                entry_price=price,
                entry_date=date,
            )

        self.trade_history.append(
            {
                "date": date,
                "ticker": ticker,
                "action": "buy",
                "shares": shares,
                "price": price,
                "cost": cost,
            }
        )

        return True

    def sell(
        self,
        ticker: str,
        shares: float,
        price: float,
        date: str,
        cost: float = 0.0,
    ) -> bool:
        if ticker not in self.positions:
            return False

        position = self.positions[ticker]

        if shares > position.shares:
            shares = position.shares

        proceeds = shares * price - cost
        self.cash += proceeds

        position.shares -= shares

        if position.shares <= 0:
            del self.positions[ticker]

        self.trade_history.append(
            {
                "date": date,
                "ticker": ticker,
                "action": "sell",
                "shares": shares,
                "price": price,
                "cost": cost,
            }
        )

        return True

    def equity(self, prices: Dict[str, float]) -> float:
        position_value = sum(
            pos.shares * prices.get(pos.ticker, 0.0)
            for pos in self.positions.values()
        )
        return self.cash + position_value