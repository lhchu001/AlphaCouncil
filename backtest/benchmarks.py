# backtest/benchmarks.py
from __future__ import annotations

import pandas as pd


def buy_and_hold(data: pd.DataFrame) -> float:
    close = data["Close"]
    return close.iloc[-1] / close.iloc[0] - 1.0


def random_entry_exit(
    data: pd.DataFrame,
    entry_prob: float = 0.1,
    exit_prob: float = 0.1,
) -> float:
    import numpy as np

    np.random.seed(42)

    in_position = False
    entry_price = 0.0
    returns = []

    for timestamp, row in data.iterrows():
        price = float(row["Close"])

        if not in_position and np.random.random() < entry_prob:
            in_position = True
            entry_price = price

        elif in_position and np.random.random() < exit_prob:
            in_position = False
            returns.append(price / entry_price - 1.0)

    if in_position:
        returns.append(data["Close"].iloc[-1] / entry_price - 1.0)

    return sum(returns) / len(returns) if returns else 0.0