# backtest/walk_forward.py
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from backtest.engine import run_backtest


def walk_forward_backtest(
    data: pd.DataFrame,
    signal_function: Callable[[pd.Series], str],
    train_size: int,
    test_size: int,
) -> Dict[str, object]:
    results = []

    start = 0

    while start + train_size + test_size <= len(data):
        train_end = start + train_size
        test_end = train_end + test_size

        train = data.iloc[start:train_end]
        test = data.iloc[train_end:test_end]

        result = run_backtest(
            data=test,
            signal_function=signal_function,
        )

        result["train_start"] = str(train.index[0])
        result["train_end"] = str(train.index[-1])
        result["test_start"] = str(test.index[0])
        result["test_end"] = str(test.index[-1])

        results.append(result)
        start += test_size

    return {
        "windows": results,
        "window_count": len(results),
    }