# AlphaCouncil test steps

## 1. Environment and compilation checks

From the repository root:

```powershell
python -m compileall agent tools council infrastructure evaluation backtest sources
```

Verify imports:

```powershell
python -c "from sources.equity_tracker import find_hot_equity_topics; print('Equity_Tracker import OK')"
python -c "from agent.indicator_agent import build_indicator_agent; print('indicator agent OK')"
python -c "from agent.fundamental_agent import build_fundamental_agent; print('fundamental agent OK')"
python -c "from agent.recommendation_agent import build_recommendation_agent; print('recommendation agent OK')"
python -c "from council.council import run_council; print('council OK')"
```

## 2. Environment-variable validation

Ensure `.env` contains:

```dotenv
ALPHACOUNCIL_START_DATE=2025-09-01
ALPHACOUNCIL_END_DATE=2026-09-01
ALPHACOUNCIL_PRICE_INTERVAL=1d
ALPHACOUNCIL_FUNDAMENTAL_PERIOD=yearly
ALPHACOUNCIL_MAX_STOCKS=10
```

Run a quick check:

```powershell
python -c "from infrastructure.config import load_settings; s = load_settings(); print(s.start_date, s.end_date, s.price_interval, s.fundamental_period, s.max_stocks)"
```

## 3. Equity_Tracker smoke test

```powershell
python -c "from sources.equity_tracker import find_hot_equity_topics; symbols = find_hot_equity_topics(xueqiu_us=True, xueqiu_hk=True, akshare_us=True, akshare_hk=True, stocktwits=True, reddit_us=True, reddit_hk=True, top_n=5); print(symbols)"
```

Confirm that:

- Symbols are returned.
- US symbols are 1–5 uppercase letters.
- Hong Kong symbols are in the form `XXXX.HK`.
- Invalid symbols are filtered out.

## 4. Main application smoke test

Run:

```powershell
python -m main
```

Confirm that:

- The application prints Council provider, Equity source, dates, interval, period, and maximum stocks.
- A list of selected equities is printed.
- For each equity, a recommendation block is printed with:
  - Ticker.
  - Signal.
  - Score.
  - Human-review status.
  - Council status.
  - Risk flags.
  - Final analysis.
- The application does not crash on invalid symbols or per-symbol failures.
- `data/signal_history.jsonl` is created and updated.

## 5. Backtest smoke test

Create a temporary test file outside the package, for example `test_backtest.py`:

```python
import pandas as pd

from backtest.engine import run_backtest


def signal_function(row):
    if row["Close"] <= 100:
        return "buy"
    if row["Close"] >= 110:
        return "sell"
    return "hold"


data = pd.DataFrame(
    {
        "Close": [100, 102, 105, 110, 108],
    },
    index=pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    ),
)

result = run_backtest(
    data=data,
    signal_function=signal_function,
    initial_cash=100_000.0,
    position_fraction=0.95,
)

assert result["final_equity"] >= 0
assert result["trade_count"] >= 0
assert not result["equity_curve"].empty
print(result)
```

Run:

```powershell
python test_backtest.py
```

Confirm that:

- The backtest completes without errors.
- `final_equity` and `trade_count` are non-negative.
- The equity curve is non-empty.

## 6. Walk-forward test (optional)

Create `test_walkforward.py`:

```python
import pandas as pd

from backtest.walk_forward import walk_forward_backtest


def signal_function(row):
    if row["Close"] <= 100:
        return "buy"
    return "hold"


data = pd.DataFrame(
    {
        "Close": list(range(90, 120)),
    },
    index=pd.date_range(
        "2025-01-01",
        periods=30,
        freq="D",
    ),
)

result = walk_forward_backtest(
    data=data,
    signal_function=signal_function,
    train_size=10,
    test_size=5,
)

print(result)
```

Run:

```powershell
python test_walkforward.py
```

Confirm that:

- Multiple windows are returned.
- Each window has correct start and end dates.
- No empty test window is silently accepted.

## 7. Cleanup

Remove temporary test files:

```powershell
Remove-Item test_backtest.py
Remove-Item test_walkforward.py
```

If a development checkpoint database is corrupted or you changed the schema:

```powershell
copy data\checkpoints.sqlite data\checkpoints.sqlite.bak
Remove-Item data\checkpoints.sqlite
```

Re-run the main application to regenerate a clean checkpoint.