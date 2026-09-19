# AlphaCouncil

AlphaCouncil is an experimental, multi-agent equity-analysis and recommendation system that combines:

- Automated discovery of hot equities from AkShare, Reddit, StockTwits, and Xueqiu.
- Technical and fundamental specialist agents.
- A Council review layer for disagreement, weak signals, or low-confidence outputs.
- Structured evidence, provenance, and data-quality metadata.
- Configurable LLM providers and models.
- LangGraph orchestration and SQLite checkpointing.
- Risk flags and optional human-review requirements.
- Basic event-driven backtesting and walk-forward testing.

**Inspired by** [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council).

> **Disclaimer:** AlphaCouncil is an educational and research project. It is not investment advice, does not guarantee accuracy, and should not be used as the sole basis for trading or investment decisions.

## Project layout

```text
AlphaCouncil/
├── __init__.py
├── main.py
├── requirements.txt
├── layout.txt
├── workflow.txt
├── .env.example
├── README.md
├── test_steps.md
├── LICENSE
│
├── agent/
│   ├── __init__.py
│   ├── schemas.py
│   ├── indicator_agent.py
│   ├── fundamental_agent.py
│   └── recommendation_agent.py
│
├── tools/
│   ├── __init__.py
│   ├── market_data.py
│   ├── technical_indicators.py
│   ├── technical_indicators_tool.py
│   └── fundamentals_tool.py
│
├── council/
│   ├── __init__.py
│   ├── council.py
│   ├── policy.py
│   └── output_parser.py
│
├── infrastructure/
│   ├── __init__.py
│   ├── config.py
│   ├── cache.py
│   ├── reliability.py
│   ├── provenance.py
│   ├── persistence.py
│   └── history.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── evaluate_specialists.py
│   └── calibrate_confidence.py
│
├── backtest/
│   ├── __init__.py
│   ├── engine.py
│   ├── costs.py
│   ├── portfolio.py
│   ├── walk_forward.py
│   └── benchmarks.py
│
├── discovery/
│   ├── __init__.py
│   ├── equity_tracker.py
│   ├── stocktwits_source.py
│   ├── reddit_source.py
│   ├── xueqiu_source.py
│   └── akshare_source.py
│
└── llm/
    ├── __init__.py
    ├── factory.py
    ├── LoadbyGoogle.py
    ├── LoadbyHFLocalonly.py
    ├── LoadbyHuggingFace.py
    ├── LoadbyLlamaCpp.py
    ├── LoadbyNvidia.py
    ├── LoadbyOllama.py
    └── LoadbyOpenRouter.py
```

## Requirements

- Python 3.13.7 or a compatible Python 3.13 environment.
- LangChain 1.3.1.
- LangGraph 1.2.0.
- CrewAI 1.14.4, if used by optional components.
- AutoGen 0.12.2, if used by optional components.
- Internet access for market and financial data retrieval.
- API credentials for the selected LLM provider.

The exact dependency versions are defined by `requirements.txt`. If that file pins different versions, follow the repository file as the source of truth.

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/<your-account>/AlphaCouncil.git
cd AlphaCouncil
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file from the example:

```powershell
copy .env.example .env
```

At minimum, configure the API key required by the selected provider. A representative configuration is:

```dotenv
NVIDIA_API_KEY=replace_with_your_key

INDICATOR_LLM_PROVIDER=Nvidia_API
INDICATOR_LLM_MODEL=nvidia/nemotron-3-super-120b-a12b

FUNDAMENTAL_LLM_PROVIDER=Nvidia_API
FUNDAMENTAL_LLM_MODEL=nvidia/nemotron-3-super-120b-a12b

COUNCIL_PROVIDER=Nvidia_API
COUNCIL_MODELS=nvidia/nemotron-3-super-120b-a12b,google/gemma-4-31b-it,openai/gpt-oss-120b

LLM_TIMEOUT_SECONDS=90
LLM_MAX_RETRIES=2

TECHNICAL_WEIGHT=0.45
FUNDAMENTAL_WEIGHT=0.55
LOW_CONFIDENCE_THRESHOLD=0.60
WEAK_SIGNAL_THRESHOLD=0.35

DATA_CACHE_TTL_SECONDS=900
DATA_MAX_AGE_SECONDS=3600
LANGGRAPH_CHECKPOINT_DB=data/checkpoints.sqlite

# Equity analysis parameters
ALPHACOUNCIL_START_DATE=2025-09-01
ALPHACOUNCIL_END_DATE=2026-09-01
ALPHACOUNCIL_PRICE_INTERVAL=1d
ALPHACOUNCIL_FUNDAMENTAL_PERIOD=yearly
ALPHACOUNCIL_MAX_STOCKS=10
```

Use the environment-variable names supported by the checked-in `infrastructure/config.py`. Never commit `.env` or API keys.

### 5. Create runtime directories

```powershell
New-Item -ItemType Directory -Force data
New-Item -ItemType Directory -Force logs
```

### 6. Start the application

```powershell
python -m main
```

The application will:

- Load hot equities from AkShare, Reddit, StockTwits, and Xueqiu.
- Validate US and Hong Kong symbols.
- Evaluate each equity using technical and fundamental specialists.
- Combine signals and invoke the Council when required.
- Print recommendations and append results to `data/signal_history.jsonl`.

## Workflow

1. `main.py` loads environment variables and application settings.
2. Equity_Tracker queries multiple sources for hot equities.
3. Symbols are validated and deduplicated.
4. For each symbol:
   - The technical specialist retrieves price data and computes indicators.
   - Technical evidence is created with provenance and quality metadata.
   - The technical LLM interprets the supplied evidence.
   - The fundamental specialist retrieves company and statement data.
   - Fundamental evidence is created and interpreted.
   - The decision combiner calculates the weighted score.
   - Risk flags are added for disagreement, weak signals, low confidence, missing evidence, or stale data.
   - The Council is invoked when the result needs additional review.
   - The final result is printed and recorded in signal history.
5. LangGraph checkpoints preserve graph state when checkpointing is enabled.

A Council result should be treated as provisional when only one member returns valid output. A full consensus requires multiple valid responses.

## Validation before running

Run the following checks from the repository root:

```powershell
python -m compileall agent tools council infrastructure evaluation backtest sources
```

Check imports:

```powershell
python -c "from discovery.equity_tracker import find_hot_equity_topics; print('Equity_Tracker import OK')"
python -c "from agent.indicator_agent import build_indicator_agent; print('indicator agent OK')"
python -c "from agent.fundamental_agent import build_fundamental_agent; print('fundamental agent OK')"
python -c "from agent.recommendation_agent import build_recommendation_agent; print('recommendation agent OK')"
```

If an old SQLite checkpoint contains serialized Pydantic objects and you are working in development, remove it before a clean run:

```powershell
Remove-Item .\data\checkpoints.sqlite -ErrorAction SilentlyContinue
```

Do not delete a production checkpoint without first backing it up.

## Testing procedure

### Unit and import tests

At minimum, verify:

- All modules compile.
- The technical, fundamental, recommendation, and Council graphs import successfully.
- Required environment variables are loaded.
- API keys are not printed in logs.
- Invalid or missing model output fails safely to `hold`.
- Missing evidence is marked unavailable or degraded.
- Signals are restricted to `buy`, `hold`, and `sell`.
- Confidence values are clamped to the interval `[0, 1]`.

If a test suite is added, run:

```powershell
pytest -q
```

### Interactive smoke test

Run:

```powershell
python -m main
```

Confirm that the output contains:

- Council provider.
- Equity source.
- Start date, end date, price interval, fundamental period, maximum stocks.
- Selected equities.
- For each equity:
  - Ticker.
  - Signal.
  - Score.
  - Human-review status.
  - Council status.
  - Risk flags.
  - Final analysis.

Inspect any Council debug output. A model that returns prose instead of JSON should be recorded as a failed or provisional member rather than crashing the application.

## Backtesting

The backtest engine is separate from the interactive LLM workflow. This separation is important: historical performance should be evaluated using reproducible signals and historical data, not by allowing a current-date LLM to see future information.

### Backtest principles

- Use historical data only within each simulation window.
- Avoid look-ahead bias.
- Include transaction costs and slippage assumptions.
- Do not use future fundamental statements before their publication date.
- Keep training and test periods separate.
- Compare results with a buy-and-hold benchmark.
- Record the exact data range, parameters, strategy version, and costs.
- Treat a backtest as research evidence, not a guarantee of future performance.

### Backtest engine contract

The basic engine expects a pandas DataFrame containing a `Close` column and a signal function:

```python
signal = signal_function(row)
```

The signal function should return one of:

```text
buy
sell
hold
```

A typical call is:

```python
from backtest.engine import run_backtest

result = run_backtest(
    data=data,
    signal_function=my_signal_function,
    initial_cash=100_000.0,
    position_fraction=0.95,
)
```

The result should include:

- Initial cash.
- Final equity.
- Total return.
- Maximum drawdown.
- Trade count.
- Trade records.
- Equity curve.

### Example deterministic signal

```python
def moving_average_signal(row):
    close = row["Close"]
    sma = row.get("SMA_50")

    if sma is None:
        return "hold"

    if close > sma:
        return "buy"

    return "sell"
```

In a real backtest, calculate the moving average using data available up to the current row only. Avoid calculating a feature with the complete dataset and then using it as if it had been known historically.

### Backtest smoke test

Create a small synthetic DataFrame to test order handling and accounting:

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

Save this as a temporary test file outside the package or convert it into a proper pytest test.

### Buy-and-hold comparison

Use the benchmark utility:

```python
from backtest.benchmarks import buy_and_hold

buy_hold_return = buy_and_hold(data)
print("Strategy return:", result["total_return"])
print("Buy-and-hold return:", buy_hold_return)
```

The function name may differ if your checked-in module uses a different naming convention. Confirm the import in `backtest/benchmarks.py`.

### Walk-forward testing

Walk-forward testing repeatedly trains or calibrates on an earlier window and evaluates on a later unseen window:

```python
from backtest.walk_forward import walk_forward_backtest

result = walk_forward_backtest(
    data=data,
    signal_function=my_signal_function,
    train_size=252,
    test_size=63,
)
```

For every window, verify:

- The training window precedes the test window.
- The test period is not used to calculate training parameters.
- The returned dates are correct.
- The number of windows is as expected.
- No empty test window is silently accepted.

Aggregate the individual test windows only after inspecting them separately. Report average return, dispersion, drawdown, trade count, and the proportion of profitable windows.

### Backtest acceptance checklist

Before trusting a result, check:

- Data has a monotonic datetime index.
- Duplicate timestamps are removed or handled explicitly.
- Missing `Close` values are handled.
- Indicators do not use future rows.
- Orders occur at a clearly defined price and time.
- Costs are applied to every trade.
- Position sizing is bounded.
- Cash cannot become negative unintentionally.
- Open positions are valued at the final available close.
- Drawdown is calculated from the equity curve.
- Strategy returns are compared with buy-and-hold.
- Results are tested over more than one period and ticker.
- Parameters are not selected solely from the test period.

## LLM output and reliability

LLM providers do not always return the same response format. Some return plain JSON, while others return Markdown, explanatory text, content blocks, or truncated output.

AlphaCouncil should therefore:

- Prefer provider-supported structured output when available.
- Keep prompts concise and request a small response.
- Normalize message content across providers.
- Extract JSON from fenced or embedded responses.
- Detect truncated responses.
- Retry transient failures.
- Mark invalid responses as unavailable or abstain.
- Never treat malformed output as a validated recommendation.

A large context window can reduce truncation, but it does not guarantee valid JSON. Parser validation and safe fallback behavior remain necessary.

## Checkpointing

LangGraph checkpointing can preserve state between runs. During development, a checkpoint database may contain serialized Pydantic objects such as `Evidence`, `SpecialistInterpretation`, and `Decision`.

If warnings appear after a schema change, back up and remove the development checkpoint:

```powershell
copy data\checkpoints.sqlite data\checkpoints.sqlite.bak
Remove-Item data\checkpoints.sqlite
```

For long-term compatibility, prefer checkpoint state made only from JSON-compatible primitives:

- strings.
- numbers.
- booleans.
- lists.
- dictionaries.
- null values.

Convert dictionaries to Pydantic models inside nodes when validation or methods such as `model_dump()` are required.

## Security

- Do not commit `.env` files.
- Do not commit API keys, tokens, or private data.
- Add `.env`, `data/`, logs, caches, and local databases to `.gitignore`.
- Review prompts before sending company or portfolio data to an external provider.
- Treat model output as untrusted text.
- Validate signals, confidence, and report fields before using them.

Recommended `.gitignore` entries:

```gitignore
.env
.env.*
!.env.example
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
data/*.sqlite
data/*.db
logs/
```

## Limitations

- Market and financial data may be delayed, incomplete, revised, or unavailable.
- LLM interpretations can be inconsistent across providers and runs.
- Technical and fundamental signals can conflict.
- Historical backtests may contain data, survivorship, execution, and look-ahead biases.
- Simple transaction-cost models may not represent actual trading costs.
- A larger context window can increase cost and latency.
- A Council consensus is not evidence that a recommendation is correct.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Add or update tests.
4. Run compilation, import, and backtest checks.
5. Keep secrets and local databases out of commits.
6. Submit a pull request describing the design and testing performed.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.