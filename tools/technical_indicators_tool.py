# tools/technical_indicators_tool.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langchain_core.tools import BaseTool
import yfinance as yf
import pandas as pd

from .technical_indicators import (
    close_50_sma,
    close_200_sma,
    close_10_ema,
    macd_line,
    macd_signal,
    macd_histogram,
    rsi,
    boll_middle,
    boll_upper,
    boll_lower,
    atr,
    vwma,
)


IndicatorName = Literal[
    "close_50_sma",
    "close_200_sma",
    "close_10_ema",
    "macd",
    "macds",
    "macdh",
    "rsi",
    "boll",
    "boll_ub",
    "boll_lb",
    "atr",
    "vwma",
]


def _fetch_ohlc(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Index is DatetimeIndex.
    """
    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
    )
    # Handle multi-level columns for newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    # Ensure expected columns exist
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns from yfinance: {missing}")
    return data[required]


def _compute_indicator_series(
    df: pd.DataFrame,
    name: IndicatorName,
) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    if name == "close_50_sma":
        return close_50_sma(close)
    if name == "close_200_sma":
        return close_200_sma(close)
    if name == "close_10_ema":
        return close_10_ema(close)
    if name == "macd":
        return macd_line(close)
    if name == "macds":
        return macd_signal(close)
    if name == "macdh":
        return macd_histogram(close)
    if name == "rsi":
        return rsi(close)
    if name == "boll":
        return boll_middle(close)
    if name == "boll_ub":
        return boll_upper(close)
    if name == "boll_lb":
        return boll_lower(close)
    if name == "atr":
        return atr(high, low, close)
    if name == "vwma":
        return vwma(close, volume, window=20)

    raise ValueError(f"Unknown indicator: {name}")


class TechnicalIndicatorsTool(BaseTool):
    """
    Tool to compute technical indicators from yfinance data.

    Usage in an agent:
        - Provide ticker, start, end, and a list of indicators.
        - Returns latest values and a small context summary for the LLM.
    """

    name: str = "technical_indicators"
    description: str = (
        "Compute technical indicators (SMA, EMA, MACD, RSI, Bollinger, ATR, VWMA) "
        "for a given ticker and date range using Yahoo Finance data. "
        "Use this when the user asks for trend, momentum, volatility, or volume-weighted analysis."
    )

    def _run(
        self,
        ticker: str,
        start: str,
        end: str,
        indicators: List[IndicatorName],
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """
        Compute requested indicators.

        Parameters
        ----------
        ticker : str
            Stock ticker, e.g. "AAPL".
        start : str
            Start date, YYYY-MM-DD.
        end : str
            End date, YYYY-MM-DD.
        indicators : List[IndicatorName]
            List of indicator names to compute.
        interval : str
            Data interval, e.g. "1d", "1h".

        Returns
        -------
        Dict[str, Any]
            {
              "ticker": ...,
              "start": ...,
              "end": ...,
              "indicators": [
                {
                  "name": "rsi",
                  "latest": float,
                  "latest_date": str,
                  "summary": "RSI 14-day latest value and regime (overbought/oversold/neutral).",
                },
                ...
              ],
              "data_points": int,
            }
        """
        df = _fetch_ohlc(ticker, start, end, interval)

        if df.empty:
            return {
                "ticker": ticker,
                "start": start,
                "end": end,
                "error": "No data returned from yfinance for this range.",
                "indicators": [],
                "data_points": 0,
            }

        results = []
        for ind in indicators:
            series = _compute_indicator_series(df, ind)
            # Drop NaNs at the beginning due to warmup
            series_clean = series.dropna()
            if series_clean.empty:
                latest_val = None
                latest_date = None
            else:
                latest_val = float(series_clean.iloc[-1])
                latest_date = str(series_clean.index[-1].date())

            summary = _indicator_summary(ind, latest_val, df, series)

            results.append(
                {
                    "name": ind,
                    "latest": latest_val,
                    "latest_date": latest_date,
                    "summary": summary,
                }
            )

        return {
            "ticker": ticker,
            "start": start,
            "end": end,
            "indicators": results,
            "data_points": len(df),
        }

    async def _arun(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        # For simplicity, run sync version; in production you can wrap with asyncio.to_thread
        return self._run(*args, **kwargs)


def _indicator_summary(
    name: str,
    latest: Optional[float],
    df: pd.DataFrame,
    series: pd.Series,
) -> str:
    """
    Generate a short, LLM-friendly summary for each indicator.
    This is where you encode the 'tips' from your original spec.
    """
    if latest is None:
        return "Insufficient data to compute."

    close = df["Close"].iloc[-1]

    if name == "close_50_sma":
        position = "above" if close > latest else "below"
        return (
            f"50-day SMA at {latest:.2f}; price is {position} the 50 SMA. "
            "Use as medium-term trend filter and dynamic support/resistance; "
            "combine with faster indicators for timing."
        )

    if name == "close_200_sma":
        position = "above" if close > latest else "below"
        return (
            f"200-day SMA at {latest:.2f}; price is {position} the 200 SMA. "
            "Use for long-term trend confirmation and golden/death cross analysis; "
            "not ideal for frequent entry/exit timing on its own."
        )

    if name == "close_10_ema":
        position = "above" if close > latest else "below"
        return (
            f"10-day EMA at {latest:.2f}; price is {position} the 10 EMA. "
            "Short-term momentum proxy; sensitive to noise, so use with longer averages for filtering."
        )

    if name == "macd":
        return (
            f"MACD line at {latest:.4f}. "
            "Look for crossovers with signal line and divergence vs price for trend-change signals; "
            "confirm in low-volatility or sideways markets."
        )

    if name == "macds":
        return (
            f"MACD signal line at {latest:.4f}. "
            "Use crossovers with MACD line as trade triggers within a broader strategy to reduce false positives."
        )

    if name == "macdh":
        return (
            f"MACD histogram at {latest:.4f}. "
            "Visualizes momentum strength and divergence; can be volatile, so use with additional filters."
        )

    if name == "rsi":
        if latest > 70:
            regime = "overbought"
        elif latest < 30:
            regime = "oversold"
        else:
            regime = "neutral"
        return (
            f"RSI(14) at {latest:.2f} ({regime}). "
            "Use 70/30 thresholds and divergence for reversals; in strong trends RSI may stay extreme, "
            "so cross-check with trend indicators."
        )

    if name == "boll":
        return (
            f"Bollinger middle (20 SMA) at {latest:.2f}. "
            "Acts as dynamic benchmark; combine with upper/lower bands for breakouts and reversals."
        )

    if name == "boll_ub":
        return (
            f"Bollinger upper band at {latest:.2f}. "
            "Potential overbought/breakout zone; confirm with other tools as price can ride the band in strong trends."
        )

    if name == "boll_lb":
        return (
            f"Bollinger lower band at {latest:.2f}. "
            "Potential oversold zone; use additional analysis to avoid false reversal signals."
        )

    if name == "atr":
        return (
            f"ATR(14) at {latest:.2f}. "
            "Use for stop-loss placement and position sizing based on current volatility; "
            "it is reactive, so embed in a broader risk framework."
        )

    if name == "vwma":
        vwma_val = latest
        price_vs_vwma = "above" if close > vwma_val else "below"
        return (
            f"20-day VWMA at {vwma_val:.2f}; price is {price_vs_vwma} VWMA. "
            "Confirms trends by integrating price and volume; watch for distortion from volume spikes."
        )

    return f"{name} latest value: {latest}."