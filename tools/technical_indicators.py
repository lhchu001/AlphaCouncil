# technical_indicators.py
"""
Technical indicators for quantitative analysis.

Dependencies:
    pip install pandas numpy

Usage example:
    import pandas as pd
    import yfinance as yf
    from technical_indicators import (
        sma, ema, macd, macd_signal, macd_histogram,
        rsi, bollinger_bands, atr, vwma
    )

    df = yf.download("AAPL", period="2y", interval="1d")
    df["sma_50"] = sma(df["Close"], 50)
    df["sma_200"] = sma(df["Close"], 200)
    df["ema_10"] = ema(df["Close"], 10)

    macd_line, signal_line, hist = macd(df["Close"])
    df["macd"] = macd_line
    df["macds"] = signal_line
    df["macdh"] = hist

    df["rsi"] = rsi(df["Close"])
    df["boll"], df["boll_ub"], df["boll_lb"] = bollinger_bands(df["Close"])
    df["atr"] = atr(df["High"], df["Low"], df["Close"])
    df["vwma"] = vwma(df["Close"], df["Volume"], 20)
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd


def _ewm_span_to_alpha(span: int) -> float:
    """Convert pandas EWM span to smoothing factor α = 2/(span+1)."""
    return 2.0 / (span + 1.0)


def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (SMA).

    Parameters
    ----------
    series : pd.Series
        Price series (e.g., close).
    window : int
        Lookback window (e.g., 50, 200).

    Returns
    -------
    pd.Series
        SMA values; first `window-1` entries are NaN.

    Notes
    -----
    - Used for close_50_sma, close_200_sma, and bollinger middle (window=20).
    - Lags price; combine with faster indicators (e.g., 10 EMA) for timely signals.
    """
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    Parameters
    ----------
    series : pd.Series
        Price series.
    span : int
        EMA span (e.g., 10, 12, 26).

    Returns
    -------
    pd.Series
        EMA values; first `span-1` entries are NaN.

    Notes
    -----
    - Used for close_10_ema and MACD components.
    - More responsive than SMA but prone to noise in choppy markets.
    """
    return series.ewm(span=span, adjust=False).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD, Signal, and Histogram.

    Parameters
    ----------
    series : pd.Series
        Price series (typically close).
    fast : int
        Fast EMA span (default 12).
    slow : int
        Slow EMA span (default 26).
    signal_span : int
        Signal line EMA span (default 9).

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (macd_line, signal_line, histogram)

    Notes
    -----
    - macd = EMA(fast) - EMA(slow)
    - macds (signal) = EMA(macd, span=signal_span)
    - macdh = macd - macds
    - Use crossovers and divergence for trend-change signals.
    - Confirm with other indicators in low-volatility or sideways markets.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_span)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Parameters
    ----------
    series : pd.Series
        Price series (close).
    window : int
        Lookback window (default 14).

    Returns
    -------
    pd.Series
        RSI values (0–100); first `window` entries are NaN.

    Notes
    -----
    - Flags overbought (>70) and oversold (<30) conditions.
    - Watch for divergence to signal reversals.
    - In strong trends, RSI may remain extreme; cross-check with trend analysis.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    return rsi_values


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: middle, upper, lower.

    Parameters
    ----------
    series : pd.Series
        Price series (close).
    window : int
        SMA window for middle band (default 20).
    num_std : float
        Number of standard deviations for bands (default 2).

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (middle, upper, lower)

    Notes
    -----
    - Middle = 20 SMA (boll)
    - Upper = middle + num_std * stddev (boll_ub)
    - Lower = middle - num_std * stddev (boll_lb)
    - Use bands together to spot breakouts/reversals; prices can ride bands in strong trends.
    """
    middle = sma(series, window)
    rolling_std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return middle, upper, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Average True Range (ATR).

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    window : int
        ATR EMA window (default 14).

    Returns
    -------
    pd.Series
        ATR values; first `window` entries are NaN.

    Notes
    -----
    - True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    - ATR = EMA of True Range
    - Use for stop-loss sizing and position sizing based on volatility.
    - Reactive measure; use within a broader risk management framework.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's smoothing is equivalent to EMA with span=window
    atr_values = true_range.ewm(span=window, adjust=False).mean()
    return atr_values


def vwma(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Volume-Weighted Moving Average (VWMA).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume series.
    window : int
        Lookback window (default 20).

    Returns
    -------
    pd.Series
        VWMA values; first `window-1` entries are NaN.

    Notes
    -----
    - VWMA = rolling sum(close * volume) / rolling sum(volume)
    - Confirms trends by integrating price with volume.
    - Watch for skewed results from volume spikes; combine with other volume analyses.
    """
    pv = close * volume
    vwma_values = (
        pv.rolling(window=window, min_periods=window).sum()
        / volume.rolling(window=window, min_periods=window).sum()
    )
    return vwma_values


# Convenience wrappers matching your exact field names
def close_50_sma(series: pd.Series) -> pd.Series:
    """50-day SMA for medium-term trend and dynamic support/resistance."""
    return sma(series, 50)


def close_200_sma(series: pd.Series) -> pd.Series:
    """200-day SMA for long-term trend and golden/death cross confirmation."""
    return sma(series, 200)


def close_10_ema(series: pd.Series) -> pd.Series:
    """10-day EMA for short-term momentum and quick entry points."""
    return ema(series, 10)


def macd_line(series: pd.Series) -> pd.Series:
    """MACD line (12/26) without signal/histogram."""
    m, _, _ = macd(series)
    return m


def macd_signal(series: pd.Series) -> pd.Series:
    """MACD signal line (9-day EMA of MACD)."""
    _, s, _ = macd(series)
    return s


def macd_histogram(series: pd.Series) -> pd.Series:
    """MACD histogram (MACD - Signal)."""
    _, _, h = macd(series)
    return h


def boll_middle(series: pd.Series) -> pd.Series:
    """Bollinger middle band (20 SMA)."""
    m, _, _ = bollinger_bands(series)
    return m


def boll_upper(series: pd.Series) -> pd.Series:
    """Bollinger upper band (20 SMA + 2σ)."""
    _, u, _ = bollinger_bands(series)
    return u


def boll_lower(series: pd.Series) -> pd.Series:
    """Bollinger lower band (20 SMA - 2σ)."""
    _, _, l = bollinger_bands(series)
    return l


# Example usage block (commented out for import safety)
if __name__ == "__main__":
    # Quick demo with synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), index=dates)
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    volume = pd.Series(np.random.randint(1e6, 5e6, size=n), index=dates)

    df = pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume})

    df["close_50_sma"] = close_50_sma(df["Close"])
    df["close_200_sma"] = close_200_sma(df["Close"])
    df["close_10_ema"] = close_10_ema(df["Close"])

    df["macd"] = macd_line(df["Close"])
    df["macds"] = macd_signal(df["Close"])
    df["macdh"] = macd_histogram(df["Close"])

    df["rsi"] = rsi(df["Close"])

    df["boll"] = boll_middle(df["Close"])
    df["boll_ub"] = boll_upper(df["Close"])
    df["boll_lb"] = boll_lower(df["Close"])

    df["atr"] = atr(df["High"], df["Low"], df["Close"])

    df["vwma"] = vwma(df["Close"], df["Volume"])

    print(df.tail(10))