# tools/fundamentals_tool.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, TypedDict

import pandas as pd
import yfinance as yf
from langchain_core.tools import BaseTool


StatementName = Literal[
    "fundamentals",
    "balance_sheet",
    "cash_flow",
    "income_statement",
    "all",
]


class FundamentalsToolInput(TypedDict, total=False):
    ticker: str
    statement: StatementName
    period: Literal["yearly", "quarterly"]


def _get_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


def _get_fundamentals_info(tk: yf.Ticker) -> Dict[str, Any]:
    info = tk.info or {}
    keys = [
        "marketCap",
        "enterpriseValue",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "pegRatio",
        "profitMargins",
        "operatingMargins",
        "grossMargins",
        "returnOnAssets",
        "returnOnEquity",
        "debtToEquity",
        "dividendYield",
        "beta",
        "sector",
        "industry",
        "longName",
        "currency",
    ]
    return {k: info.get(k) for k in keys}


def _latest_period_label(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    return str(df.columns[0])


def _safe_get_row(df: pd.DataFrame, row: str, col: Optional[Any] = None) -> Optional[float]:
    if df is None or df.empty or row not in df.index:
        return None
    s = df.loc[row]
    if isinstance(s, pd.Series):
        if col is None:
            val = s.iloc[0]
        elif col in s.index:
            val = s[col]
        else:
            return None
    else:
        val = s
    if pd.isna(val):
        return None
    try:
        return float(val)
    except Exception:
        return None


def analyze_fundamentals_info(info: Dict[str, Any]) -> Dict[str, Any]:
    market_cap = info.get("marketCap")
    pe = info.get("trailingPE")
    fpe = info.get("forwardPE")
    pb = info.get("priceToBook")
    margins = info.get("profitMargins")
    roe = info.get("returnOnEquity")
    beta = info.get("beta")

    return {
        "size": (
            "mega-cap" if market_cap and market_cap >= 200e9 else
            "large-cap" if market_cap and market_cap >= 10e9 else
            "mid-cap" if market_cap and market_cap >= 2e9 else
            "small-cap" if market_cap and market_cap >= 300e6 else
            "unknown"
        ),
        "valuation": (
            "attractive" if sum(x is not None and x < y for x, y in [(pe, 15), (fpe, 15), (pb, 3)]) >= 2 else
            "expensive" if sum(x is not None and x > y for x, y in [(pe, 30), (fpe, 30), (pb, 8)]) >= 2 else
            "mixed"
        ),
        "profitability": (
            "strong" if sum(x is not None and x >= y for x, y in [(margins, 0.15), (roe, 0.20)]) >= 1 else
            "average" if margins is not None or roe is not None else
            "unknown"
        ),
        "risk": (
            "defensive" if beta is not None and beta < 0.9 else
            "market-like" if beta is not None and beta <= 1.2 else
            "high-volatility" if beta is not None else
            "unknown"
        ),
    }


def analyze_balance_sheet_df(bs: pd.DataFrame) -> Dict[str, Any]:
    if bs is None or bs.empty:
        return {"error": "No balance sheet data available"}

    latest = bs.columns[0]
    current_assets = _safe_get_row(bs, "Current Assets", latest)
    current_liabilities = _safe_get_row(bs, "Current Liabilities", latest)
    total_assets = _safe_get_row(bs, "Total Assets", latest)
    total_liabilities = _safe_get_row(bs, "Total Liabilities Net Minority Interest", latest)
    cash = _safe_get_row(bs, "Cash And Cash Equivalents", latest)
    debt = _safe_get_row(bs, "Total Debt", latest)
    equity = _safe_get_row(bs, "Stockholders Equity", latest)

    current_ratio = (current_assets / current_liabilities) if current_assets and current_liabilities else None
    debt_to_equity = (debt / equity) if debt and equity else None
    liabilities_to_assets = (total_liabilities / total_assets) if total_liabilities and total_assets else None
    cash_ratio = (cash / current_liabilities) if cash and current_liabilities else None

    return {
        "latest_period": str(latest),
        "current_ratio": current_ratio,
        "debt_to_equity": debt_to_equity,
        "liabilities_to_assets": liabilities_to_assets,
        "cash_ratio": cash_ratio,
        "summary": (
            "liquid" if current_ratio and current_ratio >= 1.5 else
            "adequate liquidity" if current_ratio and current_ratio >= 1.0 else
            "tight liquidity" if current_ratio is not None else
            "insufficient data"
        ),
    }


def analyze_cash_flow_df(cf: pd.DataFrame) -> Dict[str, Any]:
    if cf is None or cf.empty:
        return {"error": "No cash flow data available"}

    latest = cf.columns[0]
    ocf = _safe_get_row(cf, "Operating Cash Flow", latest)
    capex = _safe_get_row(cf, "Capital Expenditure", latest)
    fcf = (ocf + capex) if ocf is not None and capex is not None else None
    sbc = _safe_get_row(cf, "Stock Based Compensation", latest)
    dividends = _safe_get_row(cf, "Cash Dividends Paid", latest)

    return {
        "latest_period": str(latest),
        "operating_cash_flow": ocf,
        "capex": capex,
        "free_cash_flow": fcf,
        "stock_based_compensation": sbc,
        "dividends_paid": dividends,
        "summary": (
            "positive free cash flow" if fcf is not None and fcf > 0 else
            "negative free cash flow" if fcf is not None and fcf < 0 else
            "mixed cash generation" if ocf is not None else
            "insufficient data"
        ),
    }


def analyze_income_statement_df(inc: pd.DataFrame) -> Dict[str, Any]:
    if inc is None or inc.empty:
        return {"error": "No income statement data available"}

    latest = inc.columns[0]
    prev = inc.columns[1] if len(inc.columns) > 1 else None

    revenue = _safe_get_row(inc, "Total Revenue", latest)
    prev_revenue = _safe_get_row(inc, "Total Revenue", prev) if prev is not None else None
    gross_profit = _safe_get_row(inc, "Gross Profit", latest)
    operating_income = _safe_get_row(inc, "Operating Income", latest)
    net_income = _safe_get_row(inc, "Net Income", latest)

    revenue_growth = ((revenue - prev_revenue) / abs(prev_revenue)) if revenue is not None and prev_revenue not in (None, 0) else None
    gross_margin = (gross_profit / revenue) if gross_profit is not None and revenue else None
    operating_margin = (operating_income / revenue) if operating_income is not None and revenue else None
    net_margin = (net_income / revenue) if net_income is not None and revenue else None

    return {
        "latest_period": str(latest),
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "summary": (
            "growing revenue" if revenue_growth and revenue_growth > 0 else
            "declining revenue" if revenue_growth is not None else
            "insufficient data"
        ),
    }


class FundamentalsTool(BaseTool):
    name: str = "fundamentals"
    description: str = (
        "Fetch and analyze fundamentals, balance sheet, cash flow, and income statement data "
        "from Yahoo Finance for a given ticker. Use this when the user asks about valuation, "
        "liquidity, leverage, cash generation, revenue growth, or profitability."
    )

    def _run(
        self,
        ticker: str,
        statement: StatementName = "all",
        period: Literal["yearly", "quarterly"] = "yearly",
    ) -> Dict[str, Any]:
        tk = _get_ticker(ticker)

        info = _get_fundamentals_info(tk)
        bs = _safe_df(tk.balance_sheet if period == "yearly" else tk.quarterly_balance_sheet)
        cf = _safe_df(tk.cash_flow if period == "yearly" else tk.quarterly_cashflow)
        inc = _safe_df(tk.income_stmt if period == "yearly" else tk.quarterly_income_stmt)

        out: Dict[str, Any] = {"ticker": ticker, "period": period}

        if statement in ("fundamentals", "all"):
            out["fundamentals"] = info
            out["fundamentals_analysis"] = analyze_fundamentals_info(info)

        if statement in ("balance_sheet", "all"):
            out["balance_sheet"] = analyze_balance_sheet_df(bs)

        if statement in ("cash_flow", "all"):
            out["cash_flow"] = analyze_cash_flow_df(cf)

        if statement in ("income_statement", "all"):
            out["income_statement"] = analyze_income_statement_df(inc)

        return out