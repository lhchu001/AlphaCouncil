# agent/fundamental_agent.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agent.schemas import (
    Evidence,
    Signal,
    SpecialistInterpretation,
)
from infrastructure.config import load_settings
from infrastructure.provenance import build_evidence
from infrastructure.reliability import retry_call
from llm import load_llm, set_langsmith
from tools.fundamentals_tool import FundamentalsTool


class State(TypedDict, total=False):
    ticker: str
    start: str
    end: str
    interval: str
    period: str
    query: str
    messages: Optional[List[Any]]

    statement: Optional[str]
    fundamentals_result: Optional[Dict[str, Any]]

    fundamental_evidence: Optional[Evidence]
    fundamental_interpretation: Optional[SpecialistInterpretation]

    fundamental_signal: Optional[Signal]
    fundamental_confidence: Optional[float]
    fundamental_score: Optional[float]
    analysis: Optional[str]


def get_llm():
    settings = load_settings()

    set_langsmith("Trading Agent")

    llm = load_llm(
        settings.fundamental_provider,
        settings.fundamental_model,
    )

    if llm is None:
        raise RuntimeError(
            "Failed to load fundamental LLM: "
            f"provider={settings.fundamental_provider}, "
            f"model={settings.fundamental_model}"
        )

    return llm


def plan_fundamentals(state: State) -> State:
    return {
        **state,
        "statement": "all",
    }


def fetch_fundamentals(state: State) -> State:
    tool = FundamentalsTool()

    try:
        result = tool._run(
            ticker=state["ticker"],
            statement=state.get("statement") or "all",
            period=state.get("period", "yearly"),
        )
    except Exception as exc:
        result = {
            "ticker": state["ticker"],
            "period": state.get("period", "yearly"),
            "error": str(exc),
        }

    return {
        **state,
        "fundamentals_result": result,
    }


def build_fundamental_evidence(state: State) -> State:
    result = state.get("fundamentals_result") or {}

    evidence = build_evidence(
        source="yfinance",
        ticker=state["ticker"],
        facts=result,
        retrieved_at=datetime.now(timezone.utc),
        market_date=state.get("end"),
        period=state.get("period", "yearly"),
        cache_hit=bool(result.get("cache_hit", False)),
        max_age_seconds=86400.0,
    )

    if result.get("error"):
        evidence.quality.status = "degraded"
        evidence.quality.missing_fields.append(
            "fundamental_data"
        )

    return {
        **state,
        "fundamental_evidence": evidence,
    }


def _extract_json(
    text: str,
) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def _extract_signal(text: str) -> Signal:
    match = re.search(
        r"(?:FINAL TRANSACTION PROPOSAL|SIGNAL)"
        r"\s*:\s*\**\s*(BUY|HOLD|SELL)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return "hold"

    return match.group(1).lower()  # type: ignore[return-value]


def _confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, result))


def _fundamental_prompt(
    state: State,
    evidence: Evidence,
) -> str:
    compact_facts = {
        "ticker": state["ticker"],
        "period": state.get("period"),
        "fundamentals": evidence.facts.get(
            "fundamentals",
            {},
        ),
        "fundamentals_analysis": evidence.facts.get(
            "fundamentals_analysis",
            {},
        ),
        "balance_sheet": evidence.facts.get(
            "balance_sheet",
            {},
        ),
        "cash_flow": evidence.facts.get(
            "cash_flow",
            {},
        ),
        "income_statement": evidence.facts.get(
            "income_statement",
            {},
        ),
    }

    return f"""
You are a quantitative fundamental analyst.

Ticker: {state["ticker"]}
Financial reporting period: {state.get("period", "yearly")}

User request:
{state.get("query", "Provide a general fundamental analysis.")}

Validated fundamental facts:
{json.dumps(compact_facts, indent=2, default=str)}

Data quality:
{json.dumps(evidence.quality.model_dump(), indent=2)}

Provenance:
{json.dumps(evidence.provenance.model_dump(), indent=2, default=str)}

Analyze:

1. Valuation and company quality.
2. Profitability and return metrics.
3. Balance-sheet liquidity and leverage.
4. Operating cash flow, capital expenditure, and free cash flow.
5. Revenue growth and margins.
6. Material risks and conflicting evidence.

Return only valid JSON:

{{
  "status": "ok",
  "signal": "buy",
  "confidence": 0.0,
  "evidence": [
    "Specific fact used from the supplied data."
  ],
  "risks": [
    "Specific risk supported by the supplied data."
  ],
  "conflicts": [
    "Conflicting evidence, or an empty list."
  ],
  "report": "Detailed Markdown fundamental analysis."
}}

Rules:
- signal must be buy, hold, or sell.
- confidence must be between 0 and 1.
- Use only the supplied facts.
- Do not invent values, ratios, dates, trends, or peer comparisons.
- If a field is missing, say it is unavailable.
- Distinguish facts from interpretations.
- End the report with a Markdown table.
- Do not provide personalized financial advice.
"""


def analyze_fundamentals(
    state: State,
    llm,
) -> State:
    evidence = state.get("fundamental_evidence")

    if evidence is None:
        interpretation = SpecialistInterpretation(
            status="unavailable",
            signal="hold",
            confidence=0.0,
            risks=[
                "Fundamental evidence was not created."
            ],
            report=(
                "Fundamental analysis is unavailable because "
                "no fundamental evidence was created."
            ),
        )

        return {
            **state,
            "fundamental_signal": "hold",
            "fundamental_confidence": 0.0,
            "fundamental_score": None,
            "analysis": interpretation.report,
            "fundamental_interpretation": interpretation,
        }

    if evidence.quality.status != "ok":
        interpretation = SpecialistInterpretation(
            status="degraded",
            signal="hold",
            confidence=0.0,
            risks=[
                "Fundamental data quality is degraded."
            ],
            report=(
                "Fundamental analysis is unavailable because "
                "the retrieved fundamental data is incomplete "
                "or otherwise degraded."
            ),
        )

        return {
            **state,
            "fundamental_signal": "hold",
            "fundamental_confidence": 0.0,
            "fundamental_score": None,
            "analysis": interpretation.report,
            "fundamental_interpretation": interpretation,
        }

    settings = load_settings()
    prompt = _fundamental_prompt(state, evidence)

    try:
        response = retry_call(
            lambda: llm.invoke(
                [HumanMessage(content=prompt)]
            ),
            retries=settings.llm_retries,
        )

        response_text = str(
            getattr(response, "content", response)
        ).strip()
        parsed = _extract_json(response_text)

    except Exception as exc:
        parsed = None
        response_text = (
            "Fundamental LLM analysis failed: "
            f"{exc}"
        )

    if parsed is not None:
        raw_signal = parsed.get("signal", "hold")

        signal: Signal = (
            raw_signal
            if raw_signal in {"buy", "hold", "sell"}
            else "hold"
        )

        confidence = _confidence(
            parsed.get("confidence", 0.0)
        )

        report = str(
            parsed.get(
                "report",
                parsed.get(
                    "analysis",
                    response_text,
                ),
            )
        )

        evidence_items = parsed.get("evidence", [])
        risks = parsed.get("risks", [])
        conflicts = parsed.get("conflicts", [])

        interpretation_status = parsed.get(
            "status",
            "ok",
        )

        if interpretation_status not in {
            "ok",
            "unavailable",
            "degraded",
        }:
            interpretation_status = "degraded"

    else:
        signal = _extract_signal(response_text)
        confidence = 0.0
        report = response_text
        evidence_items = []
        risks = [
            "The LLM response was not valid JSON."
        ]
        conflicts = []
        interpretation_status = "degraded"

    interpretation = SpecialistInterpretation(
        status=interpretation_status,
        signal=signal,
        confidence=confidence,
        evidence=[
            str(item)
            for item in evidence_items
            if item is not None
        ],
        risks=[
            str(item)
            for item in risks
            if item is not None
        ],
        conflicts=[
            str(item)
            for item in conflicts
            if item is not None
        ],
        report=report,
        model_provider=settings.fundamental_provider,
        model_name=settings.fundamental_model,
    )

    return {
        **state,
        "fundamental_signal": interpretation.signal,
        "fundamental_confidence": interpretation.confidence,
        "fundamental_score": None,
        "analysis": interpretation.report,
        "fundamental_interpretation": interpretation,
    }


def build_fundamental_agent():
    """
    Build the graph and load the fundamental LLM once.
    """
    llm = get_llm()

    builder = StateGraph(State)

    builder.add_node(
        "plan_fundamentals",
        plan_fundamentals,
    )
    builder.add_node(
        "fetch_fundamentals",
        fetch_fundamentals,
    )
    builder.add_node(
        "build_fundamental_evidence",
        build_fundamental_evidence,
    )
    builder.add_node(
        "analyze_fundamentals",
        lambda state: analyze_fundamentals(
            state,
            llm,
        ),
    )

    builder.add_edge(
        START,
        "plan_fundamentals",
    )
    builder.add_edge(
        "plan_fundamentals",
        "fetch_fundamentals",
    )
    builder.add_edge(
        "fetch_fundamentals",
        "build_fundamental_evidence",
    )
    builder.add_edge(
        "build_fundamental_evidence",
        "analyze_fundamentals",
    )
    builder.add_edge(
        "analyze_fundamentals",
        END,
    )

    return builder.compile()