# agent/indicator_agent.py
from __future__ import annotations

import json
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
from tools.technical_indicators_tool import (
    IndicatorName,
    TechnicalIndicatorsTool,
)


class State(TypedDict, total=False):
    ticker: str
    start: str
    end: str
    interval: str
    period: str
    query: str
    messages: List[Any]

    indicators_requested: List[IndicatorName]
    indicator_results: Dict[str, Any]

    technical_evidence: Optional[Evidence]
    technical_interpretation: Optional[SpecialistInterpretation]

    technical_signal: str
    technical_confidence: float
    technical_score: Optional[float]
    analysis: str


DEFAULT_INDICATORS: List[IndicatorName] = [
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


def get_llm():
    settings = load_settings()

    set_langsmith("Trading Agent")

    llm = load_llm(
        settings.indicator_provider,
        settings.indicator_model,
    )

    if llm is None:
        raise RuntimeError(
            "Failed to load technical-analysis LLM: "
            f"provider={settings.indicator_provider}, "
            f"model={settings.indicator_model}"
        )

    return llm


def plan_indicators(state: State) -> State:
    return {
        **state,
        "indicators_requested": DEFAULT_INDICATORS,
    }


def fetch_indicators(state: State) -> State:
    tool = TechnicalIndicatorsTool()

    result = tool._run(
        ticker=state["ticker"],
        start=state["start"],
        end=state["end"],
        indicators=state.get(
            "indicators_requested",
            DEFAULT_INDICATORS,
        ),
        interval=state.get("interval", "1d"),
    )

    return {
        **state,
        "indicator_results": result,
    }


def build_technical_evidence(
    state: State,
) -> State:
    result = state.get("indicator_results", {})

    if result.get("error"):
        facts = {
            "error": result["error"],
            "indicators": [],
            "data_points": 0,
        }
    else:
        facts = {
            "ticker": result.get("ticker"),
            "start": result.get("start"),
            "end": result.get("end"),
            "interval": result.get("interval"),
            "indicators": result.get("indicators", []),
            "data_points": result.get("data_points", 0),
        }

    evidence = build_evidence(
        source="yfinance",
        ticker=state["ticker"],
        facts=facts,
        retrieved_at=datetime.now(timezone.utc),
        market_date=state["end"],
        period=state.get("interval", "1d"),
        cache_hit=False,
        max_age_seconds=3600.0,
    )

    return {
        **state,
        "technical_evidence": evidence,
    }


def _technical_prompt(
    state: State,
    evidence: Evidence,
) -> str:
    compact = {
        "ticker": state["ticker"],
        "start": state["start"],
        "end": state["end"],
        "interval": state["interval"],
        "indicators": evidence.facts.get("indicators", []),
        "data_points": evidence.facts.get("data_points", 0),
    }

    return f"""
You are a technical equity analyst.

Ticker: {state["ticker"]}
Period: {state["start"]} to {state["end"]}
Interval: {state["interval"]}

Computed technical facts:
{json.dumps(compact, indent=2, default=str)}

Data quality:
{json.dumps(evidence.quality.model_dump(), indent=2)}

Provenance:
{json.dumps(evidence.provenance.model_dump(), indent=2, default=str)}

Task:
- Interpret the technical evidence.
- Produce a short, actionable technical view.

Return **only** valid JSON, no extra text:

{{
  "status": "ok",
  "signal": "buy",
  "confidence": 0.0,
  "evidence": [
    "Specific fact used."
  ],
  "risks": [
    "Specific technical risk."
  ],
  "conflicts": [],
  "report": "Short technical interpretation."
}}

Rules:
- signal must be buy, hold, or sell.
- confidence must be between 0 and 1.
- Use only the supplied facts.
- Do not invent prices, dates, indicator values, or returns.
- Keep the report concise (max ~150 words).
- Do not include any text outside the JSON object.
"""


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


def _safe_signal(text: str) -> Signal:
    match = re.search(
        r"(?:signal|SIGNAL)\s*:\s*\"?(buy|hold|sell)\"?",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return "hold"

    s = match.group(1).lower()

    if s in {"buy", "hold", "sell"}:
        return s  # type: ignore[return-value]

    return "hold"


def _safe_confidence(text: str) -> float:
    match = re.search(
        r"(?:confidence|CONFIDENCE)\s*:\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return 0.0

    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, value))


def interpret_technical(
    state: State,
    llm,
) -> State:
    evidence = state.get("technical_evidence")

    if evidence is None:
        interpretation = SpecialistInterpretation(
            status="unavailable",
            signal="hold",
            confidence=0.0,
            risks=["Technical evidence was not created."],
            report=(
                "Technical interpretation unavailable because "
                "technical evidence was not created."
            ),
        )

        return {
            **state,
            "technical_signal": "hold",
            "technical_confidence": 0.0,
            "technical_score": None,
            "analysis": interpretation.report,
            "technical_interpretation": interpretation,
        }

    if evidence.quality.status != "ok":
        interpretation = SpecialistInterpretation(
            status="degraded",
            signal="hold",
            confidence=0.0,
            risks=["Technical data quality is degraded."],
            report=(
                "Technical interpretation unavailable because "
                "technical data quality is degraded."
            ),
        )

        return {
            **state,
            "technical_signal": "hold",
            "technical_confidence": 0.0,
            "technical_score": None,
            "analysis": interpretation.report,
            "technical_interpretation": interpretation,
        }

    settings = load_settings()
    prompt = _technical_prompt(state, evidence)

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
            "Technical LLM analysis failed: "
            f"{exc}"
        )

    if parsed is not None:
        raw_signal = parsed.get("signal", "hold")

        signal: Signal = (
            raw_signal
            if raw_signal in {"buy", "hold", "sell"}
            else "hold"
        )

        try:
            confidence = float(
                parsed.get("confidence", 0.0)
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

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
            interpretation_status = "ok"

    else:
        signal = _safe_signal(response_text)
        confidence = _safe_confidence(response_text)
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
        model_provider=settings.indicator_provider,
        model_name=settings.indicator_model,
    )

    return {
        **state,
        "technical_signal": interpretation.signal,
        "technical_confidence": interpretation.confidence,
        "technical_score": None,
        "analysis": interpretation.report,
        "technical_interpretation": interpretation,
    }


def build_indicator_agent():
    llm = get_llm()

    builder = StateGraph(State)

    builder.add_node(
        "plan_indicators",
        plan_indicators,
    )
    builder.add_node(
        "fetch_indicators",
        fetch_indicators,
    )
    builder.add_node(
        "build_technical_evidence",
        build_technical_evidence,
    )
    builder.add_node(
        "interpret_technical",
        lambda state: interpret_technical(
            state,
            llm,
        ),
    )

    builder.add_edge(
        START,
        "plan_indicators",
    )
    builder.add_edge(
        "plan_indicators",
        "fetch_indicators",
    )
    builder.add_edge(
        "fetch_indicators",
        "build_technical_evidence",
    )
    builder.add_edge(
        "build_technical_evidence",
        "interpret_technical",
    )
    builder.add_edge(
        "interpret_technical",
        END,
    )

    return builder.compile()