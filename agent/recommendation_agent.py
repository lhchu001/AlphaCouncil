# agent/recommendation_agent.py
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from agent.fundamental_agent import build_fundamental_agent
from agent.indicator_agent import build_indicator_agent
from agent.schemas import (
    Decision,
    Signal,
    SpecialistInterpretation,
)
from council.council import run_council
from infrastructure.config import load_settings


def _vote(signal: str) -> float:
    return {
        "buy": 1.0,
        "hold": 0.0,
        "sell": -1.0,
    }.get(signal, 0.0)


def _safe_signal(
    interpretation: Any,
) -> Signal:
    signal = getattr(
        interpretation,
        "signal",
        "hold",
    )

    if signal not in {
        "buy",
        "hold",
        "sell",
    }:
        return "hold"

    return signal


def _safe_confidence(
    interpretation: Any,
) -> float:
    try:
        value = float(
            getattr(
                interpretation,
                "confidence",
                0.0,
            )
        )
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(1.0, value),
    )


def _evidence_is_stale(
    evidence: Any,
) -> bool:
    if evidence is None:
        return True

    quality = getattr(
        evidence,
        "quality",
        None,
    )

    if quality is None:
        return True

    return bool(
        getattr(
            quality,
            "stale",
            True,
        )
    )


def _default_technical_interpretation() -> SpecialistInterpretation:
    return SpecialistInterpretation(
        status="unavailable",
        signal="hold",
        confidence=0.0,
        risks=[
            "Technical interpretation unavailable."
        ],
        report="Technical interpretation unavailable.",
    )


def _default_fundamental_interpretation() -> SpecialistInterpretation:
    return SpecialistInterpretation(
        status="unavailable",
        signal="hold",
        confidence=0.0,
        risks=[
            "Fundamental interpretation unavailable."
        ],
        report="Fundamental interpretation unavailable.",
    )


def run_technical(
    state: Dict[str, Any],
    technical_agent,
) -> Dict[str, Any]:
    result = technical_agent.invoke(
        {
            "messages": state.get(
                "messages",
                [],
            ),
            "ticker": state["ticker"],
            "start": state["start"],
            "end": state["end"],
            "interval": state["interval"],
            "period": state["period"],
            "query": state["query"],
        }
    )

    return {
        **state,
        "technical_evidence": result.get(
            "technical_evidence"
        ),
        "technical_interpretation": result.get(
            "technical_interpretation"
        ),
        "technical_signal": result.get(
            "technical_signal",
            "hold",
        ),
        "technical_confidence": result.get(
            "technical_confidence",
            0.0,
        ),
        "technical_analysis": result.get(
            "analysis",
            "",
        ),
    }


def run_fundamental(
    state: Dict[str, Any],
    fundamental_agent,
) -> Dict[str, Any]:
    result = fundamental_agent.invoke(
        {
            "messages": state.get(
                "messages",
                [],
            ),
            "ticker": state["ticker"],
            "start": state["start"],
            "end": state["end"],
            "interval": state["interval"],
            "period": state["period"],
            "query": state["query"],
        }
    )

    return {
        **state,
        "fundamental_evidence": result.get(
            "fundamental_evidence"
        ),
        "fundamental_interpretation": result.get(
            "fundamental_interpretation"
        ),
        "fundamental_signal": result.get(
            "fundamental_signal",
            "hold",
        ),
        "fundamental_confidence": result.get(
            "fundamental_confidence",
            0.0,
        ),
        "fundamental_analysis": result.get(
            "analysis",
            "",
        ),
    }


def combine_decision(
    state: Dict[str, Any],
    settings,
) -> Dict[str, Any]:
    technical = state.get(
        "technical_interpretation"
    ) or _default_technical_interpretation()

    fundamental = state.get(
        "fundamental_interpretation"
    ) or _default_fundamental_interpretation()

    technical_signal = _safe_signal(technical)
    fundamental_signal = _safe_signal(fundamental)

    technical_confidence = _safe_confidence(technical)
    fundamental_confidence = _safe_confidence(fundamental)

    score = (
        settings.technical_weight
        * _vote(technical_signal)
        + settings.fundamental_weight
        * _vote(fundamental_signal)
    )

    if score >= settings.weak_signal_threshold:
        signal: Signal = "buy"
    elif score <= -settings.weak_signal_threshold:
        signal = "sell"
    else:
        signal = "hold"

    technical_stale = _evidence_is_stale(
        state.get("technical_evidence")
    )

    fundamental_stale = _evidence_is_stale(
        state.get("fundamental_evidence")
    )

    stale_data = (
        technical_stale
        or fundamental_stale
    )

    disagreement = (
        technical_signal != fundamental_signal
    )

    low_confidence = (
        technical_confidence
        < settings.low_confidence_threshold
        or fundamental_confidence
        < settings.low_confidence_threshold
    )

    weak_signal = (
        abs(score)
        < settings.weak_signal_threshold
    )

    risk_flags = []

    if disagreement:
        risk_flags.append(
            "technical_fundamental_disagreement"
        )

    if low_confidence:
        risk_flags.append(
            "low_specialist_confidence"
        )

    if weak_signal:
        risk_flags.append(
            "weak_combined_signal"
        )

    if stale_data:
        risk_flags.append(
            "stale_data_or_missing_evidence"
        )

    council_required = (
        disagreement
        or low_confidence
        or weak_signal
    )

    human_review_required = (
        stale_data
        or (
            signal != "hold"
            and low_confidence
        )
    )

    decision = Decision(
        signal=signal,
        score=score,
        reason=(
            "Decision generated from validated "
            "technical and fundamental interpretations."
        ),
        council_used=False,
        human_review_required=human_review_required,
        risk_flags=risk_flags,
        risk_dashboard={
            "technical_signal": technical_signal,
            "fundamental_signal": fundamental_signal,
            "technical_confidence": technical_confidence,
            "fundamental_confidence": fundamental_confidence,
            "signal_disagreement": disagreement,
            "technical_data_stale": technical_stale,
            "fundamental_data_stale": fundamental_stale,
            "council_required": council_required,
        },
    )

    return {
        **state,
        "decision": decision,
        "_council_required": council_required,
    }


def final_synthesis(
    state: Dict[str, Any],
    settings,
) -> Dict[str, Any]:
    decision = state.get("decision")

    if decision is None:
        decision = Decision(
            signal="hold",
            score=0.0,
            reason="No decision was produced.",
            council_used=False,
            human_review_required=True,
            risk_flags=["missing_decision"],
        )

    council_required = bool(
        state.get(
            "_council_required",
            False,
        )
    )

    if not council_required:
        decision.council_used = False

        final_analysis = (
            f"Decision: {decision.signal.upper()}\n"
            f"Score: {decision.score:.2f}\n"
            f"Reason: {decision.reason}\n"
            f"Risk flags: {decision.risk_flags}\n"
            f"Human review required: "
            f"{decision.human_review_required}"
        )

        return {
            **state,
            "decision": decision,
            "final_analysis": final_analysis,
        }

    technical = state.get(
        "technical_interpretation"
    ) or _default_technical_interpretation()

    fundamental = state.get(
        "fundamental_interpretation"
    ) or _default_fundamental_interpretation()

    try:
        council_result = run_council(
            technical,
            fundamental,
            state,
        )

        council_signal = council_result.get(
            "signal",
            "hold",
        )

        if council_signal not in {
            "buy",
            "hold",
            "sell",
        }:
            council_signal = "hold"

        decision.signal = council_signal
        decision.council_used = True
        decision.reason = council_result.get(
            "report",
            "Council synthesis completed.",
        )

        final_analysis = council_result.get(
            "report",
            "Council synthesis completed.",
        )

    except Exception as exc:
        decision.signal = "hold"
        decision.council_used = True
        decision.human_review_required = True
        decision.risk_flags.append(
            "council_failure"
        )
        decision.reason = (
            "Council failed; defaulting to HOLD."
        )

        final_analysis = (
            "Council synthesis failed: "
            f"{exc}\n"
            "The system defaulted to HOLD and requires "
            "human review."
        )

    return {
        **state,
        "decision": decision,
        "final_analysis": final_analysis,
    }


def build_recommendation_agent(
    checkpointer=None,
):
    """
    Build the recommendation graph.

    Specialist graphs are compiled once here. The technical and
    fundamental graphs are reused on every invocation.
    """
    settings = load_settings()

    technical_agent = build_indicator_agent()
    fundamental_agent = build_fundamental_agent()

    builder = StateGraph(dict)

    builder.add_node(
        "run_technical",
        lambda state: run_technical(
            state,
            technical_agent,
        ),
    )

    builder.add_node(
        "run_fundamental",
        lambda state: run_fundamental(
            state,
            fundamental_agent,
        ),
    )

    builder.add_node(
        "combine_decision",
        lambda state: combine_decision(
            state,
            settings,
        ),
    )

    builder.add_node(
        "final_synthesis",
        lambda state: final_synthesis(
            state,
            settings,
        ),
    )

    builder.add_edge(
        START,
        "run_technical",
    )

    builder.add_edge(
        "run_technical",
        "run_fundamental",
    )

    builder.add_edge(
        "run_fundamental",
        "combine_decision",
    )

    builder.add_edge(
        "combine_decision",
        "final_synthesis",
    )

    builder.add_edge(
        "final_synthesis",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer
    )
