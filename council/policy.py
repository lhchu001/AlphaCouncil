# council/policy.py
from __future__ import annotations

from typing import Any, Dict, List

from agent.schemas import Decision, Signal


def should_use_council(
    technical_signal: Signal,
    fundamental_signal: Signal,
    technical_confidence: float,
    fundamental_confidence: float,
    combined_score: float,
    weak_signal_threshold: float,
    low_confidence_threshold: float,
) -> bool:
    """
    Determine whether Council escalation is required.

    Council is used when:
    - Technical and fundamental signals disagree
    - Either specialist has low confidence
    - Combined signal is weak (near zero)
    """
    disagreement = technical_signal != fundamental_signal
    low_conf = (
        technical_confidence < low_confidence_threshold
        or fundamental_confidence < low_confidence_threshold
    )
    weak = abs(combined_score) < weak_signal_threshold

    return disagreement or low_conf or weak


def aggregate_council_votes(
    council_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate Council member votes with confidence weighting.

    Abstentions are excluded from the vote count.
    Requires at least 2 non-abstaining responses.
    """
    non_abstain = [
        r for r in council_results
        if r.get("signal") != "abstain"
    ]

    if len(non_abstain) < 2:
        return {
            "signal": "hold",
            "confidence": 0.0,
            "report": "Insufficient non-abstaining responses.",
        }

    votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

    for r in non_abstain:
        weight = r.get("confidence", 0.5)
        signal = r.get("signal", "hold")
        votes[signal] += weight

    winning_signal = max(votes, key=votes.get)
    total = sum(votes.values()) or 1.0
    confidence = votes[winning_signal] / total

    return {
        "signal": winning_signal,
        "confidence": confidence,
        "votes": votes,
    }


def determine_human_review(
    decision: Decision,
    stale_data: bool,
) -> bool:
    """
    Determine whether human review is required.

    Human review is required when:
    - Data is stale
    - Signal is non-HOLD with low confidence or disagreement
    """
    if stale_data:
        return True

    if decision.signal != "hold":
        risk_flags = decision.risk_flags
        if (
            "low_specialist_confidence" in risk_flags
            or "technical_fundamental_disagreement" in risk_flags
        ):
            return True

    return False


def compute_risk_flags(
    technical_signal: Signal,
    fundamental_signal: Signal,
    technical_confidence: float,
    fundamental_confidence: float,
    combined_score: float,
    weak_signal_threshold: float,
    low_confidence_threshold: float,
    stale_data: bool,
) -> List[str]:
    """
    Compute risk flags for the decision dashboard.
    """
    risk_flags = []

    if technical_signal != fundamental_signal:
        risk_flags.append("technical_fundamental_disagreement")

    if (
        technical_confidence < low_confidence_threshold
        or fundamental_confidence < low_confidence_threshold
    ):
        risk_flags.append("low_specialist_confidence")

    if abs(combined_score) < weak_signal_threshold:
        risk_flags.append("weak_combined_signal")

    if stale_data:
        risk_flags.append("stale_data")

    return risk_flags