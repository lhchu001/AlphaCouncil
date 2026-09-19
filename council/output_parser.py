# council/output_parser.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agent.schemas import Signal


def parse_council_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a Council member's JSON response from text.

    Handles:
    - Raw JSON
    - JSON wrapped in markdown code blocks
    - JSON embedded in longer text
    """
    if not text:
        return None

    cleaned = text.strip()

    # Remove markdown code blocks
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try direct JSON parse
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def validate_council_output(output: Dict[str, Any]) -> List[str]:
    """
    Validate a Council member's output against the expected schema.

    Returns a list of validation errors (empty if valid).
    """
    errors: List[str] = []

    # Validate signal
    signal = output.get("signal")
    valid_signals = {"buy", "sell", "hold", "abstain"}
    if signal not in valid_signals:
        errors.append(f"Invalid signal: {signal}. Must be one of {valid_signals}.")

    # Validate confidence
    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)):
        errors.append("Confidence is not numeric.")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append("Confidence is outside 0..1.")

    # Validate report
    if not isinstance(output.get("report"), str):
        errors.append("Report is not text.")

    # Validate role
    if not isinstance(output.get("role"), str):
        errors.append("Role is not text.")

    # Validate evidence (optional but should be list if present)
    evidence = output.get("evidence")
    if evidence is not None and not isinstance(evidence, list):
        errors.append("Evidence should be a list.")

    # Validate risks (optional but should be list if present)
    risks = output.get("risks")
    if risks is not None and not isinstance(risks, list):
        errors.append("Risks should be a list.")

    return errors


def extract_signal_from_report(report: str) -> Signal:
    """
    Extract the final signal from a Council chairman's report.

    Looks for patterns like:
    - "FINAL TRANSACTION PROPOSAL: **BUY**"
    - "FINAL RECOMMENDATION: SELL"
    """
    patterns = [
        r"FINAL TRANSACTION PROPOSAL:\s*\**\s*(BUY|HOLD|SELL)",
        r"FINAL RECOMMENDATION:\s*(BUY|HOLD|SELL)",
        r"RECOMMENDATION:\s*(BUY|HOLD|SELL)",
    ]

    for pattern in patterns:
        match = re.search(pattern, report, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()  # type: ignore[return-value]

    return "hold"


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Extract ranked labels (e.g., Response A, Response B) from text.

    Expected format:
        FINAL RANKING:
        1. Response A
        2. Response B
        3. Response C
    """
    if not ranking_text:
        return []

    ranking_section = ranking_text

    if "FINAL RANKING:" in ranking_text:
        ranking_section = ranking_text.split(
            "FINAL RANKING:",
            maxsplit=1,
        )[1]

    # Extract numbered rankings
    numbered_matches = re.findall(
        r"\d+\.\s*(Response [A-Z])",
        ranking_section,
    )

    if numbered_matches:
        return numbered_matches

    # Fallback: extract any "Response X" patterns
    return re.findall(r"Response [A-Z]", ranking_section)