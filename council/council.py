# council/council.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from agent.schemas import SpecialistInterpretation
from infrastructure.config import load_settings
from infrastructure.reliability import retry_call
from llm.factory import load_llm


def _get_ticker(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("ticker", "UNKNOWN"))

    return str(getattr(state, "ticker", "UNKNOWN"))


def _get_evidence_facts(
    state: Any,
    key: str,
) -> Dict[str, Any]:
    if isinstance(state, dict):
        evidence = state.get(key)
    else:
        evidence = getattr(state, key, None)

    if evidence is None:
        return {}

    if isinstance(evidence, dict):
        facts = evidence.get("facts", {})
        return facts if isinstance(facts, dict) else {}

    facts = getattr(evidence, "facts", {})
    return facts if isinstance(facts, dict) else {}


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content).strip()


def _extract_json_object(
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

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(
                cleaned[index:]
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            continue

    return None


def _normalise_signal(value: Any) -> str:
    if not isinstance(value, str):
        return "hold"

    signal = value.strip().lower()

    if signal in {
        "buy",
        "sell",
        "hold",
        "abstain",
    }:
        return signal

    return "hold"


def _normalise_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, confidence))


def _short_interpretation(
    interpretation: SpecialistInterpretation,
) -> Dict[str, Any]:
    return {
        "status": interpretation.status,
        "signal": interpretation.signal,
        "confidence": interpretation.confidence,
        "evidence": interpretation.evidence[:3],
        "risks": interpretation.risks[:3],
        "conflicts": interpretation.conflicts[:3],
        "report": interpretation.report[:1200],
    }


def _build_prompt(
    ticker: str,
    technical: SpecialistInterpretation,
    fundamental: SpecialistInterpretation,
    technical_facts: Dict[str, Any],
    fundamental_facts: Dict[str, Any],
    role: str,
) -> str:
    return f"""
You are the {role} member of an equity-analysis Council.

Ticker: {ticker}

Technical interpretation:
{json.dumps(_short_interpretation(technical), default=str)}

Fundamental interpretation:
{json.dumps(_short_interpretation(fundamental), default=str)}

Technical facts:
{json.dumps(technical_facts, default=str)[:2500]}

Fundamental facts:
{json.dumps(fundamental_facts, default=str)[:2500]}

Return exactly one JSON object and nothing else.
The first character must be {{ and the last character must be }}.
Do not use Markdown or code fences.
Keep the response below 250 tokens.

Required JSON shape:
{{
  "signal": "buy",
  "confidence": 0.0,
  "evidence": ["One short fact-based reason."],
  "risks": ["One short risk."],
  "report": "A concise conclusion."
}}

Allowed signal values are: buy, sell, hold, abstain.
Use only the supplied information.
"""


def run_council(
    technical: SpecialistInterpretation,
    fundamental: SpecialistInterpretation,
    state: Any,
) -> Dict[str, Any]:
    settings = load_settings()

    if not settings.council_models:
        return {
            "signal": "hold",
            "confidence": 0.0,
            "report": "Council unavailable.",
            "members": [],
        }

    ticker = _get_ticker(state)
    technical_facts = _get_evidence_facts(
        state,
        "technical_evidence",
    )
    fundamental_facts = _get_evidence_facts(
        state,
        "fundamental_evidence",
    )

    results: List[Dict[str, Any]] = []

    for index, model in enumerate(settings.council_models):
        role = (
            "bull"
            if index == 0
            else "bear"
            if index == 1
            else "neutral"
            if index == 2
            else "chairman"
        )

        response_text = ""

        try:
            llm = load_llm(
#                provider=settings.council_provider,
#                model=model,
#                timeout=settings.llm_timeout,
                settings.council_provider,
                model,
            )

            prompt = _build_prompt(
                ticker,
                technical,
                fundamental,
                technical_facts,
                fundamental_facts,
                role,
            )

            response = retry_call(
                lambda: llm.invoke(
                    [HumanMessage(content=prompt)]
                ),
                retries=settings.llm_retries,
            )

            response_text = _response_text(response)
            parsed = _extract_json_object(response_text)

            if parsed is None:
                raise ValueError(
                    "Council LLM did not return valid JSON."
                )

            signal = _normalise_signal(
                parsed.get("signal", "hold")
            )
            confidence = _normalise_confidence(
                parsed.get("confidence", 0.0)
            )

            results.append(
                {
                    "model": model,
                    "role": role,
                    "signal": signal,
                    "confidence": confidence,
                    "report": str(
                        parsed.get(
                            "report",
                            "No report supplied.",
                        )
                    ),
                    "evidence": parsed.get(
                        "evidence",
                        [],
                    ),
                    "risks": parsed.get(
                        "risks",
                        [],
                    ),
                }
            )

        except Exception as exc:
            print(
                f"Council model failed: {model} "
                f"({role}): {exc}"
            )

            if response_text:
                print(
                    "Raw Council response:"
                )
                print(response_text[:2000])

            results.append(
                {
                    "model": model,
                    "role": role,
                    "signal": "abstain",
                    "confidence": 0.0,
                    "report": f"Model failed: {exc}",
                    "evidence": [],
                    "risks": [],
                }
            )

    non_abstain = [
        result
        for result in results
        if result["signal"] != "abstain"
    ]

    if not non_abstain:
        return {
            "signal": "hold",
            "confidence": 0.0,
            "report": "No Council model returned a valid response.",
            "members": results,
        }

    if len(non_abstain) == 1:
        only_member = non_abstain[0]

        return {
            "signal": only_member["signal"],
            "confidence": only_member["confidence"],
            "report": (
                "Provisional Council suggestion: only one "
                "member returned valid JSON.\n\n"
                + only_member["report"]
            ),
            "members": results,
            "provisional": True,
        }

    votes = {
        "buy": 0.0,
        "sell": 0.0,
        "hold": 0.0,
    }

    for result in non_abstain:
        signal = result["signal"]

        if signal in votes:
            votes[signal] += result["confidence"]

    signal = max(
        votes,
        key=votes.get,
    )

    total = sum(votes.values())
    confidence = (
        votes[signal] / total
        if total > 0
        else 0.0
    )

    chairman = next(
        (
            result
            for result in results
            if result["role"] == "chairman"
            and result["signal"] != "abstain"
        ),
        None,
    )

    if chairman is not None:
        report = chairman["report"]
    else:
        report = (
            "Council decision based on weighted member votes."
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "report": report,
        "members": results,
        "provisional": False,
    }
