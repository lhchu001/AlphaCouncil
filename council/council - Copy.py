# council/council.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from agent.schemas import SpecialistInterpretation
from infrastructure.config import load_settings
from infrastructure.reliability import retry_call
from llm.factory import load_llm
from langchain_core.messages import HumanMessage


def _get_ticker(state: Any) -> str:
    if isinstance(state, dict):
        return state.get("ticker", "UNKNOWN")
    return getattr(state, "ticker", "UNKNOWN")


def _get_evidence_facts(
    state: Any,
    key: str,
) -> Dict[str, Any]:
    if isinstance(state, dict):
        ev = state.get(key)
    else:
        ev = getattr(state, key, None)

    if ev is None:
        return {}

    if isinstance(ev, dict):
        return ev.get("facts", {})

    return getattr(ev, "facts", {})


def _extract_json(text: str) -> Dict[str, Any] | None:
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

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


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
        }

    ticker = _get_ticker(state)
    tech_facts = _get_evidence_facts(state, "technical_evidence")
    fund_facts = _get_evidence_facts(state, "fundamental_evidence")

    base_prompt = f"""
You are reviewing an equity analysis.

Ticker: {ticker}

Technical interpretation:
{technical.model_dump()}

Fundamental interpretation:
{fundamental.model_dump()}

Facts:
Technical:
{tech_facts}

Fundamental:
{fund_facts}

Return JSON:
{{
  "role": "bull" | "bear" | "neutral" | "chairman",
  "signal": "buy" | "sell" | "hold" | "abstain",
  "confidence": 0–1,
  "evidence": [...],
  "risks": [...],
  "report": "..."
}}
"""

    models = list(settings.council_models)
    results: List[Dict[str, Any]] = []

    for i, model in enumerate(models):
        role = (
            "bull"
            if i == 0
            else "bear"
            if i == 1
            else "neutral"
            if i == 2
            else "chairman"
        )

        llm = load_llm(
#            provider=settings.council_provider,
#            model=model,
#            timeout=settings.llm_timeout,
            settings.council_provider,
            model,
        )

        prompt = f"{base_prompt}\n\nYour role: {role}"

        try:
            response = retry_call(
                lambda: llm.invoke(
                    [HumanMessage(content=prompt)]
                ),
                retries=settings.llm_retries,
            )

            response_text = str(
                getattr(response, "content", response)
            )

            parsed = _extract_json(response_text)

            if parsed is None:
                raise ValueError(
                    "Council LLM did not return valid JSON."
                )

            signal_raw = parsed.get("signal", "hold")

            if signal_raw not in {"buy", "sell", "hold", "abstain"}:
                signal_raw = "hold"

            try:
                conf = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            conf = max(0.0, min(1.0, conf))

            results.append(
                {
                    "model": model,
                    "role": role,
                    "signal": signal_raw,
                    "confidence": conf,
                    "report": str(
                        parsed.get("report", "No report.")
                    ),
                    "raw_error": None,
                }
            )

        except Exception as exc:
            # Log internally; you can print if needed for debugging.
            results.append(
                {
                    "model": model,
                    "role": role,
                    "signal": "abstain",
                    "confidence": 0.0,
                    "report": f"Model failed: {exc}",
                    "raw_error": str(exc),
                }
            )

    non_abstain = [
        r for r in results if r["signal"] != "abstain"
    ]

    if len(non_abstain) < 2:
        # For debugging, you can temporarily print results here.
        print("COUNCIL DEBUG:")
        for r in results:
            print(r["model"], r["role"], r["signal"], r["report"][:200])
            
        return {
            "signal": "hold",
            "confidence": 0.0,
            "report": "Insufficient Council responses.",
            "members": results,
        }

    votes = {
        "buy": 0.0,
        "sell": 0.0,
        "hold": 0.0,
    }

    for r in non_abstain:
        weight = r["confidence"]
        votes[r["signal"]] += weight

    signal = max(votes, key=votes.get)
    total = sum(votes.values()) or 1.0
    confidence = votes[signal] / total

    chairman = next(
        (r for r in results if r["role"] == "chairman"),
        None,
    )

    report = (
        chairman["report"]
        if chairman and chairman["signal"] != "abstain"
        else "Chairman abstained; decision based on votes."
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "report": report,
        "members": results,
    }