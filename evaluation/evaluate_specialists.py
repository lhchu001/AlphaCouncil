# evaluation/evaluate_specialists.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def validate_specialist_output(
    output: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []

    if output.get("signal") not in {
        "buy",
        "sell",
        "hold",
    }:
        errors.append("Invalid signal.")

    confidence = output.get("confidence")

    if not isinstance(confidence, (int, float)):
        errors.append("Confidence is not numeric.")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append("Confidence is outside 0..1.")

    if not isinstance(output.get("report"), str):
        errors.append("Report is not text.")

    return errors


def evaluate_cases(
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid = 0
    invalid = 0

    for case in cases:
        errors = validate_specialist_output(
            case["output"]
        )

        if errors:
            invalid += 1
            case["errors"] = errors
        else:
            valid += 1

    return {
        "total": len(cases),
        "valid": valid,
        "invalid": invalid,
        "structured_output_rate": (
            valid / len(cases)
            if cases
            else 0.0
        ),
    }


if __name__ == "__main__":
    path = Path("evaluation/cases.json")
    cases = json.loads(
        path.read_text(encoding="utf-8")
    )

    print(
        json.dumps(
            evaluate_cases(cases),
            indent=2,
        )
    )