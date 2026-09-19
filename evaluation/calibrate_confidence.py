# evaluation/calibrate_confidence.py
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_history(path: str = "data/signal_history.jsonl") -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def calibrate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = defaultdict(list)

    for r in records:
        conf = r["result"]["decision"]["score"]
        bucket = round(max(0.0, min(1.0, abs(conf))), 1)
        correct = r["result"]["decision"]["signal"] == "buy"
        buckets[bucket].append(correct)

    calibration = {}
    for b, vals in sorted(buckets.items()):
        calibration[b] = sum(vals) / len(vals) if vals else 0.0

    return {"calibration": calibration}


if __name__ == "__main__":
    records = load_history()
    print(json.dumps(calibrate(records), indent=2))