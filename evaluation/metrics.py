# evaluation/metrics.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def evaluate_signal(
    predicted: str,
    actual: str,
) -> Dict[str, Any]:
    """
    Evaluate a single signal prediction.

    Returns a dict with predicted, actual, and correctness.
    """
    return {
        "predicted": predicted,
        "actual": actual,
        "correct": predicted == actual,
    }


def accuracy(results: Iterable[Dict[str, Any]]) -> float:
    """
    Calculate accuracy from a list of evaluation results.
    """
    items = list(results)

    if not items:
        return 0.0

    correct = sum(
        bool(item["correct"])
        for item in items
    )

    return correct / len(items)


def precision_recall(
    results: Iterable[Dict[str, Any]],
    positive_label: str = "buy",
) -> Dict[str, float]:
    """
    Calculate precision and recall for a given positive label.
    """
    items = list(results)

    true_positives = sum(
        1 for item in items
        if item["predicted"] == positive_label and item["actual"] == positive_label
    )
    false_positives = sum(
        1 for item in items
        if item["predicted"] == positive_label and item["actual"] != positive_label
    )
    false_negatives = sum(
        1 for item in items
        if item["predicted"] != positive_label and item["actual"] == positive_label
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
    }


def f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def confusion_matrix(
    results: Iterable[Dict[str, Any]],
    labels: List[str] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Build a confusion matrix from evaluation results.

    Returns a nested dict: {actual_label: {predicted_label: count}}
    """
    if labels is None:
        labels = ["buy", "hold", "sell"]

    matrix = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }

    for item in results:
        actual = item["actual"]
        predicted = item["predicted"]

        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1

    return matrix


def calibration_error(
    predictions: List[Dict[str, float]],
    n_bins: int = 10,
) -> float:
    """
    Calculate expected calibration error.

    predictions: List of dicts with 'confidence' and 'correct' keys.
    """
    if not predictions:
        return 0.0

    bins = [[] for _ in range(n_bins)]

    for pred in predictions:
        conf = pred.get("confidence", 0.5)
        correct = pred.get("correct", 0)

        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    ece = 0.0
    total = len(predictions)

    for bin_data in bins:
        if not bin_data:
            continue

        avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
        avg_acc = sum(a for _, a in bin_data) / len(bin_data)

        ece += len(bin_data) / total * abs(avg_conf - avg_acc)

    return ece