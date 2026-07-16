from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_error_rows(
    run: dict[str, Any], review_id: int, evaluation: dict[str, Any]
) -> list[dict[str, Any]]:
    context = run.get("context_json") or {}
    league = _league(context)
    rows = []
    for detail in evaluation.get("market_evaluations") or []:
        probability = float(detail["probability"])
        actual = bool(detail["actual"])
        predicted = bool(detail.get("is_pick", probability >= 50.0))
        correct = predicted == actual
        rows.append({
            "prediction_run_id": run["id"],
            "post_match_review_id": review_id,
            "match_id": str(run.get("match_id") or ""),
            "sport": run["sport"],
            "league": league,
            "market_type": detail["market_type"],
            "match_type": _match_type(detail),
            "selection": detail["selection"],
            "predicted_probability": probability,
            "predicted_event": predicted,
            "actual_event": actual,
            "probability_difference": abs((probability / 100.0) - float(actual)),
            "brier_score": float(detail["brier_score"]),
            "correct": correct,
            "pattern_tags": _tags(probability, predicted, actual, correct),
        })
    return rows


def detect_failure_patterns(rows: list[dict[str, Any]], minimum_samples: int = 3) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["league"], row["market_type"], row["match_type"])].append(row)
    patterns = []
    for (league, market, match_type), items in groups.items():
        if len(items) < minimum_samples:
            continue
        failures = [item for item in items if not item["correct"]]
        error_rate = len(failures) / len(items)
        average_probability = sum(float(item["predicted_probability"]) for item in items) / len(items)
        average_difference = sum(float(item["probability_difference"]) for item in items) / len(items)
        labels = []
        if error_rate >= 0.60:
            labels.append("systematic_failure")
        if error_rate >= 0.40 and average_probability >= 65.0:
            labels.append("overconfidence")
        false_positives = sum(item["predicted_event"] and not item["actual_event"] for item in items)
        false_negatives = sum(not item["predicted_event"] and item["actual_event"] for item in items)
        if false_positives / len(items) >= 0.40:
            labels.append("false_positive_bias")
        if false_negatives / len(items) >= 0.40:
            labels.append("false_negative_bias")
        if labels:
            patterns.append({
                "league": league, "market_type": market, "match_type": match_type,
                "samples": len(items), "failures": len(failures), "error_rate": error_rate,
                "average_probability": average_probability,
                "average_difference": average_difference, "patterns": labels,
            })
    return sorted(patterns, key=lambda item: (-item["error_rate"], -item["samples"]))


def _league(context: dict[str, Any]) -> str:
    analysis = context.get("analysis_context") or {}
    match_metadata = context.get("match_metadata") or {}
    league = (
        context.get("league") or analysis.get("league") or analysis.get("league_name")
        or match_metadata.get("league") or match_metadata.get("league_name")
    )
    if isinstance(league, dict):
        league = league.get("name") or league.get("id")
    return str(
        league or analysis.get("league_id") or context.get("league_id")
        or match_metadata.get("league_id") or "unknown"
    )


def _match_type(detail: dict[str, Any]) -> str:
    probability = float(detail["probability"])
    if probability >= 70.0:
        return "strong_favorite"
    if 45.0 <= probability <= 55.0:
        return "balanced"
    return "standard"


def _tags(probability: float, predicted: bool, actual: bool, correct: bool) -> list[str]:
    if correct:
        return []
    tags = ["false_positive" if predicted and not actual else "false_negative"]
    if probability >= 65.0:
        tags.append("high_confidence_miss")
    if abs((probability / 100.0) - float(actual)) >= 0.65:
        tags.append("large_probability_error")
    return tags
