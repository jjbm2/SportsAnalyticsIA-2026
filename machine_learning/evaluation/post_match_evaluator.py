from __future__ import annotations

import re
from typing import Any


def evaluate_markets(
    markets: list[dict[str, Any]],
    home_score: float,
    away_score: float,
    selection_outcomes: dict[str, set[str]] | None = None,
    actual_outcome: str | None = None,
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []

    for market in markets:
        market_type = market["market_type"]
        if selection_outcomes and market_type in selection_outcomes:
            selected = _normalize_selection(market["selection"])
            outcome = selected in selection_outcomes[market_type]
        else:
            outcome = _market_outcome(
                market_type=market_type,
                home_score=home_score,
                away_score=away_score,
            )
        if outcome is None:
            continue

        probability = min(100.0, max(0.0, float(market["probability"]))) / 100
        actual = bool(outcome)
        predicted = probability >= 0.5
        evaluations.append(
            {
                "market_type": market["market_type"],
                "selection": market["selection"],
                "probability": probability * 100,
                "actual": actual,
                "is_pick": predicted,
                "correct": predicted == actual,
                "absolute_error": abs(probability - float(actual)),
                "brier_score": (probability - float(actual)) ** 2,
            }
        )

    evaluated = len(evaluations)
    correct = sum(bool(item["correct"]) for item in evaluations)
    calibration_count = len(evaluations)
    return {
        "actual_outcome": actual_outcome or _actual_outcome(home_score, away_score),
        "evaluated_markets": evaluated,
        "correct_markets": correct,
        "accuracy": (correct / evaluated) if evaluated else None,
        "calibrated_markets": calibration_count,
        "mean_absolute_error": (
            sum(item["absolute_error"] for item in evaluations) / calibration_count
            if calibration_count else None
        ),
        "mean_brier_score": (
            sum(item["brier_score"] for item in evaluations) / calibration_count
            if calibration_count else None
        ),
        "market_evaluations": evaluations,
    }


def _normalize_selection(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _actual_outcome(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def _market_outcome(
    market_type: str,
    home_score: float,
    away_score: float,
) -> bool | None:
    total = home_score + away_score
    dynamic_total = re.fullmatch(r"(over|under)_(\d+)_(\d+)_goals", market_type)
    if dynamic_total:
        direction, whole, decimal = dynamic_total.groups()
        line = float(f"{whole}.{decimal}")
        return total > line if direction == "over" else total < line
    direct = {
        "home_win": home_score > away_score,
        "draw": home_score == away_score,
        "away_win": home_score < away_score,
        "btts": home_score > 0 and away_score > 0,
        "btts_no": not (home_score > 0 and away_score > 0),
        "double_chance_home_draw": home_score >= away_score,
        "double_chance_away_draw": away_score >= home_score,
        "double_chance_no_draw": home_score != away_score,
        "home_over_0_5_goals": home_score > 0.5,
        "away_over_0_5_goals": away_score > 0.5,
        "over_2_5_goals": total > 2.5,
        "under_3_5_goals": total < 3.5,
        "over_8_5_runs": total > 8.5,
        "under_10_5_runs": total < 10.5,
        "home_over_3_5_runs": home_score > 3.5,
        "over_219_5_points": total > 219.5,
        "under_234_5_points": total < 234.5,
        "home_over_109_5_points": home_score > 109.5,
        "over_41_5_points": total > 41.5,
        "under_52_5_points": total < 52.5,
        "home_over_20_5_points": home_score > 20.5,
    }
    return direct.get(market_type)
