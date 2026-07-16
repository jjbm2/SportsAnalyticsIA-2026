from __future__ import annotations

from statistics import fmean
from typing import Any


def calculate_match_quality(
    markets: list[dict[str, Any]],
    features: dict[str, Any],
    *,
    quality_gate: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Summarize prediction reliability without changing model probabilities."""
    quality_gate = quality_gate or {}
    components = [_market_components(market) for market in markets]

    confidence = _mean(
        [float(market.get("confidence_score", 0.0)) / 100.0 for market in markets]
    )
    historical_data = min(
        1.0,
        min(
            _number(features.get("home_matches_played")),
            _number(features.get("away_matches_played")),
        ) / 20.0,
    )
    variance = (
        _number(features.get("home_scored_std"), 1.5)
        + _number(features.get("away_scored_std"), 1.5)
    ) / 2.0
    consistency = max(0.0, min(1.0, 1.0 - variance / 2.5))
    agreement = _mean([item.get("model_agreement", 0.65) / 100.0 for item in components])

    strengths = [abs(_number(market.get("probability"), 50.0) - 50.0) / 50.0 for market in markets]
    result_probabilities = [
        _number(market.get("probability"))
        for market in markets
        if str(market.get("market_type")) in {"home_win", "draw", "away_win"}
    ]
    result_probabilities.sort(reverse=True)
    result_margin = (
        (result_probabilities[0] - result_probabilities[1]) / 100.0
        if len(result_probabilities) >= 2 else 0.0
    )
    stability = min(1.0, 0.70 * _mean(strengths) + 0.30 * min(1.0, result_margin * 3.0))

    score = max(0.0, min(1.0, (
        0.30 * confidence
        + 0.20 * historical_data
        + 0.15 * consistency
        + 0.20 * agreement
        + 0.15 * stability
    )))
    # When no trained market passed its quality gate, agreement is only a
    # statistical prior and cannot justify a high-quality label by itself.
    if quality_gate and not any(quality_gate.values()):
        score = min(score, 0.69)

    rounded_score = round(score, 3)
    return {
        "match_quality_score": rounded_score,
        "match_quality_label": quality_label(rounded_score),
        "match_quality_explanation": quality_explanation(
            rounded_score, historical_data, consistency, agreement
        ),
        "match_quality_components": {
            "confidence": round(confidence, 3),
            "historical_data": round(historical_data, 3),
            "consistency": round(consistency, 3),
            "model_agreement": round(agreement, 3),
            "prediction_stability": round(stability, 3),
        },
    }


def quality_label(score: float) -> str:
    if score >= 0.75:
        return "Partido recomendado"
    if score >= 0.60:
        return "Calidad media"
    return "Calidad baja"


def quality_explanation(
    score: float, historical_data: float, consistency: float, agreement: float
) -> str:
    if score >= 0.75:
        return "Alta calidad de predicción por cobertura histórica, equipos consistentes y buen acuerdo entre modelos."
    if score >= 0.60:
        weak = min(
            ((historical_data, "cobertura histórica"), (consistency, "consistencia"), (agreement, "acuerdo entre modelos")),
            key=lambda item: item[0],
        )[1]
        return f"Calidad media: la señal es utilizable, aunque la {weak} todavía limita la confianza."
    return "Calidad baja: los datos, la consistencia o el acuerdo entre modelos no sostienen una predicción confiable."


def _market_components(market: dict[str, Any]) -> dict[str, float]:
    extra = market.get("extra_data_json") or {}
    values = extra.get("confidence_components") or {}
    return {key: _number(value) for key, value in values.items()}


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
