from __future__ import annotations

from typing import Any


def probability_risk_profile(probability: float) -> tuple[str, str]:
    """Map event probability to transparent confidence and estimated risk labels."""
    value = min(100.0, max(0.0, float(probability)))
    if value >= 75:
        return "Alta", "Bajo"
    if value >= 65:
        return "Media-Alta", "Medio"
    if value >= 58:
        return "Media", "Medio-Alto"
    if value >= 52:
        return "Baja", "Alto"
    return "Muy baja", "Muy alto"


def apply_probability_risk(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for market in markets:
        confidence, risk = probability_risk_profile(float(market.get("probability", 0.0)))
        market["confidence"] = confidence
        market["risk"] = risk
    return markets
