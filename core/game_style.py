from __future__ import annotations

from typing import Any


DEFENSIVE = "DEFENSIVE"
OFFENSIVE = "OFFENSIVE"
BALANCED = "BALANCED"
CHAOTIC = "CHAOTIC"
UNBALANCED = "UNBALANCED"

STYLE_PRESENTATION = {
    DEFENSIVE: ("Defensivo", ":material/shield:"),
    OFFENSIVE: ("Ofensivo", ":material/local_fire_department:"),
    BALANCED: ("Balanceado", ":material/balance:"),
    CHAOTIC: ("Caótico", ":material/casino:"),
    UNBALANCED: ("Desbalanceado", ":material/emoji_events:"),
}


def classify_game_style(
    features: dict[str, Any],
    match_quality: dict[str, Any] | float,
) -> dict[str, Any]:
    """Classify match context without changing predicted probabilities."""
    quality_score, consistency = _quality_values(match_quality)
    avg_goals = (
        _number(features.get("home_avg_scored"), _number(features.get("home_avg_scored_last5"), 1.0))
        + _number(features.get("away_avg_conceded"), _number(features.get("away_avg_conceded_last5"), 1.0))
        + _number(features.get("away_avg_scored"), _number(features.get("away_avg_scored_last5"), 1.0))
        + _number(features.get("home_avg_conceded"), _number(features.get("home_avg_conceded_last5"), 1.0))
    ) / 2.0
    btts_rate = (
        _number(features.get("home_btts_rate"), _number(features.get("home_btts_rate_last5"), 0.5))
        + _number(features.get("away_btts_rate"), _number(features.get("away_btts_rate_last5"), 0.5))
    ) / 2.0
    variance = (
        _number(features.get("home_scored_std"), 1.0)
        + _number(features.get("away_scored_std"), 1.0)
    ) / 2.0
    level_difference = min(1.0, max(
        abs(_number(features.get("diff_points_last5"))) / 15.0,
        abs(_number(features.get("diff_avg_scored"))) / 2.5,
        abs(_number(features.get("real_home_advantage"))) / 3.0,
    ))

    if variance >= 1.45 or (variance >= 1.15 and btts_rate >= 0.65):
        style = CHAOTIC
        reason = f"La varianza de gol es alta ({variance:.2f}) y reduce la estabilidad del escenario."
    elif level_difference >= 0.50:
        style = UNBALANCED
        reason = f"La diferencia relativa de nivel es amplia ({level_difference:.0%}) entre ambos equipos."
    elif avg_goals <= 2.15 and btts_rate <= 0.50:
        style = DEFENSIVE
        reason = f"El perfil combinado es de {avg_goals:.2f} goles y {btts_rate:.0%} de BTTS."
    elif avg_goals >= 2.90 or (avg_goals >= 2.55 and btts_rate >= 0.62):
        style = OFFENSIVE
        reason = f"El perfil combinado es de {avg_goals:.2f} goles y {btts_rate:.0%} de BTTS."
    else:
        style = BALANCED
        reason = f"Ataque y defensa presentan un perfil intermedio de {avg_goals:.2f} goles."

    label, icon = STYLE_PRESENTATION[style]
    confidence_adjustment = _confidence_adjustment(style, quality_score, consistency)
    return {
        "game_style": style,
        "game_style_label": label,
        "game_style_icon": icon,
        "game_style_explanation": f"Partido {label.lower()}. {reason}",
        "game_style_confidence_adjustment": confidence_adjustment,
        "game_style_components": {
            "average_goals": round(avg_goals, 3),
            "btts_rate": round(btts_rate, 3),
            "goal_variance": round(variance, 3),
            "level_difference": round(level_difference, 3),
            "consistency": round(consistency, 3),
            "match_quality_score": round(quality_score, 3),
        },
    }


def apply_game_style_to_markets(
    markets: list[dict[str, Any]], style_result: dict[str, Any]
) -> list[dict[str, Any]]:
    """Adjust interpretation confidence only; probabilities stay untouched."""
    adjustment = float(style_result.get("game_style_confidence_adjustment", 0.0))
    prefix = str(style_result.get("game_style_explanation") or "").strip()
    for market in markets:
        original = _number(market.get("confidence_score"), 0.0)
        adjusted = round(max(0.0, min(100.0, original + adjustment)), 1)
        market["confidence_score"] = adjusted
        market["confidence"] = _confidence_label(adjusted)
        market["explanation"] = f"{prefix} {market.get('explanation', '')}".strip()
        extra = dict(market.get("extra_data_json") or {})
        extra.update({
            "game_style": style_result.get("game_style"),
            "game_style_explanation": prefix,
            "confidence_before_game_style": original,
            "confidence_score": adjusted,
            "explanation": market["explanation"],
        })
        market["extra_data_json"] = extra
    return markets


def _quality_values(match_quality: dict[str, Any] | float) -> tuple[float, float]:
    if isinstance(match_quality, dict):
        score = _number(match_quality.get("match_quality_score"), 0.0)
        components = match_quality.get("match_quality_components") or {}
        consistency = _number(components.get("consistency"), 0.5)
        return score, consistency
    return _number(match_quality), 0.5


def _confidence_adjustment(style: str, quality: float, consistency: float) -> float:
    if style == CHAOTIC:
        return -10.0
    if style == UNBALANCED:
        return 3.0 if quality >= 0.60 else 0.0
    if style == BALANCED and consistency >= 0.70:
        return 3.0
    if style in {DEFENSIVE, OFFENSIVE} and quality >= 0.60:
        return 2.0
    return 0.0


def _confidence_label(score: float) -> str:
    if score >= 75.0:
        return "Alta"
    if score >= 60.0:
        return "Media-Alta"
    if score >= 45.0:
        return "Media"
    return "Baja"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
