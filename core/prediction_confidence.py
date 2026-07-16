from __future__ import annotations

from typing import Any


MARKET_SIGNAL = {
    "home_win": ("home_win", "home_win_probability", "result"),
    "draw": ("draw", "draw_probability", "result"),
    "away_win": ("away_win", "away_win_probability", "result"),
    "over_2_5_goals": ("over25", "over_25_probability", "over_2_5"),
    "btts": ("btts", "btts_probability", "btts"),
}


def enrich_football_markets(
    markets: list[dict[str, Any]],
    features: dict[str, Any],
    ml_probabilities: dict[str, float],
    statistical_probabilities: dict[str, Any],
    quality_gate: dict[str, bool],
) -> list[dict[str, Any]]:
    home_matches = float(features.get("home_matches_played", 0.0))
    away_matches = float(features.get("away_matches_played", 0.0))
    sample_score = min(100.0, min(home_matches, away_matches) / 20.0 * 100.0)
    consistency = max(0.0, 100.0 - (
        float(features.get("home_scored_std", 1.5))
        + float(features.get("away_scored_std", 1.5))
    ) / 2.0 / 2.5 * 100.0)

    for market in markets:
        probability = float(market.get("probability", 0.0))
        probability_strength = min(100.0, abs(probability - 50.0) * 2.0)
        signal = MARKET_SIGNAL.get(str(market.get("market_type")))
        agreement = 65.0
        model_quality = 60.0
        if signal:
            ml_key, statistical_key, gate_key = signal
            model_quality = 90.0 if quality_gate.get(gate_key) else 60.0
            if ml_key in ml_probabilities and statistical_key in statistical_probabilities:
                distance = abs(
                    float(ml_probabilities[ml_key])
                    - float(statistical_probabilities[statistical_key])
                )
                agreement = max(0.0, 100.0 - distance * 2.0)
        score = round(
            0.30 * probability_strength
            + 0.25 * sample_score
            + 0.15 * consistency
            + 0.20 * agreement
            + 0.10 * model_quality,
            1,
        )
        explanation = explain_football_market(market, features, home_matches, away_matches)
        market["confidence_score"] = score
        market["confidence"] = _confidence_label(score)
        market["explanation"] = explanation
        extra = dict(market.get("extra_data_json") or {})
        extra.update({
            "confidence_score": score,
            "explanation": explanation,
            "confidence_components": {
                "probability_strength": round(probability_strength, 1),
                "historical_data": round(sample_score, 1),
                "consistency": round(consistency, 1),
                "model_agreement": round(agreement, 1),
                "model_quality": round(model_quality, 1),
            },
        })
        market["extra_data_json"] = extra
    return markets


def explain_football_market(
    market: dict[str, Any], features: dict[str, Any], home_matches: float, away_matches: float
) -> str:
    market_type = str(market.get("market_type") or "")
    if min(home_matches, away_matches) < 5:
        return (
            f"Confianza limitada: solo hay {int(home_matches)} partidos del local "
            f"y {int(away_matches)} del visitante en el historial usado."
        )
    if market_type in {"home_win", "draw", "away_win"}:
        points_diff = float(features.get("diff_points_last5", 0.0))
        attack_diff = float(features.get("recent_home_attack_vs_away_defense", 0.0))
        advantage = float(features.get("real_home_advantage", 0.0))
        return (
            f"La forma reciente difiere {points_diff:+.1f} puntos; el ataque local frente a la defensa "
            f"visitante marca {attack_diff:+.2f} goles y la ventaja local histórica es {advantage:+.2f} puntos por partido."
        )
    if market_type.startswith(("over_", "under_")):
        expected = (
            float(features.get("home_avg_scored_last5", 0.0))
            + float(features.get("away_avg_scored_last5", 0.0))
        )
        conceded = (
            float(features.get("home_avg_conceded_last5", 0.0))
            + float(features.get("away_avg_conceded_last5", 0.0))
        )
        return f"Los equipos anotan {expected:.2f} y reciben {conceded:.2f} goles combinados por partido en su forma reciente."
    if market_type.startswith("btts"):
        rate = (
            float(features.get("home_btts_rate_last5", 0.0))
            + float(features.get("away_btts_rate_last5", 0.0))
        ) / 2.0
        return f"Ambos equipos marcaron en {rate * 100:.0f}% de sus partidos recientes combinados."
    clean_sheets = (
        float(features.get("home_clean_sheet_rate", 0.0))
        + float(features.get("away_clean_sheet_rate", 0.0))
    ) / 2.0
    return f"La señal considera forma, producción ofensiva y una tasa combinada de porterías en cero de {clean_sheets * 100:.0f}%."


def _confidence_label(score: float) -> str:
    if score >= 75.0:
        return "Alta"
    if score >= 60.0:
        return "Media-Alta"
    if score >= 45.0:
        return "Media"
    return "Baja"
