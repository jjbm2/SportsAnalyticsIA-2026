from __future__ import annotations

from typing import Any


LIMITED_WARNING = (
    "Pronóstico orientativo: no hay suficiente historial disponible para este evento. "
    "La estimación usa referencias estadísticas conservadoras, tiene confianza baja "
    "y no es 100% confiable."
)


def build_limited_prediction(
    sport: str,
    home_team: str,
    away_team: str,
    simulations: int,
) -> dict[str, Any]:
    """Return an honest baseline when the specialized data source is unavailable.

    The baseline is deliberately symmetric: without team history it must not invent
    an advantage for the home side. It is kept separate from trained-model output.
    """
    markets = _winner_markets(home_team, away_team)
    markets.extend(_reference_markets(sport, home_team))

    return {
        "model_name": f"{sport} - estimación estadística limitada",
        "summary_cards": [
            {"label": f"Victoria {home_team}", "value": "50.0%"},
            {"label": f"Victoria {away_team}", "value": "50.0%"},
        ],
        "extra_metrics": {
            "Calidad de datos": "Limitada",
            "Confianza": "Baja",
            "Modelo": "Referencia estadística conservadora",
        },
        "markets_to_save": markets,
        "context_json": {
            "mode": "limited_statistical_estimate",
            "limited_history": True,
            "warning": LIMITED_WARNING,
            "simulations_requested": simulations,
        },
        "limited_prediction": True,
        "warning": LIMITED_WARNING,
    }


def _winner_markets(home_team: str, away_team: str) -> list[dict[str, Any]]:
    return [
        _market("home_win", home_team, 50.0),
        _market("away_win", away_team, 50.0),
    ]


def _reference_markets(sport: str, home_team: str) -> list[dict[str, Any]]:
    # These are neutral reference priors, not team-specific claims. Keeping them
    # close to 50% prevents a data outage from being shown as a strong signal.
    definitions = {
        "Béisbol": [
            ("over_8_5_runs", "Over 8.5 carreras", 50.0),
            ("under_10_5_runs", "Under 10.5 carreras", 55.0),
            ("home_over_3_5_runs", f"{home_team} over 3.5 carreras", 50.0),
        ],
        "Basketball": [
            ("over_219_5_points", "Over 219.5 puntos", 50.0),
            ("under_234_5_points", "Under 234.5 puntos", 55.0),
            ("home_over_109_5_points", f"{home_team} over 109.5 puntos", 50.0),
        ],
        "NFL": [
            ("over_41_5_points", "Over 41.5 puntos", 50.0),
            ("under_52_5_points", "Under 52.5 puntos", 55.0),
            ("home_over_20_5_points", f"{home_team} over 20.5 puntos", 50.0),
        ],
    }
    return [_market(*definition) for definition in definitions.get(sport, [])]


def _market(
    market_type: str,
    selection: str,
    probability: float,
) -> dict[str, Any]:
    return {
        "market_type": market_type,
        "selection": selection,
        "probability": probability,
        "confidence": "Baja",
        "confidence_score": 20.0,
        "risk": "Alto",
        "explanation": LIMITED_WARNING,
        "extra_data_json": {
            "limited_history": True,
            "reference_probability": True,
            "warning": LIMITED_WARNING,
        },
    }
