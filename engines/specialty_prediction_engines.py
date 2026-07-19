from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.logger import logger
from services.hockey_api import HockeyAPI
from services.mma_api import MMAAPI
from services.historical_season import accessible_history_season


def _probability_from_difference(difference: float, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-difference / max(scale, 0.001)))


def _result(
    selected_match: dict[str, Any],
    simulations: int,
    probability: float,
    model_name: str,
    signals: dict[str, str],
) -> dict[str, Any]:
    probability = min(0.90, max(0.10, probability))
    rng = np.random.default_rng()
    home_wins = int(rng.binomial(simulations, probability))
    home_probability = home_wins / simulations * 100
    away_probability = 100 - home_probability
    home = selected_match["home"]
    away = selected_match["away"]
    favorite = home if home_probability >= away_probability else away
    favorite_probability = max(home_probability, away_probability)
    markets = [
        {"Mercado": f"{home} gana", "Probabilidad": f"{home_probability:.1f}%", "Confianza": "Inicial", "Riesgo": "Medio"},
        {"Mercado": f"{away} gana", "Probabilidad": f"{away_probability:.1f}%", "Confianza": "Inicial", "Riesgo": "Medio"},
    ]
    saved = [
        {"market_type": "home_win", "selection": home, "probability": home_probability, "confidence": "Inicial", "risk": "Medio"},
        {"market_type": "away_win", "selection": away, "probability": away_probability, "confidence": "Inicial", "risk": "Medio"},
    ]
    return {
        "model_name": model_name,
        "summary_cards": [
            {"label": "Favorito", "value": favorite},
            {"label": "Probabilidad estimada", "value": f"{favorite_probability:.1f}%"},
            {"label": "Simulaciones", "value": f"{simulations:,}"},
        ],
        "extra_metrics": signals,
        "markets": markets,
        "markets_to_save": saved,
        "context_json": {"signals": signals, "base_probability": probability, "provider": "api_sports"},
    }


class HockeyPredictionEngine:
    def __init__(self) -> None:
        try:
            self.api_sports = HockeyAPI()
        except (ValueError, OSError) as error:
            logger.warning("Datos especializados de hockey no disponibles: %s", error)
            self.api_sports = None

    def analyze_match(self, selected_match: dict[str, Any], simulations: int, force_refresh: bool = False) -> dict[str, Any]:
        context = selected_match.get("analysis_context") or {}
        requested_season = int(context.get("season") or str(selected_match.get("date"))[:4])
        season = accessible_history_season("hockey", requested_season)
        standings = []
        if self.api_sports is not None:
            try:
                standings = self.api_sports.get(
                    "standings",
                    {"league": context.get("league_id"), "season": season},
                    cache_key=f"standings_{context.get('league_id')}_{season}",
                    force_refresh=force_refresh,
                    max_hours=12,
                ).get("response", [])
            except Exception as error:
                logger.warning("Standings de hockey no disponibles: %s", error)
        home = self._api_sports_stats(standings, selected_match["home_id"])
        away = self._api_sports_stats(standings, selected_match["away_id"])
        home_pct = float(home.get("points_pct", 0.5))
        away_pct = float(away.get("points_pct", 0.5))
        goal_edge = float(home.get("goal_differential", 0)) - float(away.get("goal_differential", 0))
        probability = _probability_from_difference((home_pct - away_pct) * 5 + goal_edge / 60)
        return _result(selected_match, simulations, probability, "NHL Season Form + Monte Carlo", {
            "Rendimiento local": f"{home_pct * 100:.1f}%",
            "Rendimiento visitante": f"{away_pct * 100:.1f}%",
            "Diferencial de goles": f"{goal_edge:+.0f}",
            "Calidad de datos": "Limitada" if not standings else "Historial disponible",
            "Historial usado": f"Temporada {season}",
        })

    @classmethod
    def _api_sports_stats(cls, payload: Any, team_id: Any) -> dict[str, float]:
        for item in cls._walk_dicts(payload):
            team = item.get("team") or {}
            if str(team.get("id")) != str(team_id):
                continue
            games = item.get("games") or {}
            played = float(games.get("played") or games.get("total") or 0)
            wins = games.get("wins") or {}
            win_total = float(wins.get("total") or wins.get("all") or 0) if isinstance(wins, dict) else float(wins or 0)
            goals = item.get("goals") or {}
            goals_for = float(goals.get("for") or 0)
            goals_against = float(goals.get("against") or 0)
            points_pct = float(item.get("percentage") or 0)
            if points_pct <= 0 and played:
                points_pct = win_total / played
            return {"points_pct": points_pct or 0.5, "goal_differential": goals_for - goals_against}
        return {"points_pct": 0.5, "goal_differential": 0.0}

    @classmethod
    def _walk_dicts(cls, value: Any):
        if isinstance(value, dict):
            yield value
            for nested in value.values():
                yield from cls._walk_dicts(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from cls._walk_dicts(nested)


class MMAPredictionEngine:
    def analyze_match(self, selected_match: dict[str, Any], simulations: int, force_refresh: bool = False) -> dict[str, Any]:
        context = selected_match.get("analysis_context") or {}
        home = context.get("home_profile") or {}
        away = context.get("away_profile") or {}
        if selected_match.get("provider") == "api_sports":
            try:
                api = MMAAPI()
                home = self._merge_record(home, api.get_fighter_record(selected_match["home_id"], force_refresh))
                away = self._merge_record(away, api.get_fighter_record(selected_match["away_id"], force_refresh))
            except Exception as error:
                logger.warning("Récord especializado de MMA no disponible: %s", error)
        home_total = max(1, int(home.get("record_wins") or 0) + int(home.get("record_losses") or 0))
        away_total = max(1, int(away.get("record_wins") or 0) + int(away.get("record_losses") or 0))
        home_rate = int(home.get("record_wins") or 0) / home_total
        away_rate = int(away.get("record_wins") or 0) / away_total
        reach_edge = float(home.get("reach_inches") or 0) - float(away.get("reach_inches") or 0)
        probability = _probability_from_difference((home_rate - away_rate) * 5 + reach_edge / 20)
        has_records = any(home.get(key) or away.get(key) for key in ("record_wins", "record_losses"))
        return _result(selected_match, simulations, probability, "MMA Record Profile + Monte Carlo", {
            "Récord peleador 1": f"{home.get('record_wins', 0)}-{home.get('record_losses', 0)}",
            "Récord peleador 2": f"{away.get('record_wins', 0)}-{away.get('record_losses', 0)}",
            "División": str(context.get("weight_class") or "No disponible"),
            "Calidad de datos": "Historial disponible" if has_records else "Limitada",
        })

    @staticmethod
    def _merge_record(profile: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        total = record.get("total") or {}
        merged = dict(profile)
        merged["record_wins"] = total.get("win") or 0
        merged["record_losses"] = total.get("loss") or 0
        reach = str(profile.get("reach") or profile.get("reach_inches") or "0")
        try:
            merged["reach_inches"] = float(reach.replace("'", "").replace('"', "").strip())
        except ValueError:
            merged["reach_inches"] = 0
        return merged
