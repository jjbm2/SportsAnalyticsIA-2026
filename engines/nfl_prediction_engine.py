from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from core.market_risk import apply_probability_risk
from services.nfl_data_service import NFLDataService


class NFLPredictionEngine:
    def __init__(self):
        self.data_service = NFLDataService()

    @staticmethod
    def _infer_season(selected_match: dict[str, Any] | None) -> int | None:
        if not selected_match:
            return None
        raw_date = selected_match.get("date")
        if not raw_date:
            return None
        try:
            return int(str(raw_date)[:4])
        except (TypeError, ValueError):
            return None

    def get_team_profiles(
        self,
        home_team_id: int,
        away_team_id: int,
        season: int | None = None,
        league_id: int | None = None,
        force_refresh: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        home_profile = self.data_service.build_team_profile(
            team_id=home_team_id,
            season=season,
            league_id=league_id,
            last=12,
            force_refresh=force_refresh,
        )
        away_profile = self.data_service.build_team_profile(
            team_id=away_team_id,
            season=season,
            league_id=league_id,
            last=12,
            force_refresh=force_refresh,
        )
        return home_profile, away_profile

    def calculate_expected_points(
        self,
        home_profile: dict[str, Any],
        away_profile: dict[str, Any],
    ) -> tuple[float, float]:
        home_expected = max(
            10.0,
            ((home_profile["avg_scored"] * 1.03) + away_profile["avg_allowed"]) / 2,
        )
        away_expected = max(
            10.0,
            ((away_profile["avg_scored"] * 0.99) + home_profile["avg_allowed"]) / 2,
        )
        return home_expected, away_expected

    def run_monte_carlo(
        self,
        home_expected: float,
        away_expected: float,
        home_std: float,
        away_std: float,
        simulations: int,
    ) -> dict[str, Any]:
        home_points = np.random.normal(home_expected, max(home_std, 5.5), simulations).round().astype(int)
        away_points = np.random.normal(away_expected, max(away_std, 5.5), simulations).round().astype(int)

        home_points = np.clip(home_points, 3, 60)
        away_points = np.clip(away_points, 3, 60)

        home_wins = int((home_points > away_points).sum())
        away_wins = int((home_points < away_points).sum())
        draws = int((home_points == away_points).sum())

        totals = home_points + away_points
        over_415 = int((totals > 41.5).sum())
        under_525 = int((totals < 52.5).sum())
        home_over_205 = int((home_points > 20.5).sum())

        score_counter = Counter(zip(home_points.tolist(), away_points.tolist()))
        top_scores = score_counter.most_common(5)

        return {
            "home_win_probability": (home_wins / simulations) * 100,
            "draw_probability": (draws / simulations) * 100,
            "away_win_probability": (away_wins / simulations) * 100,
            "over_415_probability": (over_415 / simulations) * 100,
            "under_525_probability": (under_525 / simulations) * 100,
            "home_over_205_probability": (home_over_205 / simulations) * 100,
            "top_scores": top_scores,
        }

    def analyze_match(
        self,
        selected_match: dict[str, Any],
        simulations: int,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        home_team = selected_match["home"]
        away_team = selected_match["away"]
        home_id = selected_match["home_id"]
        away_id = selected_match["away_id"]
        season = self._infer_season(selected_match)

        home_profile, away_profile = self.get_team_profiles(
            home_team_id=home_id,
            away_team_id=away_id,
            season=season,
            league_id=None,
            force_refresh=force_refresh,
        )

        home_expected, away_expected = self.calculate_expected_points(home_profile, away_profile)

        result = self.run_monte_carlo(
            home_expected=home_expected,
            away_expected=away_expected,
            home_std=home_profile["score_std"],
            away_std=away_profile["score_std"],
            simulations=simulations,
        )

        decisive_total = result["home_win_probability"] + result["away_win_probability"]
        if decisive_total:
            result["home_win_probability"] = (
                result["home_win_probability"] / decisive_total * 100
            )
            result["away_win_probability"] = 100 - result["home_win_probability"]

        markets_df = pd.DataFrame(
            {
                "Mercado": [
                    f"{home_team} gana",
                    f"{away_team} gana",
                    "Over 41.5 puntos",
                    "Under 52.5 puntos",
                    f"{home_team} over 20.5 puntos",
                ],
                "Probabilidad": [
                    f"{result['home_win_probability']:.1f}%",
                    f"{result['away_win_probability']:.1f}%",
                    f"{result['over_415_probability']:.1f}%",
                    f"{result['under_525_probability']:.1f}%",
                    f"{result['home_over_205_probability']:.1f}%",
                ],
                "Confianza": ["Media"] * 5,
                "Riesgo": ["Medio"] * 5,
            }
        )

        markets_to_save = [
            {"market_type": "home_win", "selection": home_team, "probability": result["home_win_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "away_win", "selection": away_team, "probability": result["away_win_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "over_41_5_points", "selection": "Over 41.5 puntos", "probability": result["over_415_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "under_52_5_points", "selection": "Under 52.5 puntos", "probability": result["under_525_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "home_over_20_5_points", "selection": f"{home_team} over 20.5 puntos", "probability": result["home_over_205_probability"], "confidence": "Media", "risk": "Medio"},
        ]
        apply_probability_risk(markets_to_save)
        markets_df["Confianza"] = [item["confidence"] for item in markets_to_save]
        markets_df["Riesgo"] = [item["risk"] for item in markets_to_save]

        return {
            "model_name": "Drive/Points Model + Monte Carlo",
            "summary_cards": [
                {"label": f"Victoria {home_team}", "value": f"{result['home_win_probability']:.1f}%"},
                {"label": f"Victoria {away_team}", "value": f"{result['away_win_probability']:.1f}%"},
            ],
            "extra_metrics": {
                "Puntos esperados local": f"{home_expected:.1f}",
                "Puntos esperados visitante": f"{away_expected:.1f}",
                "Simulaciones": f"{simulations:,}",
                "Modelo": "Drive/Points + Monte Carlo",
                "Historial usado": f"Temporada {home_profile.get('history_season', season)}",
            },
            "markets": markets_df.to_dict(orient="records"),
            "markets_to_save": markets_to_save,
            "context_json": {
                "home_profile": home_profile,
                "away_profile": away_profile,
                "home_expected": home_expected,
                "away_expected": away_expected,
                "top_scores": result["top_scores"],
                "history_is_stale": bool(home_profile.get("history_is_stale") or away_profile.get("history_is_stale")),
            },
        }
