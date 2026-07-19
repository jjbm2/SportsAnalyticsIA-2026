from __future__ import annotations

from collections import Counter
from math import exp, factorial
from typing import Any

import numpy as np
import pandas as pd

from core.market_risk import apply_probability_risk
from services.baseball_data_service import BaseballDataService


class BaseballPredictionEngine:
    def __init__(self):
        self.data_service = BaseballDataService()

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

    @staticmethod
    def poisson_probability(lmbda: float, value: int) -> float:
        return (exp(-lmbda) * (lmbda ** value)) / factorial(value)

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

    def calculate_expected_runs(
        self,
        home_profile: dict[str, Any],
        away_profile: dict[str, Any],
    ) -> tuple[float, float]:
        home_lambda = max(
            1.5,
            ((home_profile["avg_scored"] * 1.04) + away_profile["avg_allowed"]) / 2,
        )
        away_lambda = max(
            1.5,
            ((away_profile["avg_scored"] * 0.98) + home_profile["avg_allowed"]) / 2,
        )
        return home_lambda, away_lambda

    def run_monte_carlo(
        self,
        home_lambda: float,
        away_lambda: float,
        simulations: int,
    ) -> dict[str, Any]:
        home_runs = np.random.poisson(home_lambda, simulations)
        away_runs = np.random.poisson(away_lambda, simulations)

        home_wins = int((home_runs > away_runs).sum())
        away_wins = int((home_runs < away_runs).sum())
        draws = int((home_runs == away_runs).sum())

        totals = home_runs + away_runs
        over_85 = int((totals > 8.5).sum())
        under_105 = int((totals < 10.5).sum())
        home_over_35 = int((home_runs > 3.5).sum())

        score_counter = Counter(zip(home_runs.tolist(), away_runs.tolist()))
        top_scores = score_counter.most_common(5)

        return {
            "home_win_probability": (home_wins / simulations) * 100,
            "draw_probability": (draws / simulations) * 100,
            "away_win_probability": (away_wins / simulations) * 100,
            "over_85_probability": (over_85 / simulations) * 100,
            "under_105_probability": (under_105 / simulations) * 100,
            "home_over_35_probability": (home_over_35 / simulations) * 100,
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

        home_lambda, away_lambda = self.calculate_expected_runs(home_profile, away_profile)
        result = self.run_monte_carlo(home_lambda, away_lambda, simulations)
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
                    "Over 8.5 carreras",
                    "Under 10.5 carreras",
                    f"{home_team} over 3.5 carreras",
                ],
                "Probabilidad": [
                    f"{result['home_win_probability']:.1f}%",
                    f"{result['away_win_probability']:.1f}%",
                    f"{result['over_85_probability']:.1f}%",
                    f"{result['under_105_probability']:.1f}%",
                    f"{result['home_over_35_probability']:.1f}%",
                ],
                "Confianza": ["Media"] * 5,
                "Riesgo": ["Medio"] * 5,
            }
        )

        markets_to_save = [
            {"market_type": "home_win", "selection": home_team, "probability": result["home_win_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "away_win", "selection": away_team, "probability": result["away_win_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "over_8_5_runs", "selection": "Over 8.5 carreras", "probability": result["over_85_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "under_10_5_runs", "selection": "Under 10.5 carreras", "probability": result["under_105_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "home_over_3_5_runs", "selection": f"{home_team} over 3.5 carreras", "probability": result["home_over_35_probability"], "confidence": "Media", "risk": "Medio"},
        ]
        apply_probability_risk(markets_to_save)
        markets_df["Confianza"] = [item["confidence"] for item in markets_to_save]
        markets_df["Riesgo"] = [item["risk"] for item in markets_to_save]

        return {
            "model_name": "Runs Model + Monte Carlo",
            "summary_cards": [
                {"label": f"Victoria {home_team}", "value": f"{result['home_win_probability']:.1f}%"},
                {"label": f"Victoria {away_team}", "value": f"{result['away_win_probability']:.1f}%"},
            ],
            "extra_metrics": {
                "Carreras esperadas local": f"{home_lambda:.2f}",
                "Carreras esperadas visitante": f"{away_lambda:.2f}",
                "Simulaciones": f"{simulations:,}",
                "Modelo": "Runs + Monte Carlo",
                "Historial usado": f"Temporada {home_profile.get('history_season', season)}",
            },
            "markets": markets_df.to_dict(orient="records"),
            "markets_to_save": markets_to_save,
            "context_json": {
                "home_profile": home_profile,
                "away_profile": away_profile,
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "top_scores": result["top_scores"],
                "history_is_stale": bool(home_profile.get("history_is_stale") or away_profile.get("history_is_stale")),
            },
        }
