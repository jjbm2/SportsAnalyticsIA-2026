from collections import Counter
from math import exp, factorial
from typing import Any

import numpy as np

from core.market_risk import probability_risk_profile
from services.football_data_service import FootballDataService


class FootballPredictionEngine:
    def __init__(self):
        self.data_service = FootballDataService()

    @staticmethod
    def poisson_probability(lmbda: float, goals: int) -> float:
        return (exp(-lmbda) * (lmbda ** goals)) / factorial(goals)

    def get_team_profiles(
        self,
        home_team_id: int,
        away_team_id: int,
        force_refresh: bool = False,
        provider: str = "api_sports",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        home_profile = self.data_service.build_team_profile(
            team_id=home_team_id,
            last=10,
            force_refresh=force_refresh,
            provider=provider,
        )

        away_profile = self.data_service.build_team_profile(
            team_id=away_team_id,
            last=10,
            force_refresh=force_refresh,
            provider=provider,
        )

        return home_profile, away_profile

    def calculate_expected_goals(
        self,
        home_profile: dict[str, Any],
        away_profile: dict[str, Any],
    ) -> tuple[float, float]:
        home_lambda = max(
            0.2,
            ((home_profile["avg_scored"] * 1.10) + away_profile["avg_conceded"]) / 2,
        )

        away_lambda = max(
            0.2,
            ((away_profile["avg_scored"] * 0.95) + home_profile["avg_conceded"]) / 2,
        )

        return home_lambda, away_lambda

    def run_monte_carlo(
        self,
        home_lambda: float,
        away_lambda: float,
        simulations: int,
    ) -> dict[str, Any]:
        home_scores = np.random.poisson(home_lambda, simulations)
        away_scores = np.random.poisson(away_lambda, simulations)

        home_wins = int((home_scores > away_scores).sum())
        draws = int((home_scores == away_scores).sum())
        away_wins = int((home_scores < away_scores).sum())

        over_25 = int(((home_scores + away_scores) > 2).sum())
        under_35 = int(((home_scores + away_scores) < 4).sum())
        btts = int(((home_scores > 0) & (away_scores > 0)).sum())
        totals = home_scores + away_scores
        goal_lines = {}
        for line in (1.5, 2.5, 3.5, 4.5):
            over_probability = float((totals > line).mean() * 100)
            goal_lines[str(line)] = {
                "over": over_probability,
                "under": 100.0 - over_probability,
            }
        recommended_total = self.select_goal_line(goal_lines)

        score_counter = Counter(zip(home_scores.tolist(), away_scores.tolist()))
        top_scores = score_counter.most_common(5)

        return {
            "home_win_probability": (home_wins / simulations) * 100,
            "draw_probability": (draws / simulations) * 100,
            "away_win_probability": (away_wins / simulations) * 100,
            "over_25_probability": (over_25 / simulations) * 100,
            "under_35_probability": (under_35 / simulations) * 100,
            "btts_probability": (btts / simulations) * 100,
            "home_score_probability": float((home_scores > 0).mean() * 100),
            "away_score_probability": float((away_scores > 0).mean() * 100),
            "goal_lines": goal_lines,
            "recommended_total": recommended_total,
            "top_scores": top_scores,
        }

    @staticmethod
    def select_goal_line(goal_lines: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Select a useful total near 60%, instead of the easiest high-probability line."""
        candidates = []
        for raw_line, probabilities in goal_lines.items():
            for direction in ("over", "under"):
                probability = float(probabilities[direction])
                if probability >= 50.0:
                    candidates.append((abs(probability - 60.0), float(raw_line), direction, probability))
        _, line, direction, probability = min(candidates, key=lambda item: (item[0], item[1]))
        return {
            "line": line,
            "direction": direction,
            "probability": probability,
            "label": f"{direction.title()} {line:.1f} goles",
            "market_type": f"{direction}_{str(line).replace('.', '_')}_goals",
        }

    @classmethod
    def build_market_options(
        cls, home_name: str, away_name: str, home_win: float, draw: float,
        away_win: float, goal_lines: dict[str, dict[str, float]], btts: float,
        home_scores: float, away_scores: float,
    ) -> list[dict[str, Any]]:
        raw = [
            ("Resultado", "home_win", f"{home_name} gana", home_win),
            ("Resultado", "draw", "Empate", draw),
            ("Resultado", "away_win", f"{away_name} gana", away_win),
            ("Doble oportunidad", "double_chance_home_draw", f"{home_name} o empate", home_win + draw),
            ("Doble oportunidad", "double_chance_away_draw", f"{away_name} o empate", away_win + draw),
            ("Doble oportunidad", "double_chance_no_draw", "Cualquiera gana", home_win + away_win),
        ]
        for raw_line in ("1.5", "2.5", "3.5", "4.5"):
            encoded = raw_line.replace(".", "_")
            line = float(raw_line)
            raw.extend([
                ("Total de goles", f"over_{encoded}_goals", f"Over {line:.1f} goles", goal_lines[raw_line]["over"]),
                ("Total de goles", f"under_{encoded}_goals", f"Under {line:.1f} goles", goal_lines[raw_line]["under"]),
            ])
        raw.extend([
            ("Ambos anotan", "btts", "Ambos anotan: Sí", btts),
            ("Ambos anotan", "btts_no", "Ambos anotan: No", 100.0 - btts),
            ("Goles por equipo", "home_over_0_5_goals", f"{home_name} marca", home_scores),
            ("Goles por equipo", "away_over_0_5_goals", f"{away_name} marca", away_scores),
        ])
        return [cls._market(category, market_type, selection, probability) for category, market_type, selection, probability in raw]

    @staticmethod
    def _market(category: str, market_type: str, selection: str, probability: float) -> dict[str, Any]:
        probability = min(100.0, max(0.0, float(probability)))
        confidence, risk = probability_risk_profile(probability)
        return {
            "market_type": market_type, "selection": selection,
            "probability": probability, "confidence": confidence, "risk": risk,
            "extra_data_json": {"category": category, "risk_basis": "simulated_probability"},
        }

    def analyze_match(
        self,
        home_team_id: int,
        away_team_id: int,
        home_team_name: str,
        away_team_name: str,
        simulations: int,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        home_profile, away_profile = self.get_team_profiles(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            force_refresh=force_refresh,
        )

        home_lambda, away_lambda = self.calculate_expected_goals(
            home_profile=home_profile,
            away_profile=away_profile,
        )

        simulation_result = self.run_monte_carlo(
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            simulations=simulations,
        )

        return {
            "home_team": home_team_name,
            "away_team": away_team_name,
            "home_profile": home_profile,
            "away_profile": away_profile,
            "home_lambda": home_lambda,
            "away_lambda": away_lambda,
            **simulation_result,
        }
