from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from services.formula1_api import Formula1API


class Formula1PredictionEngine:
    def __init__(self) -> None:
        self.api = Formula1API()

    def analyze_match(
        self,
        selected_match: dict[str, Any],
        simulations: int,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        season = int(selected_match.get("season") or str(selected_match.get("date"))[:4])
        target_round = int(selected_match.get("round") or 99)
        results = self.api.get_results(season, force_refresh)
        completed = [race for race in results if int(race.get("round") or 0) < target_round]
        if not completed:
            completed = self.api.get_results(season - 1, force_refresh)

        driver_scores: defaultdict[str, list[float]] = defaultdict(list)
        driver_names: dict[str, str] = {}
        constructors: dict[str, str] = {}
        for race in completed[-8:]:
            for result in race.get("Results", []):
                driver = result.get("Driver") or {}
                driver_id = driver.get("driverId")
                if not driver_id:
                    continue
                position = int(result.get("position") or 20)
                points = max(1.0, 26.0 - position)
                driver_scores[driver_id].append(points)
                driver_names[driver_id] = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
                constructors[driver_id] = ((result.get("Constructor") or {}).get("name") or "")

        if len(driver_scores) < 3:
            raise ValueError("No hay resultados históricos suficientes para analizar esta carrera.")

        driver_ids = list(driver_scores)
        strengths = np.array([
            sum(scores[-5:]) / len(scores[-5:]) for scores in driver_scores.values()
        ], dtype=float)
        strengths = np.maximum(strengths, 0.5)
        probabilities = strengths / strengths.sum()
        wins = defaultdict(int)
        podiums = defaultdict(int)
        rng = np.random.default_rng()
        for _ in range(simulations):
            podium = rng.choice(driver_ids, size=3, replace=False, p=probabilities)
            wins[str(podium[0])] += 1
            for driver_id in podium:
                podiums[str(driver_id)] += 1

        ranking = sorted(driver_ids, key=lambda item: wins[item], reverse=True)[:5]
        saved: list[dict[str, Any]] = []
        display_rows: list[dict[str, str]] = []
        for driver_id in ranking:
            win_probability = wins[driver_id] / simulations * 100
            podium_probability = podiums[driver_id] / simulations * 100
            name = driver_names[driver_id]
            display_rows.append({
                "Piloto": name,
                "Equipo": constructors[driver_id],
                "Victoria": f"{win_probability:.1f}%",
                "Podio": f"{podium_probability:.1f}%",
            })
            saved.extend([
                {"market_type": "f1_win", "selection": name, "probability": win_probability, "confidence": "Inicial", "risk": "Alto"},
                {"market_type": "f1_podium", "selection": name, "probability": podium_probability, "confidence": "Inicial", "risk": "Medio-Alto"},
            ])

        favorite = ranking[0]
        return {
            "model_name": "F1 Recent Form + Monte Carlo",
            "summary_cards": [
                {"label": "Favorito", "value": driver_names[favorite]},
                {"label": "Probabilidad de victoria", "value": f"{wins[favorite] / simulations * 100:.1f}%"},
                {"label": "Pilotos evaluados", "value": str(len(driver_ids))},
            ],
            "extra_metrics": {
                "Carreras recientes": str(min(8, len(completed))),
                "Simulaciones": f"{simulations:,}",
                "Modelo": "Forma reciente + Monte Carlo",
            },
            "markets": pd.DataFrame(display_rows).to_dict(orient="records"),
            "markets_to_save": saved,
            "context_json": {
                "season": season,
                "round": target_round,
                "drivers_evaluated": len(driver_ids),
                "recent_races": min(8, len(completed)),
            },
        }
