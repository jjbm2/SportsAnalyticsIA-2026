from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.logger import logger
from database.prediction_repository import PredictionRepository
from machine_learning.predictors.baseball_predictor import BaseballPredictor
from machine_learning.predictors.basketball_predictor import BasketballPredictor
from machine_learning.predictors.football_predictor import FootballPredictor
from machine_learning.predictors.formula1_predictor import Formula1Predictor
from machine_learning.predictors.nfl_predictor import NFLPredictor


class ShadowValidationService:
    model_root = Path("machine_learning/models_store")
    sport_dirs = {
        "Fútbol": "football",
        "Béisbol": "baseball",
        "Basketball": "basketball",
        "NFL": "nfl",
        "Fórmula 1": "formula1",
    }

    def run(self, sport: str, selected_match: dict[str, Any], simulations: int) -> int | None:
        match_id = selected_match.get("game_id")
        candidate = self.find_candidate(sport)
        if match_id is None or candidate is None:
            return None
        metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
        version = str(metadata["model_version"])
        repository = PredictionRepository()
        if repository.shadow_run_exists(sport, str(match_id), version):
            return None
        try:
            result = self._analyze(sport, candidate, selected_match, simulations)
            context = dict(result.get("context_json") or {})
            context.update({"model_version": version, "shadow": True})
            return repository.save_prediction_run(
                sport=sport,
                match_id=str(match_id),
                home_team=selected_match["home"],
                away_team=selected_match["away"],
                model_name=f'{result["model_name"]} · candidato {version}',
                simulations=simulations,
                markets=result.get("markets_to_save", []),
                status="shadow",
                context_json=context,
            )
        except Exception as error:
            logger.warning("Validación en sombra omitida para %s: %s", sport, error)
            return None

    def find_candidate(self, sport: str) -> Path | None:
        folder = self.sport_dirs.get(sport)
        if not folder:
            return None
        active_version = self._active_version(folder)
        candidates: list[tuple[float, str, Path]] = []
        for metadata_path in (self.model_root / "versions" / folder).glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                metadata.get("status") == "candidate_qualified"
                and str(metadata.get("model_version")) != active_version
            ):
                candidates.append((self._candidate_score(metadata), metadata_path.parent.name, metadata_path.parent))
        return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None

    @staticmethod
    def _candidate_score(metadata: dict[str, Any]) -> float:
        """Rank candidates by discrimination, lift and calibration, not recency."""
        metrics = metadata.get("metrics") or {}
        qualified = set(metadata.get("qualified_markets") or [])
        nested_rows = [
            row for market, row in metrics.items()
            if isinstance(row, dict) and (not qualified or market in qualified)
        ]
        if nested_rows:
            scores = []
            for row in nested_rows:
                try:
                    auc = float(row.get("roc_auc", 0.5))
                    lift = float(row.get("accuracy_lift", 0.0))
                    brier = float(row.get("brier_score", 0.25))
                except (TypeError, ValueError):
                    continue
                scores.append(auc + lift - brier)
            return sum(scores) / len(scores) if scores else float("-inf")

        # Football metadata stores market metrics in a flat structure.
        try:
            result_lift = float(metrics.get("result_accuracy_lift", 0.0))
            over_edge = float(metrics.get("over_auc", 0.5)) - 0.5
            btts_edge = float(metrics.get("btts_auc", 0.5)) - 0.5
        except (TypeError, ValueError):
            return float("-inf")
        score = result_lift if not qualified or "result" in qualified else 0.0
        if "over_2_5" in qualified:
            score += over_edge
        if "btts" in qualified:
            score += btts_edge
        return score

    def _active_version(self, folder: str) -> str:
        path = self.model_root / f"{folder}_metadata.json"
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("model_version"))
        except (OSError, ValueError):
            return ""

    @staticmethod
    def _analyze(sport: str, candidate: Path, match: dict[str, Any], simulations: int) -> dict[str, Any]:
        if sport == "Fútbol":
            return FootballPredictor(candidate).predict_match(
                match["home_id"], match["away_id"], match["home"], match["away"],
                simulations=simulations, provider=match.get("provider", "api_sports"),
            )
        if sport == "Fórmula 1":
            return Formula1Predictor(candidate).analyze_match(match, simulations)
        predictor_class = {
            "Béisbol": BaseballPredictor,
            "Basketball": BasketballPredictor,
            "NFL": NFLPredictor,
        }[sport]
        return predictor_class(candidate).analyze_match(match, simulations)
