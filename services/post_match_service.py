from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.game_status import extract_final_score, is_finished_status
from core.logger import logger
from core.paths import CACHE_DIR
from database.post_match_review_repository import PostMatchReviewRepository
from database.model_error_analysis_repository import ModelErrorAnalysisRepository
from database.prediction_repository import PredictionRepository
from machine_learning.evaluation.post_match_evaluator import evaluate_markets
from machine_learning.evaluation.model_error_analyzer import build_error_rows
from services.mma_api import MMAAPI


SPORT_CACHE_NAMES = {
    "football": "Fútbol",
    "sportmonks_football": "Fútbol",
    "baseball": "Béisbol",
    "basketball": "Basketball",
    "balldontlie_nba": "Basketball",
    "nfl": "NFL",
    "hockey": "Hockey",
    "mma": "MMA",
    "formula1": "Fórmula 1",
}


class PostMatchService:
    def __init__(self) -> None:
        self.predictions = PredictionRepository()
        self.reviews = PostMatchReviewRepository()
        self.error_analysis = ModelErrorAnalysisRepository()

    def process_games(self, sport: str, games: list[dict[str, Any]]) -> int:
        processed = 0
        for game in games:
            raw_status = (game.get("fixture") or {}).get("status") or game.get("status")
            if not game.get("is_finished") and not is_finished_status(raw_status):
                continue
            home_score = game.get("home_score")
            away_score = game.get("away_score")
            if home_score is None or away_score is None:
                home_score, away_score = extract_final_score(game, sport)
            processed += self._evaluate_match(
                sport=sport,
                match_id=game.get("game_id"),
                home_score=home_score,
                away_score=away_score,
            )
        return processed

    def process_cached_results(self) -> int:
        self.error_analysis.backfill_existing_reviews()
        processed = 0
        for cache_name, sport in SPORT_CACHE_NAMES.items():
            cache_dir = CACHE_DIR / cache_name
            if not cache_dir.exists():
                continue
            for cache_file in cache_dir.glob("*.json"):
                processed += self._process_cache_file(cache_file, sport)
        return processed

    def _process_cache_file(self, cache_file: Path, sport: str) -> int:
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0

        if sport == "Fórmula 1" and isinstance(payload, dict):
            return self._process_formula1_payload(payload)

        processed = 0
        for game in self._payload_games(payload):
            if not isinstance(game, dict):
                continue
            if sport == "MMA":
                game = MMAAPI._normalize(game)
            fixture = game.get("fixture") or {}
            status = fixture.get("status") if sport == "Fútbol" else game.get("status")
            if not is_finished_status(status):
                continue
            match_id = fixture.get("id") if sport == "Fútbol" else game.get("id")
            home_score, away_score = extract_final_score(game, sport)
            processed += self._evaluate_match(
                sport=sport,
                match_id=match_id,
                home_score=home_score,
                away_score=away_score,
            )
        return processed

    @staticmethod
    def _payload_games(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("response", "data", "games"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []

    def _process_formula1_payload(self, payload: dict[str, Any]) -> int:
        race_table = ((payload.get("MRData") or {}).get("RaceTable") or {})
        season = race_table.get("season")
        processed = 0
        for race in race_table.get("Races") or []:
            results = race.get("Results") or []
            if not results:
                continue
            ordered = sorted(results, key=lambda item: int(item.get("position") or 999))
            podium = [self._driver_name(item) for item in ordered[:3]]
            podium = [name for name in podium if name]
            if not podium:
                continue
            race_season = race.get("season") or season
            match_id = f"f1:{race_season}:{race.get('round')}"
            normalized_podium = {self._normalize_selection(name) for name in podium}
            processed += self._evaluate_match(
                sport="Fórmula 1",
                match_id=match_id,
                home_score=0,
                away_score=0,
                selection_outcomes={
                    "f1_win": {self._normalize_selection(podium[0])},
                    "f1_podium": normalized_podium,
                },
                actual_outcome=f"winner:{podium[0]}",
            )
        return processed

    @staticmethod
    def _driver_name(result: dict[str, Any]) -> str:
        driver = result.get("Driver") or {}
        return " ".join(
            part for part in (driver.get("givenName"), driver.get("familyName")) if part
        ).strip()

    @staticmethod
    def _normalize_selection(value: Any) -> str:
        return " ".join(str(value or "").casefold().split())

    def _evaluate_match(
        self,
        sport: str,
        match_id: Any,
        home_score: Any,
        away_score: Any,
        selection_outcomes: dict[str, set[str]] | None = None,
        actual_outcome: str | None = None,
    ) -> int:
        if match_id is None or home_score is None or away_score is None:
            return 0

        processed = 0
        try:
            runs = self.predictions.list_runs_by_match_id(str(match_id), sport=sport)
            for run in runs:
                evaluation = self.reviews.get_review(run["id"])
                if evaluation is None:
                    markets = self.predictions.list_markets_by_run(run["id"])
                    evaluation = evaluate_markets(
                        markets,
                        home_score,
                        away_score,
                        selection_outcomes=selection_outcomes,
                        actual_outcome=actual_outcome,
                    )
                    review_id = self.reviews.save_review(
                        prediction_run_id=run["id"],
                        match_id=str(match_id),
                        sport=sport,
                        home_score=home_score,
                        away_score=away_score,
                        evaluation=evaluation,
                    )
                    processed += 1
                else:
                    review_id = int(evaluation["id"])
                self.error_analysis.save_rows(build_error_rows(run, review_id, evaluation))
        except Exception as error:
            logger.exception(
                "No se pudo evaluar el partido finalizado %s: %s",
                match_id,
                error,
            )
        return processed
