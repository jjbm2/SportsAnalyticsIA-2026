from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.database import Base
from database.models import PredictionMarket, PredictionRun
from database.prediction_repository import PredictionRepository
from machine_learning.predictors.football_predictor import FootballPredictor


class FootballEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, future=True)
        self.session_patch = patch(
            "database.prediction_repository.get_session",
            self.session_factory,
        )
        self.session_patch.start()
        self.repository = PredictionRepository()

    def tearDown(self) -> None:
        self.session_patch.stop()
        self.engine.dispose()

    @staticmethod
    def _markets() -> list[dict]:
        return [
            {
                "market_type": "home_win",
                "selection": "Local",
                "probability": 57.5,
                "confidence": "Media-Alta",
                "risk": "Medio",
                "extra_data_json": {"category": "Resultado"},
            },
            {
                "market_type": "under_3_5_goals",
                "selection": "Menos de 3.5 goles",
                "probability": 71.0,
                "confidence": "Alta",
                "risk": "Bajo",
                "extra_data_json": {"category": "Goles"},
            },
        ]

    def test_predictor_result_persists_complete_run_and_history(self) -> None:
        context = {
            "quality_gate": {"result": True, "over_2_5": False, "btts": False},
            "ml_probabilities": {"home_win": 60.0, "draw": 24.0, "away_win": 16.0},
            "analysis_type": "hybrid_ai",
        }
        run_id = self.repository.save_prediction_run(
            sport="Fútbol",
            match_id="fixture-101",
            home_team="Local",
            away_team="Visitante",
            model_name="Football ML + Poisson + Monte Carlo",
            simulations=10_000,
            markets=self._markets(),
            context_json=context,
        )

        history = self.repository.list_recent_runs(limit=10)
        markets = self.repository.list_markets_by_run(run_id)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["id"], run_id)
        self.assertEqual(history[0]["match_id"], "fixture-101")
        self.assertEqual(history[0]["context_json"], context)
        self.assertEqual(len(markets), 2)
        self.assertEqual({market["run_id"] for market in markets}, {run_id})

    def test_fallback_result_is_persisted_with_statistical_context(self) -> None:
        missing_models = Path("data/test_missing_football_model")
        with self.assertRaises(FileNotFoundError):
            FootballPredictor(model_dir=missing_models)

        context = {
            "home_lambda": 1.42,
            "away_lambda": 1.03,
            "analysis_type": "statistical",
        }
        run_id = self.repository.save_prediction_run(
            sport="Fútbol",
            match_id="fixture-fallback",
            home_team="Local",
            away_team="Visitante",
            model_name="Poisson + Monte Carlo",
            simulations=10_000,
            markets=self._markets(),
            context_json=context,
        )

        stored = self.repository.list_runs_by_match_id(
            "fixture-fallback", sport="Fútbol"
        )
        self.assertEqual([row["id"] for row in stored], [run_id])
        self.assertEqual(stored[0]["model_name"], "Poisson + Monte Carlo")
        self.assertEqual(stored[0]["context_json"], context)

    def test_market_failure_rolls_back_run_and_all_markets(self) -> None:
        invalid_markets = self._markets() + [
            {
                "market_type": "draw",
                # selection is deliberately missing after the run has been flushed.
                "probability": 20.0,
            }
        ]
        with self.assertRaises(KeyError):
            self.repository.save_prediction_run(
                sport="Fútbol",
                match_id="fixture-invalid",
                home_team="Local",
                away_team="Visitante",
                model_name="Football ML + Poisson + Monte Carlo",
                simulations=10_000,
                markets=invalid_markets,
                context_json={"quality_gate": {"result": False}},
            )

        with self.session_factory() as session:
            run_count = session.scalar(select(func.count()).select_from(PredictionRun))
            market_count = session.scalar(select(func.count()).select_from(PredictionMarket))
        self.assertEqual(run_count, 0)
        self.assertEqual(market_count, 0)

    def test_active_football_model_is_rejected_without_baseline_evidence(self) -> None:
        metadata_path = Path("machine_learning/models_store/football_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictor = object.__new__(FootballPredictor)
        predictor.metadata = metadata

        self.assertFalse(predictor._legacy_result_is_qualified())
        self.assertLess(float(metadata["metrics"]["over_auc"]), 0.50)
        self.assertLess(float(metadata["metrics"]["btts_auc"]), 0.50)


if __name__ == "__main__":
    unittest.main()
