from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from core.market_risk import apply_probability_risk
from core.logger import logger
from engines.baseball_prediction_engine import BaseballPredictionEngine
from machine_learning.features.baseball_features import BaseballFeatures
from machine_learning.model_quality import market_model_is_qualified, validated_ml_weight


class BaseballPredictor:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or Path("machine_learning/models_store")
        self.features = BaseballFeatures()
        self.engine = BaseballPredictionEngine()
        self.metadata: dict[str, Any] = {}
        self.models_available = False
        try:
            with self._path("baseball_metadata.json", "metadata.json").open(encoding="utf-8") as handle:
                self.metadata = json.load(handle)
            if self.metadata.get("status") not in {"active", "candidate_qualified"}:
                return
            self.home_win_model = joblib.load(
                self._path("baseball_home_win_model.joblib", "home_win_model.joblib")
            )
            self.over_model = joblib.load(self._path("baseball_over85_model.joblib", "over85_model.joblib"))
            self.home_over_model = joblib.load(
                self._path("baseball_home_over35_model.joblib", "home_over35_model.joblib")
            )
            self.models_available = True
        except (FileNotFoundError, OSError, ValueError) as error:
            logger.warning("Modelos de béisbol no disponibles; se usará fallback: %s", error)

    def _path(self, active_name: str, candidate_name: str) -> Path:
        active = self.model_dir / active_name
        return active if active.exists() else self.model_dir / candidate_name

    def analyze_match(
        self,
        selected_match: dict[str, Any],
        simulations: int,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        is_mlb = str(selected_match.get("league", "")).strip().upper() == "MLB"
        if not self.models_available or not is_mlb:
            return self.engine.analyze_match(selected_match, simulations, force_refresh)

        season = self._history_season(
            self.engine._infer_season(selected_match),
            self.metadata.get("test_season"),
        )
        feature_df = self.features.build_live_feature_row(
            home_team_id=selected_match["home_id"],
            away_team_id=selected_match["away_id"],
            season=season,
            league_id=1,
            force_refresh=force_refresh,
            match_date=selected_match.get("date"),
            include_elo="diff_elo" in self.metadata["feature_columns"],
        )
        columns = self.metadata["feature_columns"]
        X = feature_df[columns]
        quality_gate = {
            "home_win": market_model_is_qualified(
                self.metadata, "home_win", minimum_lift=0.01, minimum_auc=0.55
            ),
            "over_8_5": market_model_is_qualified(
                self.metadata, "over_85", minimum_lift=0.01, minimum_auc=0.55
            ),
            "home_over_3_5": market_model_is_qualified(
                self.metadata, "home_over_35", minimum_lift=0.01, minimum_auc=0.55
            ),
        }
        ml_probabilities: dict[str, float] = {}
        if quality_gate["home_win"]:
            ml_probabilities["home_win"] = float(
                self.home_win_model.predict_proba(X)[0][1] * 100
            )
        if quality_gate["over_8_5"]:
            ml_probabilities["over_8_5"] = float(
                self.over_model.predict_proba(X)[0][1] * 100
            )
        if quality_gate["home_over_3_5"]:
            ml_probabilities["home_over_3_5"] = float(
                self.home_over_model.predict_proba(X)[0][1] * 100
            )

        home_profile, away_profile = self.engine.get_team_profiles(
            selected_match["home_id"],
            selected_match["away_id"],
            season=season,
            force_refresh=force_refresh,
        )
        home_lambda, away_lambda = self.engine.calculate_expected_runs(
            home_profile, away_profile
        )
        simulation = self.engine.run_monte_carlo(home_lambda, away_lambda, simulations)
        decisive_total = (
            simulation["home_win_probability"] + simulation["away_win_probability"]
        )
        engine_home = (
            simulation["home_win_probability"] / decisive_total * 100
            if decisive_total else 50.0
        )
        weights = {
            "home_win": validated_ml_weight(self.metadata, "home_win"),
            "over_8_5": validated_ml_weight(self.metadata, "over_85"),
            "home_over_3_5": validated_ml_weight(self.metadata, "home_over_35"),
        }
        home_win = (
            ml_probabilities["home_win"] * weights["home_win"] + engine_home * (1 - weights["home_win"])
            if quality_gate["home_win"]
            else engine_home
        )
        away_win = 100 - home_win
        over_85 = (
            ml_probabilities["over_8_5"] * weights["over_8_5"] + simulation["over_85_probability"] * (1 - weights["over_8_5"])
            if quality_gate["over_8_5"]
            else simulation["over_85_probability"]
        )
        home_over_35 = (
            ml_probabilities["home_over_3_5"] * weights["home_over_3_5"] + simulation["home_over_35_probability"] * (1 - weights["home_over_3_5"])
            if quality_gate["home_over_3_5"]
            else simulation["home_over_35_probability"]
        )
        home_team = selected_match["home"]
        away_team = selected_match["away"]

        markets_to_save = [
            {"market_type": "home_win", "selection": home_team, "probability": home_win, "confidence": "Media", "risk": "Medio"},
            {"market_type": "away_win", "selection": away_team, "probability": away_win, "confidence": "Media", "risk": "Medio"},
            {"market_type": "over_8_5_runs", "selection": "Over 8.5 carreras", "probability": over_85, "confidence": "Media", "risk": "Medio"},
            {"market_type": "under_10_5_runs", "selection": "Under 10.5 carreras", "probability": simulation["under_105_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "home_over_3_5_runs", "selection": f"{home_team} over 3.5 carreras", "probability": home_over_35, "confidence": "Media", "risk": "Medio"},
        ]
        apply_probability_risk(markets_to_save)
        markets = pd.DataFrame(
            {
                "Mercado": [item["selection"] for item in markets_to_save],
                "Probabilidad": [f'{item["probability"]:.1f}%' for item in markets_to_save],
                "Confianza": [item["confidence"] for item in markets_to_save],
                "Riesgo": [item["risk"] for item in markets_to_save],
            }
        )
        return {
            "model_name": "Baseball ML + Runs + Monte Carlo",
            "summary_cards": [
                {"label": f"Victoria {home_team}", "value": f"{home_win:.1f}%"},
                {"label": f"Victoria {away_team}", "value": f"{away_win:.1f}%"},
            ],
            "extra_metrics": {
                "Carreras esperadas local": f"{home_lambda:.2f}",
                "Carreras esperadas visitante": f"{away_lambda:.2f}",
                "Simulaciones": f"{simulations:,}",
                "Modelo": "ML + Runs",
            },
            "markets": markets.to_dict(orient="records"),
            "markets_to_save": markets_to_save,
            "context_json": {
                "model_version": self.metadata["model_version"],
                "quality_gate": quality_gate,
                "ml_weights": weights,
                "features": feature_df.to_dict(orient="records"),
                "ml_probabilities": ml_probabilities,
                "home_profile": home_profile,
                "away_profile": away_profile,
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "top_scores": simulation["top_scores"],
            },
        }

    @staticmethod
    def _history_season(requested: Any, validated: Any) -> int | None:
        try:
            validated_year = int(validated)
        except (TypeError, ValueError):
            validated_year = None
        try:
            requested_year = int(requested)
        except (TypeError, ValueError):
            requested_year = None
        if validated_year is None:
            return requested_year
        return min(requested_year, validated_year) if requested_year else validated_year
