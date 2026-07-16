from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from core.market_risk import apply_probability_risk
from core.logger import logger
from engines.nfl_prediction_engine import NFLPredictionEngine
from machine_learning.features.nfl_features import NFLFeatures
from machine_learning.model_quality import market_model_is_qualified, validated_ml_weight


class NFLPredictor:
    def __init__(self, model_dir: Path | None = None) -> None:
        root = model_dir or Path("machine_learning/models_store")
        self.engine = NFLPredictionEngine()
        self.features = NFLFeatures()
        self.available = False
        self.metadata: dict[str, Any] = {}
        try:
            self.metadata = json.loads(self._path(root, "nfl_metadata.json", "metadata.json").read_text(encoding="utf-8"))
            if self.metadata.get("status") not in {"active", "candidate_qualified"}:
                return
            self.home_model = joblib.load(self._path(root, "nfl_home_win_model.joblib", "home_win_model.joblib"))
            self.over_model = joblib.load(self._path(root, "nfl_over_415_model.joblib", "over_415_model.joblib"))
            self.home_over_model = joblib.load(self._path(root, "nfl_home_over_205_model.joblib", "home_over_205_model.joblib"))
            self.available = True
        except (FileNotFoundError, OSError, ValueError) as error:
            logger.warning("Modelos NFL no disponibles; se usará fallback: %s", error)

    @staticmethod
    def _path(root: Path, active_name: str, candidate_name: str) -> Path:
        active = root / active_name
        return active if active.exists() else root / candidate_name

    def analyze_match(
        self,
        selected_match: dict[str, Any],
        simulations: int,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        if not self.available or str(selected_match.get("league", "")).strip().upper() != "NFL":
            return self.engine.analyze_match(selected_match, simulations, force_refresh)

        season = self.engine._infer_season(selected_match)
        # El plan actual de la API solo entrega históricos hasta la temporada
        # validada en los metadatos. Evita perfiles neutros para juegos futuros.
        history_season = min(season or self.metadata["test_season"], self.metadata["test_season"])
        feature_df = self.features.build_live_feature_row(
            selected_match["home_id"], selected_match["away_id"],
            season=history_season, league_id=1, force_refresh=force_refresh,
            match_date=selected_match.get("date"),
            include_elo="diff_elo" in self.metadata["feature_columns"],
        )
        X = feature_df[self.metadata["feature_columns"]]
        quality_gate = {
            "home_win": market_model_is_qualified(self.metadata, "home_win"),
            "over_41_5": market_model_is_qualified(self.metadata, "over_415"),
            "home_over_20_5": market_model_is_qualified(self.metadata, "home_over_205"),
        }
        ml_probabilities: dict[str, float] = {}
        if quality_gate["home_win"]:
            ml_probabilities["home_win"] = float(self.home_model.predict_proba(X)[0][1] * 100)
        if quality_gate["over_41_5"]:
            ml_probabilities["over_41_5"] = float(self.over_model.predict_proba(X)[0][1] * 100)
        if quality_gate["home_over_20_5"]:
            ml_probabilities["home_over_20_5"] = float(self.home_over_model.predict_proba(X)[0][1] * 100)

        home_profile, away_profile = self.engine.get_team_profiles(
            selected_match["home_id"], selected_match["away_id"],
            season=history_season, league_id=1, force_refresh=force_refresh,
        )
        home_expected, away_expected = self.engine.calculate_expected_points(home_profile, away_profile)
        simulation = self.engine.run_monte_carlo(
            home_expected, away_expected, home_profile["score_std"],
            away_profile["score_std"], simulations,
        )
        decisive = simulation["home_win_probability"] + simulation["away_win_probability"]
        engine_home = simulation["home_win_probability"] / decisive * 100 if decisive else 50.0
        weights = {
            "home_win": validated_ml_weight(self.metadata, "home_win"),
            "home_over_20_5": validated_ml_weight(self.metadata, "home_over_205"),
        }
        home_win = (ml_probabilities["home_win"] * weights["home_win"] + engine_home * (1 - weights["home_win"])
                    if quality_gate["home_win"] else engine_home)
        away_win = 100 - home_win
        over = ((ml_probabilities["over_41_5"] + simulation["over_415_probability"]) / 2
                if quality_gate["over_41_5"] else simulation["over_415_probability"])
        home_over = (ml_probabilities["home_over_20_5"] * weights["home_over_20_5"] + simulation["home_over_205_probability"] * (1 - weights["home_over_20_5"])
                     if quality_gate["home_over_20_5"] else simulation["home_over_205_probability"])
        home, away = selected_match["home"], selected_match["away"]
        saved = [
            {"market_type": "home_win", "selection": home, "probability": home_win, "confidence": "Media", "risk": "Medio"},
            {"market_type": "away_win", "selection": away, "probability": away_win, "confidence": "Media", "risk": "Medio"},
            {"market_type": "over_41_5_points", "selection": "Over 41.5 puntos", "probability": over, "confidence": "Media", "risk": "Medio"},
            {"market_type": "under_52_5_points", "selection": "Under 52.5 puntos", "probability": simulation["under_525_probability"], "confidence": "Media", "risk": "Medio"},
            {"market_type": "home_over_20_5_points", "selection": f"{home} over 20.5 puntos", "probability": home_over, "confidence": "Media", "risk": "Medio"},
        ]
        apply_probability_risk(saved)
        markets = pd.DataFrame({
            "Mercado": [item["selection"] for item in saved],
            "Probabilidad": [f'{item["probability"]:.1f}%' for item in saved],
            "Confianza": [item["confidence"] for item in saved],
            "Riesgo": [item["risk"] for item in saved],
        })
        return {
            "model_name": "NFL ML + Drive/Points + Monte Carlo",
            "summary_cards": [
                {"label": f"Victoria {home}", "value": f"{home_win:.1f}%"},
                {"label": f"Victoria {away}", "value": f"{away_win:.1f}%"},
            ],
            "extra_metrics": {
                "Puntos esperados local": f"{home_expected:.1f}",
                "Puntos esperados visitante": f"{away_expected:.1f}",
                "Simulaciones": f"{simulations:,}",
                "Modelo": "NFL ML + Drive/Points",
            },
            "markets": markets.to_dict(orient="records"),
            "markets_to_save": saved,
            "context_json": {
                "model_version": self.metadata["model_version"],
                "quality_gate": quality_gate,
                "ml_weights": weights,
                "history_season": history_season,
                "features": feature_df.to_dict(orient="records"),
                "ml_probabilities": ml_probabilities,
                "home_profile": home_profile,
                "away_profile": away_profile,
                "home_expected": home_expected,
                "away_expected": away_expected,
                "top_scores": simulation["top_scores"],
            },
        }
