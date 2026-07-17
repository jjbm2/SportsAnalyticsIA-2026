from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from core.market_risk import apply_probability_risk
from core.logger import logger
from engines.basketball_prediction_engine import BasketballPredictionEngine
from machine_learning.features.basketball_features import BasketballFeatures
from machine_learning.model_quality import market_model_is_qualified, validated_ml_weight


class BasketballPredictor:
    def __init__(self, model_dir: Path | None = None) -> None:
        root = model_dir or Path("machine_learning/models_store")
        self.engine = BasketballPredictionEngine()
        self.features = BasketballFeatures()
        self.available = False
        self.metadata = {}
        try:
            self.metadata = json.loads(self._path(root, "basketball_metadata.json", "metadata.json").read_text(encoding="utf-8"))
            if self.metadata.get("status") not in {"active", "candidate_qualified"}:
                return
            self.home_model = joblib.load(self._path(root, "basketball_home_win_model.joblib", "home_win_model.joblib"))
            self.over_model = joblib.load(self._path(root, "basketball_over_2195_model.joblib", "over_2195_model.joblib"))
            self.home_over_model = joblib.load(self._path(root, "basketball_home_over_1095_model.joblib", "home_over_1095_model.joblib"))
            self.available = True
        except (FileNotFoundError, OSError, ValueError) as error:
            logger.warning("Modelos NBA no disponibles; se usará fallback: %s", error)

    @staticmethod
    def _path(root: Path, active_name: str, candidate_name: str) -> Path:
        active = root / active_name
        return active if active.exists() else root / candidate_name

    def analyze_match(self, selected_match: dict[str, Any], simulations: int, force_refresh: bool = False) -> dict[str, Any]:
        if not self.available or str(selected_match.get("league", "")).strip().upper() != "NBA":
            return self.engine.analyze_match(selected_match, simulations, force_refresh)

        season = self._history_season(
            self._season(selected_match.get("date")),
            self.metadata.get("test_season"),
        )
        features = self.features.build_live_feature_row(
            selected_match["home_id"], selected_match["away_id"],
            season=season, league_id=12, force_refresh=force_refresh,
            match_date=selected_match.get("date"),
            include_elo="diff_elo" in self.metadata["feature_columns"],
        )
        X = features[self.metadata["feature_columns"]]
        quality_gate = {
            "home_win": market_model_is_qualified(
                self.metadata, "home_win", minimum_lift=0.01, minimum_auc=0.60
            ),
            "over_219_5": market_model_is_qualified(
                self.metadata, "over_2195", minimum_lift=0.01, minimum_auc=0.60
            ),
            "home_over_109_5": market_model_is_qualified(
                self.metadata, "home_over_1095", minimum_lift=0.01, minimum_auc=0.60
            ),
        }
        ml_probabilities: dict[str, float] = {}
        if quality_gate["home_win"]:
            ml_probabilities["home_win"] = float(self.home_model.predict_proba(X)[0][1] * 100)
        if quality_gate["over_219_5"]:
            ml_probabilities["over_219_5"] = float(self.over_model.predict_proba(X)[0][1] * 100)
        if quality_gate["home_over_109_5"]:
            ml_probabilities["home_over_109_5"] = float(self.home_over_model.predict_proba(X)[0][1] * 100)

        home_profile, away_profile = self.engine.get_team_profiles(
            selected_match["home_id"], selected_match["away_id"],
            season=season, league_id=12, force_refresh=force_refresh,
        )
        home_expected, away_expected = self.engine.calculate_expected_points(home_profile, away_profile)
        sim = self.engine.run_monte_carlo(
            home_expected, away_expected, home_profile["score_std"],
            away_profile["score_std"], simulations,
        )
        decisive = sim["home_win_probability"] + sim["away_win_probability"]
        engine_home = sim["home_win_probability"] / decisive * 100 if decisive else 50.0
        weights = {"home_win": validated_ml_weight(self.metadata, "home_win")}
        home_win = (ml_probabilities["home_win"] * weights["home_win"] + engine_home * (1 - weights["home_win"])
                    if quality_gate["home_win"] else engine_home)
        away_win = 100 - home_win
        over = ((ml_probabilities["over_219_5"] + sim["over_2195_probability"]) / 2
                if quality_gate["over_219_5"] else sim["over_2195_probability"])
        home_over = ((ml_probabilities["home_over_109_5"] + sim["home_over_1095_probability"]) / 2
                     if quality_gate["home_over_109_5"] else sim["home_over_1095_probability"])
        home, away = selected_match["home"], selected_match["away"]
        saved = [
            {"market_type":"home_win","selection":home,"probability":home_win,"confidence":"Media-Alta","risk":"Medio"},
            {"market_type":"away_win","selection":away,"probability":away_win,"confidence":"Media-Alta","risk":"Medio"},
            {"market_type":"over_219_5_points","selection":"Over 219.5 puntos","probability":over,"confidence":"Media","risk":"Medio"},
            {"market_type":"under_234_5_points","selection":"Under 234.5 puntos","probability":sim["under_2345_probability"],"confidence":"Media","risk":"Medio"},
            {"market_type":"home_over_109_5_points","selection":f"{home} over 109.5 puntos","probability":home_over,"confidence":"Media","risk":"Medio"},
        ]
        apply_probability_risk(saved)
        markets = pd.DataFrame({
            "Mercado":[item["selection"] for item in saved],
            "Probabilidad":[f'{item["probability"]:.1f}%' for item in saved],
            "Confianza":[item["confidence"] for item in saved],
            "Riesgo":[item["risk"] for item in saved],
        })
        return {
            "model_name":"Basketball ML + Pace + Monte Carlo",
            "summary_cards":[
                {"label":f"Victoria {home}","value":f"{home_win:.1f}%"},
                {"label":f"Victoria {away}","value":f"{away_win:.1f}%"},
            ],
            "extra_metrics":{"Puntos esperados local":f"{home_expected:.1f}","Puntos esperados visitante":f"{away_expected:.1f}","Simulaciones":f"{simulations:,}","Modelo":"NBA ML + Pace"},
            "markets":markets.to_dict(orient="records"), "markets_to_save":saved,
            "context_json":{"model_version":self.metadata["model_version"],"quality_gate":quality_gate,"ml_weights":weights,"features":features.to_dict(orient="records"),"ml_probabilities":ml_probabilities,"home_profile":home_profile,"away_profile":away_profile,"home_expected":home_expected,"away_expected":away_expected,"top_scores":sim["top_scores"]},
        }

    @staticmethod
    def _season(value: Any) -> str:
        year = int(str(value)[:4])
        month = int(str(value)[5:7])
        return f"{year}-{year + 1}" if month >= 9 else f"{year - 1}-{year}"

    @staticmethod
    def _history_season(requested: str, validated: Any) -> str:
        validated_season = str(validated or "").strip()
        if not validated_season:
            return requested
        try:
            requested_year = int(str(requested).split("-", 1)[0])
            validated_year = int(validated_season.split("-", 1)[0])
        except (TypeError, ValueError):
            return validated_season
        return requested if requested_year <= validated_year else validated_season
