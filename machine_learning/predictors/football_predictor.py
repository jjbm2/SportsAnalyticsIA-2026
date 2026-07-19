from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from engines.football_prediction_engine import FootballPredictionEngine
from machine_learning.features.football_features import FootballFeatures
from machine_learning.calibration import FootballProbabilityCalibrator
from machine_learning.model_quality import metric_is_qualified, validated_ml_weight
from core.prediction_confidence import enrich_football_markets
from core.match_quality import calculate_match_quality
from core.game_style import apply_game_style_to_markets, classify_game_style


class FootballPredictor:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or Path("machine_learning/models_store")
        self.features_builder = FootballFeatures()
        self.poisson_engine = FootballPredictionEngine()

        metadata_path = self._path("football_metadata.json", "metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                "No existe football_metadata.json. Primero entrena el modelo."
            )

        with open(metadata_path, "r", encoding="utf-8") as fp:
            self.metadata = json.load(fp)

        self.feature_columns = self.metadata["feature_columns"]
        self.result_model = joblib.load(self._path("football_result_model.joblib", "result_model.joblib"))
        self.over_model = joblib.load(self._path("football_over25_model.joblib", "over25_model.joblib"))
        self.btts_model = joblib.load(self._path("football_btts_model.joblib", "btts_model.joblib"))
        self.probability_calibrator = FootballProbabilityCalibrator.load_compatible(
            self._path("football_calibrator.joblib", "calibrator.joblib"), self.metadata
        )

    def _path(self, active_name: str, candidate_name: str) -> Path:
        active = self.model_dir / active_name
        return active if active.exists() else self.model_dir / candidate_name

    def predict_match(
        self,
        home_team_id: int,
        away_team_id: int,
        home_team_name: str,
        away_team_name: str,
        simulations: int = 10000,
        force_refresh: bool = False,
        provider: str = "api_sports",
        competition: str | int | None = None,
    ) -> dict[str, Any]:
        features_df = self.features_builder.build_live_feature_row(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            force_refresh=force_refresh,
            provider=provider,
            competition=competition,
        )

        X = features_df[self.feature_columns]

        qualified_markets = set(self.metadata.get("qualified_markets") or [])
        quality_gate = {
            "result": self._legacy_result_is_qualified()
            if not qualified_markets else "result" in qualified_markets,
            "over_2_5": metric_is_qualified(
                self.metadata, "over_auc", minimum=0.50
            ) if not qualified_markets else "over_2_5" in qualified_markets,
            "btts": metric_is_qualified(
                self.metadata, "btts_auc", minimum=0.50
            ) if not qualified_markets else "btts" in qualified_markets,
        }

        ml_probabilities: dict[str, float] = {}
        if quality_gate["result"]:
            result_X = features_df[list(getattr(self.result_model, "feature_names_in_", self.feature_columns))]
            result_probs_raw = self.result_model.predict_proba(result_X)[0]
            result_classes = [int(value) for value in self.result_model.classes_]
            result_probs_raw = self._calibrate("result", result_probs_raw, result_classes)
            class_map = {
                label: prob
                for label, prob in zip(
                    self.result_model.classes_, result_probs_raw
                )
            }
            ml_probabilities.update(
                home_win=float(class_map.get(1, 0.0) * 100),
                draw=float(class_map.get(0, 0.0) * 100),
                away_win=float(class_map.get(-1, 0.0) * 100),
            )
        if quality_gate["over_2_5"]:
            over_X = features_df[list(getattr(self.over_model, "feature_names_in_", self.feature_columns))]
            raw_over = self.over_model.predict_proba(over_X)[0]
            over_classes = [int(value) for value in self.over_model.classes_]
            calibrated_over = self._calibrate("over_2_5", raw_over, over_classes)
            ml_probabilities["over25"] = float(
                calibrated_over[over_classes.index(1)] * 100
            )
        if quality_gate["btts"]:
            btts_X = features_df[list(getattr(self.btts_model, "feature_names_in_", self.feature_columns))]
            raw_btts = self.btts_model.predict_proba(btts_X)[0]
            btts_classes = [int(value) for value in self.btts_model.classes_]
            calibrated_btts = self._calibrate("btts", raw_btts, btts_classes)
            ml_probabilities["btts"] = float(
                calibrated_btts[btts_classes.index(1)] * 100
            )

        home_profile, away_profile = self.poisson_engine.get_team_profiles(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            force_refresh=force_refresh,
            provider=provider,
            competition=competition,
        )
        self.poisson_engine.validate_team_profiles(home_profile, away_profile)

        home_lambda, away_lambda = self.poisson_engine.calculate_expected_goals(
            home_profile=home_profile,
            away_profile=away_profile,
        )

        poisson_result = self.poisson_engine.run_monte_carlo(
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            simulations=simulations,
        )

        blend_weights = {
            market: validated_ml_weight(self.metadata, market) if qualified else 0.0
            for market, qualified in quality_gate.items()
        }

        if quality_gate["result"]:
            result_weight = blend_weights["result"]
            final_home_win = ml_probabilities["home_win"] * result_weight + poisson_result["home_win_probability"] * (1.0 - result_weight)
            final_draw = ml_probabilities["draw"] * result_weight + poisson_result["draw_probability"] * (1.0 - result_weight)
            final_away_win = ml_probabilities["away_win"] * result_weight + poisson_result["away_win_probability"] * (1.0 - result_weight)
        else:
            final_home_win = poisson_result["home_win_probability"]
            final_draw = poisson_result["draw_probability"]
            final_away_win = poisson_result["away_win_probability"]
        final_over25 = (
            ml_probabilities["over25"] * blend_weights["over_2_5"]
            + poisson_result["over_25_probability"] * (1.0 - blend_weights["over_2_5"])
            if quality_gate["over_2_5"]
            else poisson_result["over_25_probability"]
        )
        final_btts = (
            ml_probabilities["btts"] * blend_weights["btts"]
            + poisson_result["btts_probability"] * (1.0 - blend_weights["btts"])
            if quality_gate["btts"]
            else poisson_result["btts_probability"]
        )
        goal_lines = {line: dict(values) for line, values in poisson_result["goal_lines"].items()}
        if quality_gate["over_2_5"]:
            goal_lines["2.5"] = {"over": final_over25, "under": 100.0 - final_over25}
        recommended_total = self.poisson_engine.select_goal_line(goal_lines)
        result_choices = [
            (home_team_name, final_home_win), ("Empate", final_draw),
            (away_team_name, final_away_win),
        ]
        likely_result, likely_result_probability = max(result_choices, key=lambda item: item[1])

        markets_to_save = self.poisson_engine.build_market_options(
            home_team_name, away_team_name, final_home_win, final_draw, final_away_win,
            goal_lines, final_btts, poisson_result["home_score_probability"],
            poisson_result["away_score_probability"],
            poisson_result.get("team_goal_markets"),
        )
        feature_values = features_df.iloc[0].to_dict()
        enrich_football_markets(
            markets_to_save,
            feature_values,
            ml_probabilities,
            poisson_result,
            quality_gate,
        )
        match_quality = calculate_match_quality(
            markets_to_save,
            feature_values,
            quality_gate=quality_gate,
        )
        game_style = classify_game_style(feature_values, match_quality)
        apply_game_style_to_markets(markets_to_save, game_style)
        markets_df = pd.DataFrame({
            "Categoría": [item["extra_data_json"]["category"] for item in markets_to_save],
            "Mercado": [item["selection"] for item in markets_to_save],
            "Probabilidad": [f'{item["probability"]:.1f}%' for item in markets_to_save],
            "Confianza": [item["confidence"] for item in markets_to_save],
            "Puntuación de confianza": [item["confidence_score"] for item in markets_to_save],
            "Riesgo estimado": [item["risk"] for item in markets_to_save],
        })

        active_ml_markets = [market for market, qualified in quality_gate.items() if qualified]
        model_name = (
            "Football ML validado + Poisson + Monte Carlo"
            if active_ml_markets else "Poisson + Monte Carlo"
        )
        return {
            **match_quality,
            **game_style,
            "model_name": model_name,
            "summary_cards": [
                {"label": "Resultado más probable", "value": f"{likely_result} · {likely_result_probability:.1f}%"},
                {"label": "Total sugerido", "value": f'{recommended_total["label"]} · {recommended_total["probability"]:.1f}%'},
            ],
            "extra_metrics": {
                "xG local estimado": f"{home_lambda:.2f}",
                "xG visitante estimado": f"{away_lambda:.2f}",
                "Simulaciones": f"{simulations:,}",
                "Modelo": "ML validado + Poisson" if active_ml_markets else "Poisson",
                "Señales ML aprobadas": str(len(active_ml_markets)),
            },
            "markets": markets_df.to_dict(orient="records"),
            "markets_to_save": markets_to_save,
            "context_json": {
                "features": features_df.to_dict(orient="records"),
                "home_profile": home_profile,
                "away_profile": away_profile,
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "top_scores": poisson_result["top_scores"],
                "goal_lines": goal_lines,
                "recommended_total": recommended_total,
                "recommended_result": {"selection": likely_result, "probability": likely_result_probability},
                "quality_gate": quality_gate,
                "ml_probabilities": ml_probabilities,
                "ml_blend_weights": blend_weights,
                "match_quality": match_quality,
                "game_style": game_style,
                "probability_calibration": {
                    "applied": self.probability_calibrator is not None,
                    "separate_from_base_model": True,
                },
            },
        }

    def _calibrate(self, market: str, probabilities: Any, classes: list[int]) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        if self.probability_calibrator is None:
            return values
        return self.probability_calibrator.calibrate(market, values, classes)

    def _legacy_result_is_qualified(self) -> bool:
        """Reject legacy models that do not prove an out-of-sample lift."""
        metrics = self.metadata.get("metrics", {})
        try:
            accuracy = float(metrics["result_accuracy"])
            baseline = float(metrics["result_baseline_accuracy"])
            lift = float(metrics.get("result_accuracy_lift", accuracy - baseline))
        except (KeyError, TypeError, ValueError):
            return False
        return lift >= 0.02 and accuracy > baseline
