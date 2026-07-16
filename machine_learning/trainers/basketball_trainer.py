from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from database.training_runs_repository import TrainingRunsRepository
from machine_learning.model_quality import expected_calibration_error
from machine_learning.probability_ensemble import ProbabilityEnsemble
from machine_learning.features.basketball_features import BasketballFeatures


class BasketballTrainer:
    def __init__(self) -> None:
        self.model_dir = Path("machine_learning/models_store")
        self.features = BasketballFeatures.feature_columns()

    def train_from_dataframe(
        self, dataset: pd.DataFrame, test_season: str | None = None
    ) -> dict[str, Any]:
        targets = {
            "home_win": "home_win_label",
            "over_2195": "over_2195_label",
            "home_over_1095": "home_over_1095_label",
        }
        required = self.features + list(targets.values()) + ["season", "game_date"]
        clean = dataset.dropna(subset=required).sort_values("game_date")
        seasons = sorted(str(value) for value in clean.season.unique())
        selected_test_season = test_season or seasons[-1]
        train = clean[clean.season.astype(str) < str(selected_test_season)]
        test = clean[clean.season.astype(str) == str(selected_test_season)]
        if len(train) < 1000 or len(test) < 100:
            raise ValueError(f"Datos insuficientes: train={len(train)}, test={len(test)}")

        models = {}
        metrics = {}
        for index, (name, target) in enumerate(targets.items()):
            forest = CalibratedClassifierCV(
                Pipeline([
                    ("variance", VarianceThreshold()),
                    ("select", SelectKBest(f_classif, k=min(48, len(self.features)))),
                    ("model", RandomForestClassifier(
                        n_estimators=350, max_depth=9, min_samples_leaf=5,
                        random_state=42 + index, class_weight="balanced",
                    )),
                ]), method="sigmoid", cv=3,
            )
            linear = CalibratedClassifierCV(
                Pipeline([
                    ("variance", VarianceThreshold()),
                    ("select", SelectKBest(f_classif, k=min(48, len(self.features)))),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]), method="sigmoid", cv=3,
            )
            model = ProbabilityEnsemble([forest, linear], weights=[0.6, 0.4])
            model.fit(train[self.features], train[target])
            probability = model.predict_proba(test[self.features])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            accuracy = float(accuracy_score(test[target], prediction))
            rate = float(test[target].mean())
            baseline = max(rate, 1 - rate)
            metrics[name] = {
                "accuracy": accuracy, "baseline_accuracy": baseline,
                "accuracy_lift": accuracy - baseline,
                "brier_score": float(brier_score_loss(test[target], probability)),
                "roc_auc": float(roc_auc_score(test[target], probability)),
                "calibration_error": expected_calibration_error(test[target], probability),
            }
            models[name] = model

        qualified_markets = [
            name for name, item in metrics.items()
            if item["roc_auc"] >= 0.52 and item["accuracy_lift"] >= 0
        ]
        active = bool(qualified_markets)
        version = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "model_version": version,
            "status": "candidate_qualified" if active else "validation_failed",
            "league_id": 12,
            "train_seasons": sorted(str(value) for value in train.season.unique()),
            "test_season": str(selected_test_season),
            "train_rows": len(train), "test_rows": len(test),
            "feature_columns": self.features, "metrics": metrics,
            "qualified_markets": qualified_markets,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "calibration": "sigmoid_cv3",
            "ensemble": ["calibrated_random_forest", "calibrated_logistic_regression"],
            "selected_feature_limit": min(48, len(self.features)),
        }
        version_dir = self.model_dir / "versions" / "basketball" / version
        version_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, version_dir / f"{name}_model.joblib")
        (version_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        TrainingRunsRepository().save_training_run("basketball", metadata)
        return metadata
