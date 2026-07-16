from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from database.training_runs_repository import TrainingRunsRepository
from machine_learning.model_quality import expected_calibration_error
from machine_learning.features.baseball_features import BaseballFeatures


class BaseballTrainer:
    def __init__(self) -> None:
        self.model_dir = Path("machine_learning/models_store")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_columns = BaseballFeatures.feature_columns()

    def train_from_dataframe(
        self,
        dataset: pd.DataFrame,
        test_season: int = 2024,
        random_state: int = 42,
    ) -> dict[str, Any]:
        targets = ["home_win_label", "over_85_label", "home_over_35_label"]
        required = self.feature_columns + targets + ["season"]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el dataset: {missing}")

        clean = dataset.dropna(subset=required).sort_values("game_date").copy()
        train = clean[clean["season"] < test_season]
        test = clean[clean["season"] == test_season]
        if len(train) < 500 or len(test) < 100:
            raise ValueError(
                f"Datos insuficientes para validación temporal: train={len(train)}, test={len(test)}"
            )

        models = {
            "home_win": CalibratedClassifierCV(
                Pipeline([
                    ("variance", VarianceThreshold()),
                    ("select", SelectKBest(f_classif, k=min(48, len(self.feature_columns)))),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000)),
                ]),
                method="sigmoid", cv=3,
            ),
            "over_85": CalibratedClassifierCV(
                Pipeline([
                    ("variance", VarianceThreshold()),
                    ("select", SelectKBest(f_classif, k=min(48, len(self.feature_columns)))),
                    ("model", RandomForestClassifier(
                        n_estimators=350, max_depth=9, min_samples_leaf=5, random_state=random_state
                    )),
                ]), method="sigmoid", cv=3,
            ),
            "home_over_35": CalibratedClassifierCV(
                Pipeline([
                    ("variance", VarianceThreshold()),
                    ("select", SelectKBest(f_classif, k=min(48, len(self.feature_columns)))),
                    ("model", RandomForestClassifier(
                        n_estimators=350, max_depth=9, min_samples_leaf=5, random_state=random_state
                    )),
                ]), method="sigmoid", cv=3,
            ),
        }
        target_map = dict(zip(models, targets))
        metrics: dict[str, Any] = {}
        X_train = train[self.feature_columns]
        X_test = test[self.feature_columns]

        for name, model in models.items():
            target = target_map[name]
            model.fit(X_train, train[target])
            probability = model.predict_proba(X_test)[:, 1]
            prediction = (probability >= 0.5).astype(int)
            positive_rate = float(test[target].mean())
            baseline_accuracy = max(positive_rate, 1.0 - positive_rate)
            metrics[name] = {
                "accuracy": float(accuracy_score(test[target], prediction)),
                "baseline_accuracy": baseline_accuracy,
                "accuracy_lift": float(
                    accuracy_score(test[target], prediction) - baseline_accuracy
                ),
                "brier_score": float(brier_score_loss(test[target], probability)),
                "roc_auc": self._safe_auc(test[target], probability),
                "calibration_error": expected_calibration_error(test[target], probability),
            }

        qualified_markets = [
            name for name, item in metrics.items()
            if item["roc_auc"] >= 0.52 and item["accuracy_lift"] >= 0
        ]
        is_active = bool(qualified_markets)
        model_version = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "model_version": model_version,
            "status": "candidate_qualified" if is_active else "validation_failed",
            "league_id": 1,
            "train_seasons": sorted(int(value) for value in train["season"].unique()),
            "test_season": test_season,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "feature_columns": self.feature_columns,
            "metrics": metrics,
            "qualified_markets": qualified_markets,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "calibration": "sigmoid_cv3",
            "selected_feature_limit": min(48, len(self.feature_columns)),
            "validation_note": (
                "Validación temporal 2023→2024. MLB 2025 no está disponible en el plan actual."
            ),
        }

        candidate_dir = self.model_dir / "versions" / "baseball" / model_version
        candidate_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models["home_win"], candidate_dir / "home_win_model.joblib")
        joblib.dump(models["over_85"], candidate_dir / "over85_model.joblib")
        joblib.dump(models["home_over_35"], candidate_dir / "home_over35_model.joblib")
        with (candidate_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        TrainingRunsRepository().save_training_run("baseball", metadata)
        return metadata

    @staticmethod
    def _safe_auc(target: pd.Series, probability: Any) -> float:
        try:
            return float(roc_auc_score(target, probability))
        except ValueError:
            return 0.0
