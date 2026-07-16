from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from database.training_runs_repository import TrainingRunsRepository
from machine_learning.features.formula1_features import Formula1Features


class Formula1Trainer:
    def __init__(self) -> None:
        self.root = Path("machine_learning/models_store")
        self.features = Formula1Features.feature_columns()

    def train_from_dataframe(self, dataset: pd.DataFrame, test_season: int = 2025) -> dict[str, Any]:
        required = self.features + ["season", "podium_label"]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise ValueError(f"Faltan columnas F1: {missing}")
        clean = dataset.dropna(subset=required).sort_values(["season", "round"])
        train = clean[clean["season"] < test_season]
        test = clean[clean["season"] == test_season]
        if len(train) < 700 or len(test) < 250:
            raise ValueError(f"Datos F1 insuficientes: train={len(train)}, test={len(test)}")

        model = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=500, max_depth=7, min_samples_leaf=6,
                class_weight="balanced", random_state=44,
            ),
            method="sigmoid",
            cv=3,
        )
        model.fit(train[self.features], train["podium_label"])
        probability = model.predict_proba(test[self.features])[:, 1]
        rate = float(test["podium_label"].mean())
        metrics = {
            "roc_auc": float(roc_auc_score(test["podium_label"], probability)),
            "brier_score": float(brier_score_loss(test["podium_label"], probability)),
            "baseline_brier": float(rate * (1.0 - rate)),
            "positive_rate": rate,
        }
        qualified = metrics["roc_auc"] >= 0.65 and metrics["brier_score"] < metrics["baseline_brier"]
        version = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "model_version": version,
            "status": "candidate_qualified" if qualified else "validation_failed",
            "target": "podium",
            "train_seasons": sorted(int(value) for value in train["season"].unique()),
            "test_season": test_season,
            "train_rows": len(train),
            "test_rows": len(test),
            "feature_columns": self.features,
            "metrics": {"podium": metrics},
            "qualified_markets": ["podium"] if qualified else [],
            "trained_at": pd.Timestamp.utcnow().isoformat(),
        }
        candidate = self.root / "versions" / "formula1" / version
        candidate.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, candidate / "podium_model.joblib")
        (candidate / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        TrainingRunsRepository().save_training_run("formula1", metadata)
        return metadata
