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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from database.training_runs_repository import TrainingRunsRepository
from machine_learning.model_quality import expected_calibration_error, multiclass_probability_metrics
from machine_learning.features.football_features import FootballFeatures
from machine_learning.backtesting.walk_forward import WalkForwardBacktester


class FootballTrainer:
    def __init__(self):
        self.model_dir = Path("machine_learning/models_store")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_columns = FootballFeatures.feature_columns()

    def train_from_dataframe(
        self,
        dataset: pd.DataFrame,
        random_state: int = 42,
    ) -> dict[str, Any]:
        if dataset.empty:
            raise ValueError("El dataset está vacío.")

        required = self.feature_columns + ["result_label", "over_25_label", "btts_label"]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el dataset: {missing}")

        clean_df = dataset.dropna(subset=required).copy()
        if clean_df.empty:
            raise ValueError("No quedaron filas válidas después de limpiar el dataset.")

        X = clean_df[self.feature_columns]
        y_result = clean_df["result_label"]
        y_over = clean_df["over_25_label"]
        y_btts = clean_df["btts_label"]

        validation_strategy = "stratified_random"
        if "game_date" in clean_df.columns:
            parsed_dates = pd.to_datetime(clean_df["game_date"], errors="coerce", utc=True)
        else:
            parsed_dates = pd.Series(index=clean_df.index, dtype="datetime64[ns, UTC]")
        if parsed_dates.notna().mean() >= 0.8:
            ordered_indices = parsed_dates.sort_values().index.tolist()
            split_at = max(1, int(len(ordered_indices) * 0.8))
            train_idx, test_idx = ordered_indices[:split_at], ordered_indices[split_at:]
            validation_strategy = "temporal_80_20"
        else:
            train_idx, test_idx = train_test_split(
                clean_df.index,
                test_size=0.2,
                random_state=random_state,
                stratify=y_result,
            )

        if {"home_scored_std", "away_scored_std"}.issubset(clean_df.columns):
            variability = (
                clean_df.loc[train_idx, "home_scored_std"]
                + clean_df.loc[train_idx, "away_scored_std"]
            ) / 2.0
            threshold = float(variability.quantile(0.90))
            train_idx = [index for index in train_idx if float(variability.loc[index]) <= threshold]
        else:
            threshold = None

        y_result_train = y_result.loc[train_idx]
        y_result_test = y_result.loc[test_idx]

        y_over_train = y_over.loc[train_idx]
        y_over_test = y_over.loc[test_idx]

        y_btts_train = y_btts.loc[train_idx]
        y_btts_test = y_btts.loc[test_idx]

        market_features = {
            market: WalkForwardBacktester._market_features(self.feature_columns, market)
            for market in ("result", "over_2_5", "btts")
        }
        selected_limits = {"result": 24, "over_2_5": 24, "btts": 18}
        result_pipeline = CalibratedClassifierCV(Pipeline(
            steps=[
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif, k=min(selected_limits["result"], len(market_features["result"])))),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ), method="sigmoid", cv=3)
        result_pipeline.fit(X.loc[train_idx, market_features["result"]], y_result_train)

        over_model = CalibratedClassifierCV(Pipeline([
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=min(selected_limits["over_2_5"], len(market_features["over_2_5"])))),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=random_state,
            )),
        ]), method="sigmoid", cv=3)
        over_model.fit(X.loc[train_idx, market_features["over_2_5"]], y_over_train)

        btts_model = CalibratedClassifierCV(Pipeline([
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=min(selected_limits["btts"], len(market_features["btts"])))),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=random_state,
            )),
        ]), method="sigmoid", cv=3)
        btts_model.fit(X.loc[train_idx, market_features["btts"]], y_btts_train)

        result_pred = result_pipeline.predict(X.loc[test_idx, market_features["result"]])
        over_pred = over_model.predict(X.loc[test_idx, market_features["over_2_5"]])
        btts_pred = btts_model.predict(X.loc[test_idx, market_features["btts"]])

        metrics = {
            "result_accuracy": float(accuracy_score(y_result_test, result_pred)),
            "result_baseline_accuracy": float(y_result_test.value_counts(normalize=True).max()),
            "over_accuracy": float(accuracy_score(y_over_test, over_pred)),
            "over_baseline_accuracy": max(float(y_over_test.mean()), 1 - float(y_over_test.mean())),
            "btts_accuracy": float(accuracy_score(y_btts_test, btts_pred)),
            "btts_baseline_accuracy": max(float(y_btts_test.mean()), 1 - float(y_btts_test.mean())),
        }
        metrics["result_accuracy_lift"] = metrics["result_accuracy"] - metrics["result_baseline_accuracy"]
        metrics["over_accuracy_lift"] = metrics["over_accuracy"] - metrics["over_baseline_accuracy"]
        metrics["btts_accuracy_lift"] = metrics["btts_accuracy"] - metrics["btts_baseline_accuracy"]

        result_prob = result_pipeline.predict_proba(X.loc[test_idx, market_features["result"]])
        result_quality = multiclass_probability_metrics(
            y_result_test,
            result_prob,
            result_pipeline.classes_,
        )
        metrics["result_auc"] = result_quality["roc_auc"]
        metrics["result_brier_score"] = result_quality["brier_score"]
        metrics["result_calibration_error"] = result_quality["calibration_error"]
        metrics["result"] = {
            **result_quality,
            "accuracy": metrics["result_accuracy"],
            "baseline_accuracy": metrics["result_baseline_accuracy"],
            "accuracy_lift": metrics["result_accuracy_lift"],
        }

        over_prob = over_model.predict_proba(X.loc[test_idx, market_features["over_2_5"]])[:, 1]
        btts_prob = btts_model.predict_proba(X.loc[test_idx, market_features["btts"]])[:, 1]
        metrics["over_brier_score"] = float(brier_score_loss(y_over_test, over_prob))
        metrics["btts_brier_score"] = float(brier_score_loss(y_btts_test, btts_prob))
        metrics["over_calibration_error"] = expected_calibration_error(y_over_test, over_prob)
        metrics["btts_calibration_error"] = expected_calibration_error(y_btts_test, btts_prob)

        try:
            metrics["over_auc"] = float(roc_auc_score(y_over_test, over_prob))
        except ValueError:
            metrics["over_auc"] = 0.0

        try:
            metrics["btts_auc"] = float(roc_auc_score(y_btts_test, btts_prob))
        except ValueError:
            metrics["btts_auc"] = 0.0

        qualified_markets = []
        if metrics["result_auc"] >= 0.52 and metrics["result_accuracy_lift"] > 0:
            qualified_markets.append("result")
        if metrics["over_auc"] >= 0.52 and metrics["over_accuracy_lift"] >= 0:
            qualified_markets.append("over_2_5")
        if metrics["btts_auc"] >= 0.52 and metrics["btts_accuracy_lift"] >= 0:
            qualified_markets.append("btts")
        candidate_qualified = bool(qualified_markets)
        model_version = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "model_version": model_version,
            "status": "candidate_qualified" if candidate_qualified else "validation_failed",
            "feature_columns": self.feature_columns,
            "dataset_rows": int(len(clean_df)),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "validation_strategy": validation_strategy,
            "metrics": metrics,
            "qualified_markets": qualified_markets,
            "calibration": "sigmoid_cv3",
            "selected_feature_limits": selected_limits,
            "market_feature_columns": market_features,
            "inconsistency_filter_threshold": threshold,
        }

        candidate_dir = self.model_dir / "versions" / "football" / model_version
        candidate_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result_pipeline, candidate_dir / "result_model.joblib")
        joblib.dump(over_model, candidate_dir / "over25_model.joblib")
        joblib.dump(btts_model, candidate_dir / "btts_model.joblib")
        with open(candidate_dir / "metadata.json", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, ensure_ascii=False)

        TrainingRunsRepository().save_training_run("football", metadata)
        return metadata
