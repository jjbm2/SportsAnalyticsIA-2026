from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from machine_learning.model_quality import expected_calibration_error


@dataclass(frozen=True)
class BacktestSchema:
    sport_label: str
    id_column: str
    home_score: str
    away_score: str
    targets: dict[str, str]


SCHEMAS = {
    "football": BacktestSchema(
        "Fútbol", "fixture_id", "home_goals", "away_goals",
        {"result": "result_label", "over_2_5": "over_25_label", "btts": "btts_label"},
    ),
    "baseball": BacktestSchema(
        "Béisbol", "game_id", "home_score", "away_score",
        {"home_win": "home_win_label", "over_8_5": "over_85_label", "home_over_3_5": "home_over_35_label"},
    ),
    "basketball": BacktestSchema(
        "Basketball", "game_id", "home_score", "away_score",
        {"home_win": "home_win_label", "over_219_5": "over_2195_label", "home_over_109_5": "home_over_1095_label"},
    ),
    "nfl": BacktestSchema(
        "NFL", "game_id", "home_score", "away_score",
        {"home_win": "home_win_label", "over_41_5": "over_415_label", "home_over_20_5": "home_over_205_label"},
    ),
}


class WalkForwardBacktester:
    """Expanding-window backtest. Every prediction uses strictly earlier rows."""

    def __init__(self, output_root: Path | None = None) -> None:
        self.output_root = output_root or Path("data/backtesting")

    def run(
        self,
        dataset_path: Path,
        sport: str,
        league: str,
        season: str,
        min_train: int = 60,
        refit_every: int = 20,
    ) -> dict[str, Any]:
        if sport not in SCHEMAS:
            raise ValueError(f"Deporte no soportado para backtesting: {sport}")
        if min_train < 20 or refit_every < 1:
            raise ValueError("min_train debe ser >=20 y refit_every >=1.")
        schema = SCHEMAS[sport]
        data = pd.read_csv(dataset_path)
        self._validate(data, schema)
        data = data.copy()
        data["_parsed_date"] = pd.to_datetime(data["game_date"], errors="coerce", utc=True)
        data = data.dropna(subset=["_parsed_date"]).sort_values(
            ["_parsed_date", schema.id_column], kind="stable"
        ).reset_index(drop=True)

        excluded = {
            "_parsed_date", "game_date", "league_id", "season", schema.id_column,
            "home_team_id", "away_team_id", schema.home_score, schema.away_score,
            *schema.targets.values(),
        }
        if sport == "football":
            excluded.update({
                "home_matches_played", "away_matches_played",
                "home_recent_sample_size", "away_recent_sample_size",
            })
        features = [
            column for column in data.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
        ]
        if not features:
            raise ValueError("El dataset no contiene features numéricas.")

        predictions: list[dict[str, Any]] = []
        models: dict[str, Pipeline] = {}
        last_fit = -1
        for index in range(min_train, len(data)):
            if not models or index - last_fit >= refit_every:
                history = data.iloc[:index]
                models = self._fit_models(history, features, schema)
                last_fit = index
            if len(models) != len(schema.targets):
                continue
            row = data.iloc[index]
            prediction = {
                "row_index": index,
                "match_id": str(row[schema.id_column]),
                "game_date": row["_parsed_date"].isoformat(),
                "training_rows": index,
            }
            for market, target in schema.targets.items():
                model = models[market]
                market_features = list(model.feature_names_in_)
                probabilities = model.predict_proba(data.loc[[index], market_features])[0]
                classes = model.classes_
                actual = row[target]
                predicted_class = classes[int(np.argmax(probabilities))]
                prediction[f"{market}_actual"] = int(actual)
                prediction[f"{market}_prediction"] = int(predicted_class)
                prediction[f"raw_{market}_prediction"] = int(predicted_class)
                if len(classes) == 2:
                    positive_index = int(np.where(classes == 1)[0][0])
                    prediction[f"{market}_probability"] = float(probabilities[positive_index])
                    prediction[f"raw_{market}_probability"] = float(probabilities[positive_index])
                else:
                    actual_index = int(np.where(classes == actual)[0][0])
                    prediction[f"{market}_probability_actual"] = float(probabilities[actual_index])
                    prediction[f"{market}_probabilities"] = {
                        str(int(label)): float(value)
                        for label, value in zip(classes, probabilities)
                    }
                    prediction[f"raw_{market}_probabilities"] = dict(
                        prediction[f"{market}_probabilities"]
                    )
            if sport == "football":
                self._apply_football_pipeline(prediction, row, predictions)
            predictions.append(prediction)

        enriched = data.drop(columns=["_parsed_date"]).copy()
        prediction_frame = pd.DataFrame(predictions).set_index("row_index") if predictions else pd.DataFrame()
        for column in prediction_frame.columns:
            if column not in {"match_id", "game_date"}:
                enriched.loc[prediction_frame.index, f"backtest_{column}"] = prediction_frame[column]

        metrics = self._metrics(predictions, schema)
        output_dir = self.output_root / sport / self._slug(f"{league}_{season}")
        output_dir.mkdir(parents=True, exist_ok=True)
        enriched_path = output_dir / "enriched_dataset.csv"
        examples_path = output_dir / "training_examples.jsonl"
        metrics_path = output_dir / "metrics.json"
        enriched.to_csv(enriched_path, index=False)
        examples = self._examples(data, predictions, features, schema, league, season)
        examples_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in examples),
            encoding="utf-8",
        )
        summary = {
            "sport": sport, "league": league, "season": str(season),
            "source_rows": len(data), "predicted_rows": len(predictions),
            "feature_count": len(features), "min_train": min_train,
            "refit_every": refit_every, "metrics": metrics,
            "selected_features": {
                market: self._selected_feature_names(model)
                for market, model in models.items()
            },
            "enriched_dataset": str(enriched_path), "training_examples": str(examples_path),
        }
        metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    @staticmethod
    def _fit_models(data: pd.DataFrame, features: list[str], schema: BacktestSchema) -> dict[str, Pipeline]:
        models = {}
        for market, target in schema.targets.items():
            if data[target].nunique() < 2:
                continue
            market_features = WalkForwardBacktester._market_features(features, market)
            limits = {"result": 24, "over_2_5": 24, "btts": 18}
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif, k=min(limits.get(market, 30), len(market_features)))),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1200, class_weight="balanced")),
            ])
            if {"home_scored_std", "away_scored_std"}.issubset(data.columns):
                variability = (
                    pd.to_numeric(data["home_scored_std"], errors="coerce").fillna(1.0)
                    + pd.to_numeric(data["away_scored_std"], errors="coerce").fillna(1.0)
                ) / 2.0
            else:
                variability = pd.Series(1.0, index=data.index)
            sample_weight = np.clip(1.25 - variability / 3.0, 0.35, 1.0)
            model.fit(data[market_features], data[target], model__sample_weight=sample_weight)
            models[market] = model
        return models

    @staticmethod
    def _market_features(features: list[str], market: str) -> list[str]:
        if market == "result":
            excluded_tokens = ("avg_total", "btts_rate")
        elif market == "over_2_5":
            excluded_tokens = ("win_rate", "draw_rate", "points", "streak", "home_advantage")
        elif market == "btts":
            excluded_tokens = ("win_rate", "draw_rate", "points", "streak", "home_advantage")
        else:
            excluded_tokens = ()
        selected = [name for name in features if not any(token in name for token in excluded_tokens)]
        return selected or features

    @staticmethod
    def _selected_feature_names(model: Pipeline) -> list[str]:
        names = np.asarray(model.feature_names_in_)
        variance = model.named_steps["variance"].get_support()
        names = names[variance]
        selector = model.named_steps["select"]
        return names[selector.get_support()].tolist()

    @classmethod
    def _apply_football_pipeline(
        cls, prediction: dict[str, Any], row: pd.Series, prior_predictions: list[dict[str, Any]]
    ) -> None:
        fallback = cls._football_fallback(row)
        decisions = {}
        for market in ("result", "over_2_5", "btts"):
            quality = cls._raw_market_quality(prior_predictions, market)
            qualified = bool(quality.get("qualified"))
            ml_weight = float(quality.get("weight", 0.0)) if qualified else 0.0
            decisions[market] = {**quality, "ml_weight": ml_weight, "fallback_weight": 1.0 - ml_weight}
            if market == "result":
                raw = prediction["raw_result_probabilities"]
                final = {
                    label: ml_weight * float(raw.get(label, 0.0)) + (1.0 - ml_weight) * fallback[market][label]
                    for label in ("-1", "0", "1")
                }
                prediction["result_probabilities"] = final
                prediction["result_prediction"] = int(max(final, key=final.get))
                actual = str(prediction["result_actual"])
                prediction["result_probability_actual"] = final[actual]
            else:
                raw_probability = float(prediction[f"raw_{market}_probability"])
                final_probability = ml_weight * raw_probability + (1.0 - ml_weight) * fallback[market]
                prediction[f"{market}_probability"] = final_probability
                prediction[f"{market}_prediction"] = int(final_probability >= 0.5)
        prediction["pipeline_decisions"] = decisions

    @staticmethod
    def _football_fallback(row: pd.Series) -> dict[str, Any]:
        def value(name: str, default: float) -> float:
            raw = row.get(name, default)
            return default if pd.isna(raw) else float(raw)

        home_lambda = np.clip(
            (value("home_avg_scored", 1.35) + value("away_avg_conceded", 1.35)) / 2.0,
            0.20, 4.0,
        )
        away_lambda = np.clip(
            (value("away_avg_scored", 1.10) + value("home_avg_conceded", 1.10)) / 2.0,
            0.20, 4.0,
        )
        home_goals = np.asarray([
            math.exp(-home_lambda) * home_lambda ** goals / math.factorial(goals)
            for goals in range(11)
        ])
        away_goals = np.asarray([
            math.exp(-away_lambda) * away_lambda ** goals / math.factorial(goals)
            for goals in range(11)
        ])
        score_matrix = np.outer(home_goals, away_goals)
        score_matrix /= score_matrix.sum()
        result = {
            "1": float(np.tril(score_matrix, -1).sum()),
            "0": float(np.trace(score_matrix)),
            "-1": float(np.triu(score_matrix, 1).sum()),
        }
        over = float(sum(
            score_matrix[home, away]
            for home in range(11) for away in range(11) if home + away > 2
        ))
        btts = float(score_matrix[1:, 1:].sum())
        return {"result": result, "over_2_5": over, "btts": btts}

    @classmethod
    def _raw_market_quality(
        cls, predictions: list[dict[str, Any]], market: str
    ) -> dict[str, Any]:
        if len(predictions) < 30:
            return {"qualified": False, "reason": "insufficient_history", "evaluated": len(predictions)}
        actual = [row[f"{market}_actual"] for row in predictions]
        predicted = [row[f"raw_{market}_prediction"] for row in predictions]
        accuracy = float(accuracy_score(actual, predicted))
        baseline = max(actual.count(label) for label in set(actual)) / len(actual)
        try:
            if market == "result":
                labels = [-1, 0, 1]
                probabilities = np.asarray([
                    [row["raw_result_probabilities"].get(str(label), 0.0) for label in labels]
                    for row in predictions
                ])
                auc = float(roc_auc_score(actual, probabilities, labels=labels, multi_class="ovr", average="macro"))
                minimum_auc = 0.60
            else:
                probability = [row[f"raw_{market}_probability"] for row in predictions]
                auc = float(roc_auc_score(actual, probability))
                minimum_auc = 0.52
        except ValueError:
            auc = 0.0
            minimum_auc = 1.0
        qualified = auc >= minimum_auc and accuracy > baseline
        weight = min(0.70, 0.50 + max(0.0, auc - 0.50)) if qualified else 0.0
        return {
            "qualified": qualified, "evaluated": len(actual), "roc_auc": auc,
            "accuracy": accuracy, "baseline_accuracy": baseline, "weight": weight,
            "reason": "historical_quality_passed" if qualified else "historical_quality_rejected",
        }

    @staticmethod
    def _metrics(predictions: list[dict[str, Any]], schema: BacktestSchema) -> dict[str, Any]:
        metrics = {}
        for market in schema.targets:
            actual = [row[f"{market}_actual"] for row in predictions]
            predicted = [row[f"{market}_prediction"] for row in predictions]
            item: dict[str, Any] = {"evaluated": len(actual), "accuracy": float(accuracy_score(actual, predicted)) if actual else None}
            probability_key = f"{market}_probability"
            if actual and probability_key in predictions[0]:
                probability = [row[probability_key] for row in predictions]
                item.update(
                    brier_score=float(brier_score_loss(actual, probability)),
                    calibration_error=expected_calibration_error(actual, probability),
                )
                try:
                    item["roc_auc"] = float(roc_auc_score(actual, probability))
                except ValueError:
                    item["roc_auc"] = None
            elif actual and f"{market}_probabilities" in predictions[0]:
                labels = sorted(set(actual))
                probability_matrix = np.asarray([
                    [row[f"{market}_probabilities"].get(str(label), 0.0) for label in labels]
                    for row in predictions
                ])
                expected = np.asarray([
                    [1.0 if value == label else 0.0 for label in labels]
                    for value in actual
                ])
                item["brier_score"] = float(np.mean(np.sum((probability_matrix - expected) ** 2, axis=1)))
                confidence = probability_matrix.max(axis=1)
                correctness = (np.asarray(actual) == np.asarray(predicted)).astype(int)
                item["calibration_error"] = expected_calibration_error(correctness, confidence)
                try:
                    item["roc_auc"] = float(roc_auc_score(
                        actual,
                        probability_matrix,
                        labels=labels,
                        multi_class="ovr",
                        average="macro",
                    ))
                except ValueError:
                    item["roc_auc"] = None
            metrics[market] = item
        return metrics

    @staticmethod
    def _examples(
        data: pd.DataFrame, predictions: list[dict[str, Any]], features: list[str],
        schema: BacktestSchema, league: str, season: str,
    ) -> list[dict[str, Any]]:
        examples = []
        for prediction in predictions:
            row = data.iloc[int(prediction["row_index"])]
            examples.append({
                "example_id": f"backtest:{schema.sport_label}:{league}:{season}:{prediction['match_id']}",
                "sport": schema.sport_label, "match_id": prediction["match_id"],
                "prediction_run_id": None, "model_version": "walk_forward_backtest",
                "home_score": float(row[schema.home_score]), "away_score": float(row[schema.away_score]),
                "actual_outcome": "home_win" if row[schema.home_score] > row[schema.away_score] else "away_win" if row[schema.home_score] < row[schema.away_score] else "draw",
                "evaluated_at": prediction["game_date"],
                "match_metadata": {
                    "league": league,
                    "season": str(row.get("season", season)),
                    "source": "backtest",
                },
                "features": {column: None if pd.isna(row[column]) else float(row[column]) for column in features},
                "backtest_prediction": prediction,
            })
        return examples

    @staticmethod
    def _validate(data: pd.DataFrame, schema: BacktestSchema) -> None:
        required = {"game_date", schema.id_column, schema.home_score, schema.away_score, *schema.targets.values()}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
