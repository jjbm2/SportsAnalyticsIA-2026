from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from core.logger import logger
from engines.formula1_prediction_engine import Formula1PredictionEngine
from machine_learning.features.formula1_features import Formula1Features


class Formula1Predictor:
    """Modelo ML de podio con fallback al motor estadístico existente."""

    def __init__(self, model_dir: Path | None = None) -> None:
        root = model_dir or Path("machine_learning/models_store")
        explicit_candidate = model_dir is not None
        self.engine = Formula1PredictionEngine()
        self.features = Formula1Features()
        self.available = False
        self.metadata: dict[str, Any] = {}
        try:
            metadata_path = root / ("metadata.json" if explicit_candidate else "formula1_metadata.json")
            model_path = root / ("podium_model.joblib" if explicit_candidate else "formula1_podium_model.joblib")
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            allowed_statuses = {"candidate_qualified"} if explicit_candidate else {"active"}
            if self.metadata.get("status") not in allowed_statuses:
                return
            self.model = joblib.load(model_path)
            self.available = True
        except (FileNotFoundError, OSError, ValueError) as error:
            logger.warning("Modelo F1 no disponible; se usará fallback: %s", error)

    def analyze_match(self, selected_match: dict[str, Any], simulations: int, force_refresh: bool = False) -> dict[str, Any]:
        fallback = self.engine.analyze_match(selected_match, simulations, force_refresh)
        if not self.available:
            return fallback

        season = int(selected_match.get("season") or str(selected_match.get("date"))[:4])
        target_round = int(selected_match.get("round") or 99)
        current = self.engine.api.get_results(season, force_refresh)
        previous = self.engine.api.get_results(season - 1, force_refresh)
        completed = previous + [race for race in current if int(race.get("round") or 0) < target_round]
        rows = self._feature_rows(
            completed, target_round,
            selected_match.get("circuit_id") or selected_match.get("away_id"),
        )
        if rows.empty:
            return fallback

        columns = self.metadata.get("feature_columns") or self.features.feature_columns()
        rows["podium_probability"] = self.model.predict_proba(rows[columns])[:, 1] * 100
        total = float(rows["podium_probability"].sum())
        if total > 0:
            rows["podium_probability"] *= 300.0 / total
        probabilities = dict(zip(rows["driver_name"], rows["podium_probability"]))

        for market in fallback["markets_to_save"]:
            if market.get("market_type") == "f1_podium" and market.get("selection") in probabilities:
                market["probability"] = min(float(probabilities[market["selection"]]), 100.0)
                market["confidence"] = "Validada"
        for row in fallback["markets"]:
            name = row.get("Piloto")
            if name in probabilities:
                row["Podio"] = f"{min(float(probabilities[name]), 100.0):.1f}%"

        fallback["model_name"] = "F1 ML validada + Monte Carlo"
        fallback["extra_metrics"]["Modelo"] = "ML de podio + Monte Carlo"
        fallback["context_json"].update({
            "model_version": self.metadata.get("model_version"),
            "quality_gate": {"podium": True},
            "drivers_with_ml": len(rows),
        })
        return fallback

    def _feature_rows(self, races: list[dict[str, Any]], target_round: int, target_circuit: Any) -> pd.DataFrame:
        driver_history: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
        constructor_history: defaultdict[str, list[float]] = defaultdict(list)
        circuit_history: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
        names: dict[str, str] = {}
        latest_constructor: dict[str, str] = {}
        circuit_id = str(target_circuit or "")
        for race in races:
            race_circuit = str(((race.get("Circuit") or {}).get("circuitId") or ""))
            for result in race.get("Results", []):
                driver_id, constructor_id = self.features.result_identity(result)
                if not driver_id or not constructor_id:
                    continue
                try:
                    position = float(result.get("position") or 20)
                    points = float(result.get("points") or 0)
                except (TypeError, ValueError):
                    continue
                driver = result.get("Driver") or {}
                names[driver_id] = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
                latest_constructor[driver_id] = constructor_id
                driver_history[driver_id].append({"position": position, "points": points})
                constructor_history[constructor_id].append(position)
                circuit_history[(driver_id, race_circuit)].append(position)

        output: list[dict[str, Any]] = []
        for driver_id, history in driver_history.items():
            if len(history) < 3:
                continue
            values = self.features.build(
                history,
                constructor_history[latest_constructor[driver_id]],
                circuit_history[(driver_id, circuit_id)],
                target_round,
            )
            output.append({**values, "driver_name": names[driver_id]})
        return pd.DataFrame(output)
