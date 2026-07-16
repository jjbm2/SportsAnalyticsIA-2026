from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from machine_learning.features.formula1_features import Formula1Features
from services.formula1_api import Formula1API


class Formula1DatasetBuilder:
    def __init__(self) -> None:
        self.api = Formula1API()

    def build_dataset(
        self,
        seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
        min_history: int = 3,
        force_refresh: bool = False,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        driver_history: dict[str, list[dict[str, float]]] = defaultdict(list)
        constructor_history: dict[str, list[float]] = defaultdict(list)
        circuit_history: dict[tuple[str, str], list[float]] = defaultdict(list)

        for season in seasons:
            races = self.api.get_results(season, force_refresh)
            races.sort(key=lambda race: int(race.get("round") or 0))
            for race in races:
                round_number = int(race.get("round") or 0)
                circuit_id = ((race.get("Circuit") or {}).get("circuitId") or "unknown")
                results = [item for item in race.get("Results", []) if isinstance(item, dict)]
                total_rounds = max(len(races), 1)
                for result in results:
                    driver_id, constructor_id = Formula1Features.result_identity(result)
                    if not driver_id or not constructor_id:
                        continue
                    try:
                        position = float(result.get("position"))
                        points = float(result.get("points") or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if len(driver_history[driver_id]) >= min_history:
                        features = Formula1Features.build(
                            driver_history[driver_id],
                            constructor_history[constructor_id],
                            circuit_history[(driver_id, circuit_id)],
                            round_number,
                            total_rounds,
                        )
                        rows.append({
                            "season": season,
                            "round": round_number,
                            "race_name": race.get("raceName"),
                            "driver_id": driver_id,
                            "constructor_id": constructor_id,
                            "circuit_id": circuit_id,
                            "position": position,
                            "podium_label": int(position <= 3),
                            **features,
                        })
                    driver_history[driver_id].append({"position": position, "points": points})
                    constructor_history[constructor_id].append(position)
                    circuit_history[(driver_id, circuit_id)].append(position)

        dataset = pd.DataFrame(rows)
        if save_csv and not dataset.empty:
            path = Path("data/ml_datasets/formula1_2021_2025.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(path, index=False)
        return dataset
