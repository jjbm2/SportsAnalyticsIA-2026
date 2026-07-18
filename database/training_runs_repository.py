from __future__ import annotations

import json
from core.time_utils import utc_now
from pathlib import Path
from typing import Any


class TrainingRunsRepository:
    def __init__(self):
        self.base_dir = Path("data/training_runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_training_run(
        self,
        sport: str,
        metadata: dict[str, Any],
    ) -> Path:
        timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
        output_path = self.base_dir / f"{sport.lower()}_training_{timestamp}.json"

        payload = {
            "sport": sport,
            "saved_at": utc_now().isoformat(),
            "metadata": metadata,
        }

        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)

        return output_path
