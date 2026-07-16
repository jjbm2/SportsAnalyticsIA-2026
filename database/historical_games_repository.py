from __future__ import annotations

from pathlib import Path

import pandas as pd


class HistoricalGamesRepository:
    def __init__(self):
        self.base_dir = Path("data/ml_datasets")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_dataframe(
        self,
        dataframe: pd.DataFrame,
        filename: str,
    ) -> Path:
        output_path = self.base_dir / filename
        dataframe.to_csv(output_path, index=False)
        return output_path

    def load_dataframe(
        self,
        filename: str,
    ) -> pd.DataFrame:
        input_path = self.base_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"No existe el archivo {input_path}")
        return pd.read_csv(input_path)
