from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.calibration import FootballProbabilityCalibrator


def _result_probabilities(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    classes = [-1, 0, 1]
    rows = [ast.literal_eval(value) for value in frame["backtest_result_probabilities"]]
    probabilities = np.asarray([[row.get(str(label), 0.0) for label in classes] for row in rows])
    return probabilities, frame["backtest_result_actual"].to_numpy(dtype=int), classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra probabilidades de fútbol sin alterar el modelo base")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model-version", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/calibration/football"))
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset).sort_values("game_date", kind="stable")
    markets: dict[str, dict] = {}
    reports = {}
    specifications = {
        "result": _result_probabilities(frame.dropna(subset=["backtest_result_probabilities"])),
        "over_2_5": (
            frame.dropna(subset=["backtest_over_2_5_probability"])["backtest_over_2_5_probability"].to_numpy(),
            frame.dropna(subset=["backtest_over_2_5_probability"])["backtest_over_2_5_actual"].to_numpy(dtype=int),
            [0, 1],
        ),
        "btts": (
            frame.dropna(subset=["backtest_btts_probability"])["backtest_btts_probability"].to_numpy(),
            frame.dropna(subset=["backtest_btts_probability"])["backtest_btts_actual"].to_numpy(dtype=int),
            [0, 1],
        ),
    }
    for market, (probabilities, targets, classes) in specifications.items():
        bundle, report = FootballProbabilityCalibrator.fit_market(probabilities, targets, classes)
        reports[market] = report
        if bundle is not None:
            markets[market] = bundle

    calibrator = FootballProbabilityCalibrator(args.base_model_version, markets)
    calibrator.save(
        args.output_dir / "football_calibrator.joblib",
        args.output_dir / "calibration_metrics.json",
        reports,
    )
    print(json.dumps({"calibrated_markets": list(markets), "metrics": reports}, indent=2))


if __name__ == "__main__":
    main()
