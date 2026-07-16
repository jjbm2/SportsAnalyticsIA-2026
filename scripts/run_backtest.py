from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.backtesting import WalkForwardBacktester


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtesting temporal sin fuga de datos")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sport", required=True, choices=("football", "baseball", "basketball", "nfl"))
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--refit-every", type=int, default=20)
    args = parser.parse_args()
    result = WalkForwardBacktester().run(
        args.dataset, args.sport, args.league, args.season,
        min_train=args.min_train, refit_every=args.refit_every,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
