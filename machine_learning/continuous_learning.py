from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from database.database import get_session
from database.models import PostMatchReview, PredictionRun
from machine_learning.model_registry import MODEL_FILES, ModelRegistry
from machine_learning.shadow_validation import ShadowValidationService
from core.logger import logger


MIN_INCREMENTAL_EXAMPLES = 100
MIN_NEW_EXAMPLES_FOR_RETRAIN = 25
_START_LOCK = threading.Lock()
_STARTED_DAYS: set[str] = set()


def start_continuous_learning() -> bool:
    """Run the safe daily review off the Streamlit rendering thread."""
    today = date.today().isoformat()
    with _START_LOCK:
        if today in _STARTED_DAYS:
            return False
        _STARTED_DAYS.add(today)

    def worker() -> None:
        try:
            summary = ContinuousLearningService().run_daily()
            logger.info("Aprendizaje continuo revisado: %s", summary)
        except Exception as error:
            logger.exception("No se pudo completar el aprendizaje continuo: %s", error)

    threading.Thread(
        target=worker,
        name="sports-analytics-continuous-learning",
        daemon=True,
    ).start()
    return True


class ContinuousLearningService:
    """Prepare reviewed examples and apply only fully evidenced promotions."""

    def __init__(self, dataset_root: Path | None = None) -> None:
        self.dataset_root = dataset_root or Path("data/ml_datasets/incremental")
        self.state_path = Path("data/training_runs/continuous_learning_state.json")

    def run_daily(self) -> dict[str, Any]:
        examples = self._reviewed_examples()
        exported = self._write_datasets(examples)
        training = self._train_ready_candidates(examples, exported)
        promotions = self._review_candidates()
        return {
            "examples": len(examples),
            "exported": exported,
            "ready_for_training": {
                sport: count >= MIN_INCREMENTAL_EXAMPLES
                for sport, count in exported.items()
            },
            "training": training,
            "promotions": promotions,
        }

    def _reviewed_examples(self) -> list[dict[str, Any]]:
        session = get_session()
        try:
            rows = (
                session.query(PostMatchReview, PredictionRun)
                .join(PredictionRun, PredictionRun.id == PostMatchReview.prediction_run_id)
                .order_by(PostMatchReview.evaluated_at.asc())
                .all()
            )
            examples = []
            for review, run in rows:
                context = run.context_json or {}
                features = context.get("features")
                if isinstance(features, list):
                    features = features[0] if features else None
                if not isinstance(features, dict) or not features:
                    continue
                if review.home_score is None or review.away_score is None:
                    continue
                examples.append(
                    {
                        "example_id": f"{run.sport}:{review.match_id}:{run.id}",
                        "sport": run.sport,
                        "match_id": str(review.match_id),
                        "prediction_run_id": run.id,
                        "model_version": str(context.get("model_version") or "legacy"),
                        "home_score": float(review.home_score),
                        "away_score": float(review.away_score),
                        "actual_outcome": review.actual_outcome,
                        "evaluated_at": review.evaluated_at.isoformat(),
                        "match_metadata": context.get("match_metadata") or {},
                        "features": features,
                    }
                )
            combined = {item["example_id"]: item for item in examples}
            combined.update(self._backtest_examples())
            return list(combined.values())
        finally:
            session.close()

    @staticmethod
    def _backtest_examples() -> dict[str, dict[str, Any]]:
        examples: dict[str, dict[str, Any]] = {}
        for path in Path("data/backtesting").glob("*/*/training_examples.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(item, dict) or not item.get("example_id"):
                    continue
                features = item.get("features")
                if not isinstance(features, dict) or not features:
                    continue
                examples[str(item["example_id"])] = item
        return examples

    def _write_datasets(self, examples: list[dict[str, Any]]) -> dict[str, int]:
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for example in examples:
            sport_key = self._sport_key(example["sport"])
            if not sport_key:
                continue
            grouped.setdefault(sport_key, {})[example["example_id"]] = example

        counts: dict[str, int] = {}
        for sport, rows in grouped.items():
            path = self.dataset_root / f"{sport}.jsonl"
            existing = self._read_jsonl(path)
            existing.update(rows)
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix(".jsonl.pending")
            pending.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                    for _, item in sorted(existing.items())
                ),
                encoding="utf-8",
            )
            pending.replace(path)
            counts[sport] = len(existing)
        return counts

    def _train_ready_candidates(
        self,
        examples: list[dict[str, Any]],
        counts: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        """Train only supported candidates after enough new reviewed examples."""
        state = self._read_state()
        results: dict[str, dict[str, Any]] = {}
        for sport in ("football", "baseball", "basketball", "nfl"):
            sport_examples = [item for item in examples if self._sport_key(item.get("sport")) == sport]
            if sport != "football":
                sport_examples = [item for item in sport_examples if self._supported_league(sport, item)]
            count = len(sport_examples)
            previous_count = int((state.get(sport) or {}).get("examples", 0))
            if count < MIN_INCREMENTAL_EXAMPLES:
                results[sport] = {"status": "waiting_for_samples", "examples": count}
                continue
            if count - previous_count < MIN_NEW_EXAMPLES_FOR_RETRAIN:
                results[sport] = {"status": "waiting_for_new_samples", "examples": count}
                continue
            try:
                metadata = self._train_sport(sport, sport_examples)
            except (ValueError, TypeError, OSError) as error:
                results[sport] = {
                    "status": "validation_blocked",
                    "examples": count,
                    "reason": str(error),
                }
                continue
            state[sport] = {"examples": count, "model_version": metadata["model_version"]}
            self._write_state(state)
            results[sport] = {
                "status": "candidate_created",
                "examples": count,
                "model_version": metadata["model_version"],
                "qualified_markets": metadata.get("qualified_markets", []),
            }
        return results

    def _train_sport(
        self, sport: str, examples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        rows = [self._training_row(sport, item) for item in examples]
        incremental = pd.DataFrame(rows)
        if sport == "football":
            from machine_learning.trainers.football_trainer import FootballTrainer

            return FootballTrainer().train_from_dataframe(incremental)

        historical_path = self._historical_dataset_path(sport)
        historical = pd.read_csv(historical_path)
        combined = pd.concat([historical, incremental], ignore_index=True, sort=False)
        latest_date = pd.to_datetime(incremental["game_date"], errors="coerce").max()
        if pd.isna(latest_date):
            raise ValueError("Los ejemplos incrementales no tienen fechas válidas.")
        if sport == "baseball":
            from machine_learning.trainers.baseball_trainer import BaseballTrainer

            return BaseballTrainer().train_from_dataframe(
                combined, test_season=int(incremental["season"].max())
            )
        if sport == "basketball":
            from machine_learning.trainers.basketball_trainer import BasketballTrainer

            season = str(incremental["season"].max())
            return BasketballTrainer().train_from_dataframe(combined, test_season=season)
        from machine_learning.trainers.nfl_trainer import NFLTrainer

        return NFLTrainer().train_from_dataframe(
            combined, test_season=int(incremental["season"].max())
        )

    @staticmethod
    def _football_training_row(example: dict[str, Any]) -> dict[str, Any]:
        home = float(example["home_score"])
        away = float(example["away_score"])
        return {
            **dict(example["features"]),
            "game_date": example.get("evaluated_at"),
            "result_label": 1 if home > away else -1 if away > home else 0,
            "over_25_label": int(home + away > 2.5),
            "btts_label": int(home > 0 and away > 0),
        }

    @classmethod
    def _training_row(cls, sport: str, example: dict[str, Any]) -> dict[str, Any]:
        if sport == "football":
            return cls._football_training_row(example)
        home = float(example["home_score"])
        away = float(example["away_score"])
        evaluated_at = str(example.get("evaluated_at") or "")
        year = int(evaluated_at[:4])
        stored_season = (example.get("match_metadata") or {}).get("season")
        base = {**dict(example["features"]), "game_date": evaluated_at}
        if sport == "baseball":
            season = cls._numeric_season(stored_season, year)
            return {**base, "season": season, "home_win_label": int(home > away), "over_85_label": int(home + away > 8.5), "home_over_35_label": int(home > 3.5)}
        if sport == "basketball":
            month = int(evaluated_at[5:7])
            season = str(stored_season) if stored_season else (f"{year}-{year + 1}" if month >= 9 else f"{year - 1}-{year}")
            return {**base, "season": season, "home_win_label": int(home > away), "over_2195_label": int(home + away > 219.5), "home_over_1095_label": int(home > 109.5)}
        season = cls._numeric_season(stored_season, year)
        return {**base, "season": season, "home_win_label": int(home > away), "over_415_label": int(home + away > 41.5), "home_over_205_label": int(home > 20.5)}

    @staticmethod
    def _numeric_season(value: Any, fallback: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _supported_league(sport: str, example: dict[str, Any]) -> bool:
        league = str((example.get("match_metadata") or {}).get("league") or "").upper()
        return league == {"baseball": "MLB", "basketball": "NBA", "nfl": "NFL"}[sport]

    @staticmethod
    def _historical_dataset_path(sport: str) -> Path:
        root = Path("data/ml_datasets")
        if sport == "baseball":
            candidates = sorted(root.glob("baseball_mlb_seasons_*.csv"))
            if not candidates:
                raise FileNotFoundError("No existe dataset histórico MLB.")
            return candidates[-1]
        path = root / {"basketball": "basketball_nba_2022_2025.csv", "nfl": "nfl_2022_2024.csv"}[sport]
        if not path.exists():
            raise FileNotFoundError(f"No existe dataset histórico para {sport}.")
        return path

    def _read_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.state_path.with_suffix(".json.pending")
        pending.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pending.replace(self.state_path)

    def _review_candidates(self) -> dict[str, dict[str, Any]]:
        shadow = ShadowValidationService()
        registry = ModelRegistry()
        results: dict[str, dict[str, Any]] = {}
        for sport_label, sport_key in shadow.sport_dirs.items():
            if sport_key not in MODEL_FILES:
                continue
            candidate = shadow.find_candidate(sport_label)
            if candidate is None:
                continue
            decision = registry.safe_automatic_recommendation(sport_key, candidate.name)
            promoted = False
            if decision["decision"] == "promote":
                registry.promote_automatically_if_safe(sport_key, candidate.name)
                promoted = True
            results[sport_key] = {
                "candidate_version": candidate.name,
                "decision": decision["decision"],
                "reason": decision["reason"],
                "promoted": promoted,
            }
        return results

    @staticmethod
    def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict) and item.get("example_id"):
                rows[str(item["example_id"])] = item
        return rows

    @staticmethod
    def _sport_key(label: str) -> str | None:
        return {
            "Fútbol": "football",
            "Béisbol": "baseball",
            "Basketball": "basketball",
            "NFL": "nfl",
            "Fórmula 1": "formula1",
        }.get(label)
