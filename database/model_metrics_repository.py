from __future__ import annotations

from typing import Any

from database.database import get_session
from database.models import PostMatchReview, PredictionRun


class ModelMetricsRepository:
    def get_paired_model_performance(
        self,
        sport: str,
        active_version: str,
        candidate_version: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Compare versions only on matches evaluated by both models."""
        session = get_session()
        try:
            rows = (
                session.query(PostMatchReview, PredictionRun)
                .join(PredictionRun, PredictionRun.id == PostMatchReview.prediction_run_id)
                .filter(PostMatchReview.sport == sport)
                .order_by(PostMatchReview.evaluated_at.asc())
                .all()
            )
        finally:
            session.close()

        by_match_version: dict[tuple[str, str], tuple[Any, Any]] = {}
        for review, run in rows:
            version = str((run.context_json or {}).get("model_version") or "legacy")
            if version in {str(active_version), str(candidate_version)}:
                by_match_version[(str(review.match_id), version)] = (review, run)

        active_matches = {
            match_id for match_id, version in by_match_version if version == str(active_version)
        }
        candidate_matches = {
            match_id for match_id, version in by_match_version if version == str(candidate_version)
        }
        paired_matches = active_matches & candidate_matches
        if not paired_matches:
            return None, None

        active_rows = [by_match_version[(match_id, str(active_version))] for match_id in paired_matches]
        candidate_rows = [by_match_version[(match_id, str(candidate_version))] for match_id in paired_matches]
        return self._summarize_paired(active_rows, active_version), self._summarize_paired(candidate_rows, candidate_version)

    def get_performance_summary(
        self,
        sport: str | None = None,
    ) -> dict[str, Any] | None:
        session = get_session()
        try:
            query = session.query(PostMatchReview, PredictionRun).join(
                PredictionRun,
                PredictionRun.id == PostMatchReview.prediction_run_id,
            )
            if sport:
                query = query.filter(PostMatchReview.sport == sport)
            rows = query.all()
        finally:
            session.close()

        if not rows:
            return None

        models: dict[str, dict[str, Any]] = {}
        markets: dict[str, dict[str, Any]] = {}
        sports: dict[str, dict[str, Any]] = {}
        total_correct = 0
        total_evaluated = 0
        weighted_brier = 0.0
        weighted_absolute_error = 0.0
        total_calibrated = 0

        for review, run in rows:
            context = run.context_json or {}
            model_version = str(context.get("model_version") or "legacy")
            model_key = f"{run.sport}\0{run.model_name}\0{model_version}"
            model = models.setdefault(
                model_key,
                {
                    "sport": run.sport,
                    "model_name": run.model_name,
                    "model_version": model_version,
                    "reviews": 0,
                    "correct": 0,
                    "evaluated": 0,
                    "calibrated": 0,
                    "weighted_brier": 0.0,
                    "weighted_absolute_error": 0.0,
                },
            )
            sport_group = sports.setdefault(
                run.sport,
                {
                    "sport": run.sport,
                    "reviews": 0,
                    "correct": 0,
                    "evaluated": 0,
                    "calibrated": 0,
                    "weighted_brier": 0.0,
                    "weighted_absolute_error": 0.0,
                },
            )
            model["reviews"] += 1
            model["correct"] += review.correct_markets
            model["evaluated"] += review.evaluated_markets
            calibrated = len(review.details_json or [])
            model["calibrated"] += calibrated
            if review.mean_brier_score is not None:
                model["weighted_brier"] += (
                    review.mean_brier_score * calibrated
                )
            if review.mean_absolute_error is not None:
                model["weighted_absolute_error"] += (
                    review.mean_absolute_error * calibrated
                )

            sport_group["reviews"] += 1
            sport_group["correct"] += review.correct_markets
            sport_group["evaluated"] += review.evaluated_markets
            sport_group["calibrated"] += calibrated
            if review.mean_brier_score is not None:
                sport_group["weighted_brier"] += review.mean_brier_score * calibrated
            if review.mean_absolute_error is not None:
                absolute_error = review.mean_absolute_error * calibrated
                sport_group["weighted_absolute_error"] += absolute_error
                weighted_absolute_error += absolute_error

            total_correct += review.correct_markets
            total_evaluated += review.evaluated_markets
            total_calibrated += calibrated
            if review.mean_brier_score is not None:
                weighted_brier += review.mean_brier_score * calibrated

            for detail in review.details_json or []:
                market_key = f'{run.sport}\0{detail["market_type"]}'
                market = markets.setdefault(
                    market_key,
                    {
                        "sport": run.sport,
                        "market_type": detail["market_type"],
                        "correct": 0,
                        "evaluated": 0,
                        "brier_total": 0.0,
                    },
                )
                market["evaluated"] += 1
                market["correct"] += int(bool(detail.get("correct")))
                market["brier_total"] += float(detail.get("brier_score", 0.0))

        model_rows = [self._finalize_group(item) for item in models.values()]
        sport_rows = [self._finalize_group(item) for item in sports.values()]
        market_rows = [self._finalize_market(item) for item in markets.values()]
        model_rows.sort(key=lambda item: (-item["accuracy"], item["sport"], item["model_name"]))
        sport_rows.sort(key=lambda item: (-item["accuracy"], item["sport"]))
        market_rows.sort(key=lambda item: (-item["accuracy"], item["sport"], item["market_type"]))

        return {
            "reviews": len(rows),
            "evaluated_markets": total_evaluated,
            "accuracy": total_correct / total_evaluated if total_evaluated else 0.0,
            "mean_brier_score": (
                weighted_brier / total_calibrated if total_calibrated else 0.0
            ),
            "mean_absolute_error": (
                weighted_absolute_error / total_calibrated if total_calibrated else 0.0
            ),
            "sports": sport_rows,
            "models": model_rows,
            "markets": market_rows,
        }

    @staticmethod
    def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
        evaluated = group["evaluated"]
        calibrated = group["calibrated"]
        return {
            "sport": group.get("sport"),
            "model_name": group.get("model_name"),
            "model_version": group.get("model_version"),
            "reviews": group["reviews"],
            "evaluated": evaluated,
            "accuracy": group["correct"] / evaluated if evaluated else 0.0,
            "mean_brier_score": (
                group["weighted_brier"] / calibrated if calibrated else 0.0
            ),
            "mean_absolute_error": (
                group.get("weighted_absolute_error", 0.0) / calibrated
                if calibrated else 0.0
            ),
        }

    @staticmethod
    def _summarize_paired(rows: list[tuple[Any, Any]], version: str) -> dict[str, Any]:
        evaluated = sum(int(review.evaluated_markets or 0) for review, _ in rows)
        correct = sum(int(review.correct_markets or 0) for review, _ in rows)
        calibrated = sum(len(review.details_json or []) for review, _ in rows)
        weighted_brier = sum(
            float(review.mean_brier_score or 0.0) * len(review.details_json or [])
            for review, _ in rows
        )
        return {
            "model_version": str(version),
            "paired_matches": len(rows),
            "evaluated": evaluated,
            "accuracy": correct / evaluated if evaluated else 0.0,
            "mean_brier_score": weighted_brier / calibrated if calibrated else 0.0,
        }

    @staticmethod
    def _finalize_market(group: dict[str, Any]) -> dict[str, Any]:
        evaluated = group["evaluated"]
        return {
            "sport": group["sport"],
            "market_type": group["market_type"],
            "evaluated": evaluated,
            "accuracy": group["correct"] / evaluated if evaluated else 0.0,
            "mean_brier_score": (
                group["brier_total"] / evaluated if evaluated else 0.0
            ),
        }
