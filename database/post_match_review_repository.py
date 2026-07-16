from __future__ import annotations

from typing import Any

from database.database import get_session
from database.models import PostMatchReview, PredictionRun


class PostMatchReviewRepository:
    def get_review(self, prediction_run_id: int) -> dict[str, Any] | None:
        session = get_session()
        try:
            review = session.query(PostMatchReview).filter(
                PostMatchReview.prediction_run_id == prediction_run_id
            ).one_or_none()
            if review is None:
                return None
            return {
                "id": review.id,
                "actual_outcome": review.actual_outcome,
                "evaluated_markets": review.evaluated_markets,
                "correct_markets": review.correct_markets,
                "accuracy": review.accuracy,
                "mean_absolute_error": review.mean_absolute_error,
                "mean_brier_score": review.mean_brier_score,
                "market_evaluations": review.details_json or [],
            }
        finally:
            session.close()

    def review_exists(self, prediction_run_id: int) -> bool:
        session = get_session()
        try:
            return session.query(PostMatchReview.id).filter(
                PostMatchReview.prediction_run_id == prediction_run_id
            ).first() is not None
        finally:
            session.close()

    def save_review(
        self,
        prediction_run_id: int,
        match_id: str,
        sport: str,
        home_score: float,
        away_score: float,
        evaluation: dict[str, Any],
    ) -> int:
        session = get_session()
        try:
            review = (
                session.query(PostMatchReview)
                .filter(PostMatchReview.prediction_run_id == prediction_run_id)
                .one_or_none()
            )
            if review is None:
                review = PostMatchReview(prediction_run_id=prediction_run_id)
                session.add(review)

            review.match_id = str(match_id)
            review.sport = sport
            review.home_score = home_score
            review.away_score = away_score
            review.actual_outcome = evaluation["actual_outcome"]
            review.evaluated_markets = evaluation["evaluated_markets"]
            review.correct_markets = evaluation["correct_markets"]
            review.accuracy = evaluation["accuracy"]
            review.mean_absolute_error = evaluation["mean_absolute_error"]
            review.mean_brier_score = evaluation["mean_brier_score"]
            review.details_json = evaluation["market_evaluations"]

            session.commit()
            session.refresh(review)
            return review.id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_recent_reviews(
        self,
        limit: int = 20,
        sport: str | None = None,
    ) -> list[dict[str, Any]]:
        session = get_session()
        try:
            query = session.query(PostMatchReview, PredictionRun).join(
                PredictionRun,
                PredictionRun.id == PostMatchReview.prediction_run_id,
            )
            if sport:
                query = query.filter(PostMatchReview.sport == sport)
            reviews = query.order_by(
                PostMatchReview.evaluated_at.desc()
            ).limit(limit).all()
            return [
                {
                    "id": review.id,
                    "prediction_run_id": review.prediction_run_id,
                    "match_id": review.match_id,
                    "sport": review.sport,
                    "home_score": review.home_score,
                    "away_score": review.away_score,
                    "actual_outcome": review.actual_outcome,
                    "evaluated_markets": review.evaluated_markets,
                    "correct_markets": review.correct_markets,
                    "accuracy": review.accuracy,
                    "mean_absolute_error": review.mean_absolute_error,
                    "mean_brier_score": review.mean_brier_score,
                    "details_json": review.details_json,
                    "evaluated_at": review.evaluated_at,
                    "home_team": run.home_team,
                    "away_team": run.away_team,
                    "model_name": run.model_name,
                    "prediction_created_at": run.created_at,
                }
                for review, run in reviews
            ]
        finally:
            session.close()
