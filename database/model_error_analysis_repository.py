from __future__ import annotations

from typing import Any

from database.database import get_session
from database.models import ModelErrorAnalysis, PostMatchReview, PredictionRun
from machine_learning.evaluation.model_error_analyzer import build_error_rows, detect_failure_patterns


class ModelErrorAnalysisRepository:
    def backfill_existing_reviews(self) -> int:
        session = get_session()
        try:
            records = session.query(PostMatchReview, PredictionRun).join(
                PredictionRun, PredictionRun.id == PostMatchReview.prediction_run_id
            ).order_by(PostMatchReview.evaluated_at.asc()).all()
        finally:
            session.close()
        inserted = 0
        for review, run in records:
            run_data = {
                "id": run.id, "match_id": run.match_id, "sport": run.sport,
                "context_json": run.context_json or {},
            }
            evaluation = {
                "actual_outcome": review.actual_outcome,
                "market_evaluations": review.details_json or [],
            }
            inserted += self.save_rows(build_error_rows(run_data, review.id, evaluation))
        return inserted

    def save_rows(self, rows: list[dict[str, Any]]) -> int:
        session = get_session()
        inserted = 0
        try:
            for values in rows:
                existing = session.query(ModelErrorAnalysis).filter(
                    ModelErrorAnalysis.prediction_run_id == values["prediction_run_id"],
                    ModelErrorAnalysis.market_type == values["market_type"],
                    ModelErrorAnalysis.selection == values["selection"],
                ).first()
                if existing:
                    for key, value in values.items():
                        setattr(existing, key, value)
                    continue
                session.add(ModelErrorAnalysis(**values))
                inserted += 1
            session.commit()
            return inserted
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_rows(self, sport: str | None = None, errors_only: bool = False) -> list[dict[str, Any]]:
        session = get_session()
        try:
            query = session.query(ModelErrorAnalysis)
            if sport:
                query = query.filter(ModelErrorAnalysis.sport == sport)
            if errors_only:
                query = query.filter(ModelErrorAnalysis.correct.is_(False))
            records = query.order_by(ModelErrorAnalysis.created_at.asc()).all()
            return [{column.name: getattr(record, column.name) for column in ModelErrorAnalysis.__table__.columns} for record in records]
        finally:
            session.close()

    def failure_patterns(self, sport: str | None = None, minimum_samples: int = 3) -> list[dict[str, Any]]:
        return detect_failure_patterns(self.list_rows(sport=sport), minimum_samples=minimum_samples)
