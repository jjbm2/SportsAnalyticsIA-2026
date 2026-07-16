from typing import Any

from database.database import get_session
from database.models import PredictionMarket, PredictionRun


class PredictionRepository:
    def shadow_run_exists(self, sport: str, match_id: str, model_version: str) -> bool:
        session = get_session()
        try:
            runs = session.query(PredictionRun).filter(
                PredictionRun.sport == sport,
                PredictionRun.match_id == str(match_id),
                PredictionRun.status == "shadow",
            ).all()
            return any(
                str((run.context_json or {}).get("model_version")) == str(model_version)
                for run in runs
            )
        finally:
            session.close()

    def save_prediction_run(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        model_name: str,
        simulations: int,
        markets: list[dict[str, Any]],
        match_id: str | None = None,
        status: str = "completed",
        context_json: dict[str, Any] | None = None,
    ) -> int:
        session = get_session()

        try:
            run = PredictionRun(
                sport=sport,
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                model_name=model_name,
                simulations=simulations,
                status=status,
                context_json=context_json,
            )

            session.add(run)
            session.flush()

            for market in markets:
                market_row = PredictionMarket(
                    run_id=run.id,
                    market_type=market["market_type"],
                    selection=market["selection"],
                    probability=market["probability"],
                    confidence=market.get("confidence"),
                    risk=market.get("risk"),
                    extra_data_json=market.get("extra_data_json"),
                )
                session.add(market_row)

            session.commit()
            session.refresh(run)

            return run.id

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()

    def list_recent_runs(
        self,
        limit: int = 20,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        session = get_session()

        try:
            query = session.query(PredictionRun).filter(PredictionRun.status != "shadow")
            if user_id is not None:
                query = query.filter(
                    PredictionRun.context_json["user_id"].as_integer() == int(user_id)
                )
            runs = query.order_by(PredictionRun.created_at.desc()).limit(limit).all()

            return [
                {
                    "id": run.id,
                    "sport": run.sport,
                    "match_id": run.match_id,
                    "home_team": run.home_team,
                    "away_team": run.away_team,
                    "model_name": run.model_name,
                    "simulations": run.simulations,
                    "status": run.status,
                    "created_at": run.created_at,
                    "context_json": run.context_json,
                }
                for run in runs
            ]

        finally:
            session.close()

    def list_markets_by_run(
        self,
        run_id: int,
    ) -> list[dict[str, Any]]:
        session = get_session()

        try:
            markets = (
                session.query(PredictionMarket)
                .filter(PredictionMarket.run_id == run_id)
                .order_by(PredictionMarket.id.asc())
                .all()
            )

            return [
                {
                    "id": market.id,
                    "run_id": market.run_id,
                    "market_type": market.market_type,
                    "selection": market.selection,
                    "probability": market.probability,
                    "confidence": market.confidence,
                    "risk": market.risk,
                    "extra_data_json": market.extra_data_json,
                }
                for market in markets
            ]

        finally:
            session.close()

    def list_runs_by_match_id(
        self,
        match_id: str,
        sport: str | None = None,
    ) -> list[dict[str, Any]]:
        session = get_session()

        try:
            query = session.query(PredictionRun).filter(
                PredictionRun.match_id == str(match_id)
            )
            if sport:
                query = query.filter(PredictionRun.sport == sport)
            runs = query.order_by(PredictionRun.created_at.asc()).all()
            return [
                {
                    "id": run.id,
                    "sport": run.sport,
                    "match_id": run.match_id,
                    "home_team": run.home_team,
                    "away_team": run.away_team,
                    "model_name": run.model_name,
                    "simulations": run.simulations,
                    "status": run.status,
                    "created_at": run.created_at,
                    "context_json": run.context_json,
                }
                for run in runs
            ]
        finally:
            session.close()
