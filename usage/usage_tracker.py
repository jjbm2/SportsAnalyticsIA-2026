from __future__ import annotations

from datetime import date
from typing import Any, Callable

from database.database import get_session
from database.models import PromotionRedemption, Subscription, User, UserUsage
from promotions.promotion_service import OPENING_DAILY_LIMIT
from core.plans import get_plan
from core.time_utils import utc_now


class UsageTracker:
    def __init__(self, session_factory: Callable = get_session):
        self.session_factory = session_factory

    def can_user_predict(self, user_id: int, sport: str, on_date: date | None = None) -> dict[str, Any]:
        today = on_date or date.today()
        session = self.session_factory()
        try:
            user = session.get(User, int(user_id))
            if user is None:
                return {"allowed": False, "reason": "Usuario no encontrado"}
            plan_code = self._effective_plan(session, user)
            plan = get_plan(plan_code)
            usage = self._usage_row(session, user.id, sport, today)
            used = usage.predictions_count if usage else 0
            extra = usage.extra_predictions if usage else 0
            limit = plan.daily_predictions_per_sport
            promotion_active = self._promotion_active(session, user, today)
            if promotion_active and limit is not None:
                limit = max(limit, OPENING_DAILY_LIMIT)
            allowed = limit is None or used < limit + extra
            return {
                "allowed": allowed,
                "plan": plan_code,
                "used": used,
                "limit": limit,
                "extra": extra,
                "remaining": None if limit is None else max(0, limit + extra - used),
                "promotion_active": promotion_active,
                "reason": None if allowed else "Has alcanzado tu límite diario para este deporte",
            }
        finally:
            session.close()

    def record_prediction(self, user_id: int, sport: str, on_date: date | None = None) -> int:
        today = on_date or date.today()
        session = self.session_factory()
        try:
            user = session.get(User, int(user_id))
            if user is None:
                raise ValueError("Usuario no encontrado")
            plan = get_plan(self._effective_plan(session, user))
            usage = self._usage_row(session, user.id, sport, today)
            if usage is None:
                usage = UserUsage(user_id=user.id, sport=sport, date=today)
                session.add(usage)
                session.flush()
            limit = plan.daily_predictions_per_sport
            if self._promotion_active(session, user, today) and limit is not None:
                limit = max(limit, OPENING_DAILY_LIMIT)
            if limit is not None and usage.predictions_count >= limit + usage.extra_predictions:
                raise PermissionError("Has alcanzado tu límite diario para este deporte")
            usage.predictions_count += 1
            session.commit()
            return usage.predictions_count
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def daily_usage(self, user_id: int, on_date: date | None = None) -> list[dict[str, Any]]:
        today = on_date or date.today()
        session = self.session_factory()
        try:
            rows = session.query(UserUsage).filter(
                UserUsage.user_id == user_id, UserUsage.date == today
            ).order_by(UserUsage.sport).all()
            return [{"sport": row.sport, "count": row.predictions_count, "extra": row.extra_predictions} for row in rows]
        finally:
            session.close()

    def release_prediction(self, user_id: int, sport: str, on_date: date | None = None) -> int:
        """Return a reserved prediction when no complete run was persisted."""
        today = on_date or date.today()
        session = self.session_factory()
        try:
            usage = self._usage_row(session, int(user_id), sport, today)
            if usage is None or usage.predictions_count <= 0:
                return 0
            usage.predictions_count -= 1
            remaining = usage.predictions_count
            session.commit()
            return remaining
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _usage_row(session: Any, user_id: int, sport: str, on_date: date) -> UserUsage | None:
        return session.query(UserUsage).filter(
            UserUsage.user_id == user_id,
            UserUsage.sport == sport,
            UserUsage.date == on_date,
        ).one_or_none()

    @staticmethod
    def _effective_plan(session: Any, user: User) -> str:
        if user.is_admin:
            return "full"
        if user.plan == "free":
            return "free"
        active = session.query(Subscription).filter(
            Subscription.user_id == user.id,
            Subscription.active.is_(True),
            Subscription.end_date > utc_now(),
        ).order_by(Subscription.end_date.desc()).first()
        return active.plan if active else "free"

    @staticmethod
    def _promotion_active(session: Any, user: User, on_date: date) -> bool:
        if user.is_admin:
            return False
        return session.query(PromotionRedemption.id).filter(
            PromotionRedemption.user_id == user.id,
            PromotionRedemption.activated_on <= on_date,
            PromotionRedemption.ends_on >= on_date,
        ).first() is not None
