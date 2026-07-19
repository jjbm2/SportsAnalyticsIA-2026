from __future__ import annotations

from datetime import date
from typing import Any, Callable

from database.database import get_session
from database.models import PaymentRequest, Subscription, User, UserUsage
from core.plans import get_plan
from core.time_utils import utc_now


class AdminService:
    def __init__(self, session_factory: Callable = get_session):
        self.session_factory = session_factory

    def dashboard(self, admin_user_id: int) -> dict[str, Any]:
        session = self.session_factory()
        try:
            self._require_admin(session, admin_user_id)
            return {
                "total_users": session.query(User).count(),
                "approved_revenue_mxn": sum(
                    item.amount for item in session.query(PaymentRequest).filter(
                        PaymentRequest.status == "approved"
                    ).all()
                ),
                "predictions": sum(item.predictions_count for item in session.query(UserUsage).all()),
                "pending_payments": session.query(PaymentRequest).filter(
                    PaymentRequest.status == "pending"
                ).count(),
            }
        finally:
            session.close()

    def list_users(self, admin_user_id: int) -> list[dict[str, Any]]:
        session = self.session_factory()
        try:
            self._require_admin(session, admin_user_id)
            return [
                {
                    "id": user.id,
                    "email": user.email,
                    "plan": user.plan,
                    "is_admin": user.is_admin,
                    "is_banned": bool(user.is_banned),
                    "banned_at": user.banned_at,
                    "ban_reason": user.ban_reason,
                    "created_at": user.created_at,
                }
                for user in session.query(User).order_by(User.created_at.desc()).all()
            ]
        finally:
            session.close()

    def set_user_banned(
        self,
        admin_user_id: int,
        user_id: int,
        banned: bool,
        reason: str | None = None,
    ) -> None:
        session = self.session_factory()
        try:
            self._require_admin(session, admin_user_id)
            user = session.get(User, int(user_id))
            if user is None:
                raise ValueError("Usuario no encontrado")
            if user.id == int(admin_user_id) or user.is_admin:
                raise ValueError("No se puede suspender una cuenta administrativa")
            user.is_banned = bool(banned)
            user.banned_at = utc_now() if banned else None
            user.ban_reason = (reason or "").strip()[:255] or None if banned else None
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def change_plan(self, admin_user_id: int, user_id: int, plan: str) -> None:
        from datetime import timedelta
        get_plan(plan)
        session = self.session_factory()
        try:
            self._require_admin(session, admin_user_id)
            user = session.get(User, user_id)
            if user is None:
                raise ValueError("Usuario no encontrado")
            user.plan = plan
            session.query(Subscription).filter(
                Subscription.user_id == user.id, Subscription.active.is_(True)
            ).update({Subscription.active: False}, synchronize_session=False)
            if plan != "free":
                now = utc_now()
                session.add(Subscription(
                    user_id=user.id,
                    plan=plan,
                    billing_cycle="monthly",
                    start_date=now,
                    end_date=now + timedelta(days=30),
                    active=True,
                    payment_request_id=None,
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def grant_extra_predictions(
        self, admin_user_id: int, user_id: int, sport: str, amount: int, on_date: date | None = None
    ) -> None:
        if amount <= 0:
            raise ValueError("La cantidad debe ser positiva")
        today = on_date or date.today()
        session = self.session_factory()
        try:
            self._require_admin(session, admin_user_id)
            usage = session.query(UserUsage).filter(
                UserUsage.user_id == user_id, UserUsage.sport == sport, UserUsage.date == today
            ).one_or_none()
            if usage is None:
                usage = UserUsage(user_id=user_id, sport=sport, date=today)
                session.add(usage)
            usage.extra_predictions += amount
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def extend_subscription(self, admin_user_id: int, user_id: int, days: int) -> None:
        from datetime import timedelta
        if days <= 0:
            raise ValueError("Los días deben ser positivos")
        session = self.session_factory()
        try:
            self._require_admin(session, admin_user_id)
            subscription = session.query(Subscription).filter(
                Subscription.user_id == user_id, Subscription.active.is_(True)
            ).order_by(Subscription.end_date.desc()).first()
            if subscription is None:
                raise ValueError("El usuario no tiene una suscripción activa")
            subscription.end_date += timedelta(days=days)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _require_admin(session: Any, admin_user_id: int) -> User:
        admin = session.get(User, int(admin_user_id))
        if admin is None or not admin.is_admin:
            raise PermissionError("Acceso administrativo requerido")
        return admin
