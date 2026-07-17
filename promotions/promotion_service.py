from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Callable

from database.database import get_session
from database.models import PromotionRedemption, User


DEFAULT_OPENING_CODE = "APERTURA5"
OPENING_PROMOTION_DAYS = 5
OPENING_DAILY_LIMIT = 5


def configured_opening_code() -> str:
    return (os.getenv("OPENING_PROMO_CODE") or DEFAULT_OPENING_CODE).strip().upper()


class PromotionService:
    def __init__(self, session_factory: Callable = get_session):
        self.session_factory = session_factory

    def redeem(self, user_id: int, code: str, on_date: date | None = None) -> dict[str, Any]:
        today = on_date or date.today()
        normalized_code = str(code or "").strip().upper()
        if not normalized_code or normalized_code != configured_opening_code():
            raise ValueError("El código promocional no es válido")

        session = self.session_factory()
        try:
            user = session.get(User, int(user_id))
            if user is None:
                raise ValueError("Usuario no encontrado")
            if user.is_admin or user.plan != "free":
                raise ValueError("La promoción está disponible para cuentas Free")
            existing = session.query(PromotionRedemption).filter(
                PromotionRedemption.user_id == user.id
            ).one_or_none()
            if existing is not None:
                raise ValueError("Esta cuenta ya utilizó la promoción de apertura")

            redemption = PromotionRedemption(
                user_id=user.id,
                code=normalized_code,
                activated_on=today,
                ends_on=today + timedelta(days=OPENING_PROMOTION_DAYS - 1),
            )
            session.add(redemption)
            session.commit()
            session.refresh(redemption)
            return self._status(redemption, today)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def status(self, user_id: int, on_date: date | None = None) -> dict[str, Any]:
        today = on_date or date.today()
        session = self.session_factory()
        try:
            redemption = session.query(PromotionRedemption).filter(
                PromotionRedemption.user_id == int(user_id)
            ).one_or_none()
            if redemption is None:
                return {"redeemed": False, "active": False, "expired": False}
            return self._status(redemption, today)
        finally:
            session.close()

    @staticmethod
    def _status(redemption: PromotionRedemption, today: date) -> dict[str, Any]:
        active = redemption.activated_on <= today <= redemption.ends_on
        return {
            "redeemed": True,
            "active": active,
            "expired": today > redemption.ends_on,
            "code": redemption.code,
            "activated_on": redemption.activated_on,
            "ends_on": redemption.ends_on,
            "days_remaining": max(0, (redemption.ends_on - today).days + 1) if active else 0,
            "daily_limit": OPENING_DAILY_LIMIT,
        }
