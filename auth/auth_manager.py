from __future__ import annotations

import re
from typing import Any, Callable

from sqlalchemy.exc import IntegrityError

from auth.password_utils import hash_password, verify_password
from core.time_utils import utc_now
from database.database import get_session
from database.models import Subscription, User


EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class AuthManager:
    def __init__(self, session_factory: Callable = get_session):
        self.session_factory = session_factory

    def register(self, email: str, password: str, *, is_admin: bool = False) -> dict[str, Any]:
        normalized = self.normalize_email(email)
        user = User(
            email=normalized,
            password_hash=hash_password(password),
            plan="free",
            is_admin=is_admin,
        )
        session = self.session_factory()
        try:
            session.add(user)
            session.commit()
            session.refresh(user)
            return self._public_user(user)
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("Ya existe una cuenta con ese correo") from exc
        finally:
            session.close()

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        normalized = self.normalize_email(email)
        session = self.session_factory()
        try:
            user = session.query(User).filter(User.email == normalized).one_or_none()
            if user is None or not verify_password(password, user.password_hash):
                return None
            return self._public_user(user)
        finally:
            session.close()

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        session = self.session_factory()
        try:
            user = session.get(User, int(user_id))
            if user and not user.is_admin and user.plan != "free":
                active = session.query(Subscription).filter(
                    Subscription.user_id == user.id,
                    Subscription.active.is_(True),
                    Subscription.end_date > utc_now(),
                ).first()
                if active is None:
                    user.plan = "free"
                    session.commit()
            return self._public_user(user) if user else None
        finally:
            session.close()

    def ensure_admin_from_environment(self, email: str | None, password: str | None) -> None:
        if not email or not password:
            return
        normalized = self.normalize_email(email)
        session = self.session_factory()
        try:
            existing = session.query(User).filter(User.email == normalized).one_or_none()
            if existing:
                if not existing.is_admin:
                    existing.is_admin = True
                    session.commit()
                return
        finally:
            session.close()
        self.register(normalized, password, is_admin=True)

    @staticmethod
    def normalize_email(email: str) -> str:
        normalized = str(email).strip().lower()
        if len(normalized) > 254 or not EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("Correo electrónico no válido")
        return normalized

    @staticmethod
    def _public_user(user: User) -> dict[str, Any]:
        return {
            "id": user.id,
            "email": user.email,
            "plan": user.plan,
            "is_admin": bool(user.is_admin),
            "created_at": user.created_at,
        }
