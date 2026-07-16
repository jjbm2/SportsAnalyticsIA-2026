from __future__ import annotations

from calendar import monthrange
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from database.database import get_session
from database.models import PaymentRequest, Subscription, User
from core.paths import DATA_DIR
from core.plans import get_plan, plan_amount


RECEIPT_DIR = DATA_DIR / "payment_receipts"
ALLOWED_RECEIPT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}


class BillingManager:
    def __init__(self, session_factory: Callable = get_session, receipt_dir: Path = RECEIPT_DIR):
        self.session_factory = session_factory
        self.receipt_dir = receipt_dir

    def create_payment_request(
        self,
        user_id: int,
        plan: str,
        billing_cycle: str,
        receipt: bytes | None = None,
        receipt_suffix: str | None = None,
    ) -> dict[str, Any]:
        selected = get_plan(plan)
        if selected.code == "free":
            raise ValueError("El plan Free no requiere solicitud de pago")
        amount = plan_amount(plan, billing_cycle)
        proof_path = self._save_proof(receipt, receipt_suffix) if receipt else None
        session = self.session_factory()
        try:
            if session.get(User, int(user_id)) is None:
                raise ValueError("Usuario no encontrado")
            request = PaymentRequest(
                user_id=user_id,
                plan=selected.code,
                billing_cycle=billing_cycle,
                amount=amount,
                status="pending",
                proof_path=proof_path,
                proof_uploaded_at=datetime.utcnow() if proof_path else None,
            )
            session.add(request)
            session.commit()
            session.refresh(request)
            return self._payment_dict(request)
        except Exception:
            session.rollback()
            if proof_path:
                Path(proof_path).unlink(missing_ok=True)
            raise
        finally:
            session.close()

    def approve(self, request_id: int, admin_user_id: int, now: datetime | None = None) -> dict[str, Any]:
        current = now or datetime.utcnow()
        session = self.session_factory()
        try:
            admin = session.get(User, int(admin_user_id))
            request = session.get(PaymentRequest, int(request_id))
            if admin is None or not admin.is_admin:
                raise PermissionError("Acceso administrativo requerido")
            if request is None or request.status != "pending":
                raise ValueError("La solicitud no está pendiente")
            user = session.get(User, request.user_id)
            session.query(Subscription).filter(
                Subscription.user_id == user.id,
                Subscription.active.is_(True),
            ).update({Subscription.active: False}, synchronize_session=False)
            months = 12 if request.billing_cycle == "yearly" else 1
            subscription = Subscription(
                user_id=user.id,
                plan=request.plan,
                billing_cycle=request.billing_cycle,
                start_date=current,
                end_date=_add_months(current, months),
                active=True,
                payment_request_id=request.id,
            )
            user.plan = request.plan
            request.status = "approved"
            request.reviewed_at = current
            request.reviewed_by = admin.id
            self._delete_proof(request)
            session.add(subscription)
            session.commit()
            session.refresh(subscription)
            return self._subscription_dict(subscription)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def reject(self, request_id: int, admin_user_id: int) -> None:
        session = self.session_factory()
        try:
            admin = session.get(User, int(admin_user_id))
            request = session.get(PaymentRequest, int(request_id))
            if admin is None or not admin.is_admin:
                raise PermissionError("Acceso administrativo requerido")
            if request is None or request.status != "pending":
                raise ValueError("La solicitud no está pendiente")
            request.status = "rejected"
            request.reviewed_at = datetime.utcnow()
            request.reviewed_by = admin.id
            self._delete_proof(request)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_requests(self, status: str | None = None, user_id: int | None = None) -> list[dict[str, Any]]:
        session = self.session_factory()
        try:
            query = session.query(PaymentRequest)
            if status:
                query = query.filter(PaymentRequest.status == status)
            if user_id:
                query = query.filter(PaymentRequest.user_id == user_id)
            return [self._payment_dict(item) for item in query.order_by(PaymentRequest.created_at.desc()).all()]
        finally:
            session.close()

    def get_receipt(self, request_id: int, admin_user_id: int) -> dict[str, Any] | None:
        session = self.session_factory()
        try:
            admin = session.get(User, int(admin_user_id))
            request = session.get(PaymentRequest, int(request_id))
            if admin is None or not admin.is_admin:
                raise PermissionError("Acceso administrativo requerido")
            proof_path = self._proof_path(request) if request else None
            if request is None or not proof_path:
                return None
            path = Path(proof_path)
            if not path.is_file() or path.suffix.lower() not in ALLOWED_RECEIPT_SUFFIXES:
                return None
            mime = "application/pdf" if path.suffix.lower() == ".pdf" else "image/jpeg"
            if path.suffix.lower() == ".png":
                mime = "image/png"
            return {"name": f"comprobante_{request.id}{path.suffix.lower()}", "mime": mime, "content": path.read_bytes()}
        finally:
            session.close()

    def active_subscription(self, user_id: int, now: datetime | None = None) -> dict[str, Any] | None:
        current = now or datetime.utcnow()
        session = self.session_factory()
        try:
            item = session.query(Subscription).filter(
                Subscription.user_id == user_id,
                Subscription.active.is_(True),
                Subscription.end_date > current,
            ).order_by(Subscription.end_date.desc()).first()
            return self._subscription_dict(item) if item else None
        finally:
            session.close()

    def _save_proof(self, content: bytes, suffix: str | None) -> str:
        extension = str(suffix or "").lower()
        if extension not in ALLOWED_RECEIPT_SUFFIXES:
            raise ValueError("El comprobante debe ser PDF, PNG o JPG")
        if not content or not self._valid_signature(content, extension):
            raise ValueError("El contenido del comprobante no coincide con el tipo de archivo")
        if len(content) > 5 * 1024 * 1024:
            raise ValueError("El comprobante no puede superar 5 MB")
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        path = self.receipt_dir / f"{uuid4().hex}{extension}"
        path.write_bytes(content)
        return str(path)

    @staticmethod
    def _valid_signature(content: bytes, extension: str) -> bool:
        if extension == ".pdf":
            return content.startswith(b"%PDF-")
        if extension == ".png":
            return content.startswith(b"\x89PNG\r\n\x1a\n")
        return content.startswith(b"\xff\xd8\xff")

    @staticmethod
    def _proof_path(item: PaymentRequest) -> str | None:
        return item.proof_path or item.receipt_path

    @classmethod
    def _delete_proof(cls, item: PaymentRequest) -> None:
        proof_path = cls._proof_path(item)
        if proof_path:
            Path(proof_path).unlink(missing_ok=True)
        item.proof_path = None
        item.proof_uploaded_at = None
        item.receipt_path = None

    @staticmethod
    def _payment_dict(item: PaymentRequest) -> dict[str, Any]:
        values = {key: getattr(item, key) for key in (
            "id", "user_id", "plan", "billing_cycle", "amount", "status", "created_at"
        )}
        values["has_receipt"] = bool(BillingManager._proof_path(item))
        values["proof_uploaded_at"] = item.proof_uploaded_at
        return values

    @staticmethod
    def _subscription_dict(item: Subscription) -> dict[str, Any]:
        return {key: getattr(item, key) for key in (
            "id", "user_id", "plan", "billing_cycle", "start_date", "end_date", "active"
        )}


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)
