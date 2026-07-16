from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from admin.admin_service import AdminService
from auth.auth_manager import AuthManager
from billing.billing_manager import BillingManager
from database.database import Base
from database.models import PaymentRequest, Subscription, User
from database.prediction_repository import PredictionRepository
from usage.usage_tracker import UsageTracker


class SaaSFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self.auth = AuthManager(self.sessions)
        self.usage = UsageTracker(self.sessions)
        self.billing = BillingManager(self.sessions, Path("data/test_payment_receipts"))
        self.admin = AdminService(self.sessions)
        self.admin_user = self.auth.register("admin@example.com", "AdminPass123", is_admin=True)
        self.user = self.auth.register("user@example.com", "UserPass123")

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_registration_hashes_password_and_login_is_safe(self) -> None:
        session = self.sessions()
        stored = session.get(User, self.user["id"])
        self.assertNotEqual(stored.password_hash, "UserPass123")
        self.assertTrue(stored.password_hash.startswith("$2"))
        session.close()
        self.assertEqual(self.auth.authenticate(" USER@example.com ", "UserPass123")["id"], self.user["id"])
        self.assertIsNone(self.auth.authenticate("user@example.com", "wrong-password"))
        with self.assertRaises(ValueError):
            self.auth.register("USER@example.com", "AnotherPass123")

    def test_free_limit_is_per_sport_and_resets_by_date(self) -> None:
        day_one = date(2026, 7, 16)
        day_two = day_one + timedelta(days=1)
        self.assertTrue(self.usage.can_user_predict(self.user["id"], "Fútbol", day_one)["allowed"])
        self.usage.record_prediction(self.user["id"], "Fútbol", day_one)
        self.assertFalse(self.usage.can_user_predict(self.user["id"], "Fútbol", day_one)["allowed"])
        self.assertTrue(self.usage.can_user_predict(self.user["id"], "Basketball", day_one)["allowed"])
        self.assertTrue(self.usage.can_user_predict(self.user["id"], "Fútbol", day_two)["allowed"])

    def test_manual_payment_approval_activates_plan_and_subscription(self) -> None:
        request = self.billing.create_payment_request(self.user["id"], "pro", "yearly")
        session = self.sessions()
        self.assertEqual(session.get(User, self.user["id"]).plan, "free")
        self.assertEqual(session.get(PaymentRequest, request["id"]).status, "pending")
        session.close()

        subscription = self.billing.approve(
            request["id"], self.admin_user["id"], now=datetime(2026, 7, 16, 12, 0)
        )
        self.assertEqual(subscription["plan"], "pro")
        self.assertEqual(subscription["end_date"], datetime(2027, 7, 16, 12, 0))
        session = self.sessions()
        self.assertEqual(session.get(User, self.user["id"]).plan, "pro")
        self.assertEqual(session.get(PaymentRequest, request["id"]).status, "approved")
        self.assertEqual(session.query(Subscription).filter_by(user_id=self.user["id"], active=True).count(), 1)
        session.close()

    def test_non_admin_cannot_approve_payment(self) -> None:
        request = self.billing.create_payment_request(self.user["id"], "basic", "monthly")
        with self.assertRaises(PermissionError):
            self.billing.approve(request["id"], self.user["id"])

    def test_proof_is_visible_to_admin_and_deleted_after_approval(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"safe-image-content"
        request = self.billing.create_payment_request(
            self.user["id"], "basic", "monthly", png, ".png"
        )
        session = self.sessions()
        stored = session.get(PaymentRequest, request["id"])
        proof_path = Path(stored.proof_path)
        self.assertTrue(proof_path.is_file())
        self.assertIsNotNone(stored.proof_uploaded_at)
        session.close()
        proof = self.billing.get_receipt(request["id"], self.admin_user["id"])
        self.assertEqual(proof["content"], png)

        self.billing.approve(request["id"], self.admin_user["id"])
        self.assertFalse(proof_path.exists())
        session = self.sessions()
        stored = session.get(PaymentRequest, request["id"])
        self.assertIsNone(stored.proof_path)
        self.assertIsNone(stored.proof_uploaded_at)
        session.close()

    def test_rejected_proof_is_deleted_and_dangerous_content_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.billing.create_payment_request(
                self.user["id"], "basic", "monthly", b"MZ executable", ".png"
            )
        pdf = b"%PDF-1.7\nminimal"
        request = self.billing.create_payment_request(
            self.user["id"], "pro", "monthly", pdf, ".pdf"
        )
        session = self.sessions()
        proof_path = Path(session.get(PaymentRequest, request["id"]).proof_path)
        session.close()
        self.billing.reject(request["id"], self.admin_user["id"])
        self.assertFalse(proof_path.exists())

    def test_full_plan_is_unlimited(self) -> None:
        self.admin.change_plan(self.admin_user["id"], self.user["id"], "full")
        for _ in range(25):
            self.usage.record_prediction(self.user["id"], "Fútbol")
        status = self.usage.can_user_predict(self.user["id"], "Fútbol")
        self.assertTrue(status["allowed"])
        self.assertIsNone(status["limit"])

    def test_admin_is_unlimited_without_paid_subscription(self) -> None:
        for _ in range(25):
            self.usage.record_prediction(self.admin_user["id"], "Fútbol")
        status = self.usage.can_user_predict(self.admin_user["id"], "Fútbol")
        self.assertTrue(status["allowed"])
        self.assertEqual(status["plan"], "full")
        self.assertIsNone(status["limit"])

    def test_admin_can_grant_loyalty_usage(self) -> None:
        self.usage.record_prediction(self.user["id"], "Fútbol")
        self.assertFalse(self.usage.can_user_predict(self.user["id"], "Fútbol")["allowed"])
        self.admin.grant_extra_predictions(self.admin_user["id"], self.user["id"], "Fútbol", 2)
        self.assertEqual(self.usage.can_user_predict(self.user["id"], "Fútbol")["remaining"], 2)

    def test_prediction_history_is_isolated_by_user(self) -> None:
        other = self.auth.register("other@example.com", "OtherPass123")
        with patch("database.prediction_repository.get_session", self.sessions):
            repository = PredictionRepository()
            repository.save_prediction_run(
                "Fútbol", "Local", "Visitante", "Modelo", 10000, [],
                context_json={"user_id": self.user["id"]},
            )
            repository.save_prediction_run(
                "Fútbol", "Otro", "Rival", "Modelo", 10000, [],
                context_json={"user_id": other["id"]},
            )
            runs = repository.list_recent_runs(user_id=self.user["id"])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["home_team"], "Local")


if __name__ == "__main__":
    unittest.main()
