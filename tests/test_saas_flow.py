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
from auth.session_security import (
    authenticated_session_is_active,
    clear_authenticated_session,
    establish_authenticated_session,
)
from auth.streamlit_views import (
    clear_login_failures,
    login_lock_remaining,
    record_login_failure,
)
from billing.billing_manager import BillingManager
from database.database import Base
from database.models import PaymentRequest, Subscription, User
from database.prediction_repository import PredictionRepository
from promotions.promotion_service import PromotionService
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
        self.promotions = PromotionService(self.sessions)
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

    def test_registration_rejects_weak_or_oversized_passwords(self) -> None:
        for password in ("abcdefgh", "12345678", " Password1", "Password1 "):
            with self.subTest(password=password), self.assertRaises(ValueError):
                self.auth.register(f"weak-{abs(hash(password))}@example.com", password)
        with self.assertRaises(ValueError):
            self.auth.register("long@example.com", "A1" + ("ñ" * 36))

    def test_registration_rejects_fake_disposable_and_malformed_emails(self) -> None:
        invalid_emails = (
            "test@test.com",
            "fake@gmail.com",
            "real@mailinator.com",
            "double..dot@gmail.com",
            ".leading@gmail.com",
            "missing-domain@",
        )
        for email in invalid_emails:
            with self.subTest(email=email), self.assertRaises(ValueError):
                self.auth.register(email, "ValidPass123")

    def test_registration_accepts_normal_and_plus_alias_emails(self) -> None:
        user = self.auth.register("persona+promo@gmail.com", "ValidPass123")
        self.assertEqual(user["email"], "persona+promo@gmail.com")

    def test_login_attempts_are_temporarily_limited_per_session(self) -> None:
        state = {}
        for attempt in range(4):
            self.assertEqual(record_login_failure(state, now=1000 + attempt), 0)
        self.assertEqual(record_login_failure(state, now=1004), 300)
        self.assertEqual(login_lock_remaining(state, now=1005), 299)
        self.assertEqual(login_lock_remaining(state, now=1304), 0)
        self.assertEqual(state, {})
        record_login_failure(state, now=1400)
        clear_login_failures(state)
        self.assertEqual(state, {})

    def test_authenticated_session_survives_navigation_and_expires_when_idle(self) -> None:
        state = {"screen": "home", "selected_sport": "Fútbol"}
        establish_authenticated_session(state, self.user)
        session_id = state["auth_session_id"]

        state["screen"] = "sport"
        self.assertTrue(
            authenticated_session_is_active(
                state,
                now=state["auth_last_activity"] + 60,
            )
        )
        self.assertEqual(state["current_user"]["id"], self.user["id"])
        self.assertEqual(state["auth_session_id"], session_id)

        last_activity = state["auth_last_activity"]
        self.assertFalse(
            authenticated_session_is_active(
                state,
                now=last_activity + 12 * 60 * 60 + 1,
            )
        )
        self.assertIsNone(state["current_user"])
        self.assertEqual(state["screen"], "home")

    def test_logout_clears_user_private_state(self) -> None:
        state = {
            "screen": "admin",
            "current_user": self.user,
            "auth_session_id": "secret",
            "auth_last_activity": 1.0,
            "checkout_plan": "pro",
            "cookies_accepted": True,
        }

        clear_authenticated_session(state)

        self.assertIsNone(state["current_user"])
        self.assertNotIn("auth_session_id", state)
        self.assertNotIn("checkout_plan", state)
        self.assertTrue(state["cookies_accepted"])

    def test_free_limit_is_per_sport_and_resets_by_date(self) -> None:
        day_one = date(2026, 7, 16)
        day_two = day_one + timedelta(days=1)
        self.assertTrue(self.usage.can_user_predict(self.user["id"], "Fútbol", day_one)["allowed"])
        self.usage.record_prediction(self.user["id"], "Fútbol", day_one)
        self.assertFalse(self.usage.can_user_predict(self.user["id"], "Fútbol", day_one)["allowed"])
        self.assertTrue(self.usage.can_user_predict(self.user["id"], "Basketball", day_one)["allowed"])
        self.assertTrue(self.usage.can_user_predict(self.user["id"], "Fútbol", day_two)["allowed"])

    def test_failed_analysis_can_release_reserved_usage(self) -> None:
        today = date(2026, 7, 16)
        self.assertEqual(self.usage.record_prediction(self.user["id"], "Basketball", today), 1)
        self.assertFalse(self.usage.can_user_predict(self.user["id"], "Basketball", today)["allowed"])
        self.assertEqual(self.usage.release_prediction(self.user["id"], "Basketball", today), 0)
        status = self.usage.can_user_predict(self.user["id"], "Basketball", today)
        self.assertTrue(status["allowed"])
        self.assertEqual(status["used"], 0)

    def test_opening_promotion_grants_five_daily_predictions_for_five_days(self) -> None:
        start = date(2026, 7, 16)
        status = self.promotions.redeem(self.user["id"], " apertura5 ", start)
        self.assertTrue(status["active"])
        self.assertEqual(status["days_remaining"], 5)
        for _ in range(5):
            self.usage.record_prediction(self.user["id"], "Fútbol", start)
        self.assertFalse(self.usage.can_user_predict(self.user["id"], "Fútbol", start)["allowed"])
        self.assertTrue(self.usage.can_user_predict(self.user["id"], "Basketball", start)["allowed"])
        self.assertTrue(self.promotions.status(self.user["id"], start + timedelta(days=4))["active"])
        expired = self.promotions.status(self.user["id"], start + timedelta(days=5))
        self.assertTrue(expired["expired"])
        self.assertEqual(
            self.usage.can_user_predict(self.user["id"], "Fútbol", start + timedelta(days=5))["limit"],
            1,
        )

    def test_opening_promotion_can_only_be_redeemed_once(self) -> None:
        self.promotions.redeem(self.user["id"], "APERTURA5", date(2026, 7, 16))
        with self.assertRaises(ValueError):
            self.promotions.redeem(self.user["id"], "APERTURA5", date(2026, 7, 17))

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

    def test_environment_admin_promotes_existing_user(self) -> None:
        self.auth.ensure_admin_from_environment("user@example.com", "IgnoredPass123")
        promoted = self.auth.authenticate("user@example.com", "UserPass123")

        self.assertTrue(promoted["is_admin"])
        status = self.usage.can_user_predict(promoted["id"], "Fútbol")
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
