from __future__ import annotations

import secrets
import time
from collections.abc import MutableMapping
from typing import Any


SESSION_IDLE_TIMEOUT_SECONDS = 12 * 60 * 60

_AUTH_KEYS = ("current_user", "auth_session_id", "auth_last_activity")
_USER_PRIVATE_KEYS = (
    "checkout_plan",
    "checkout_cycle",
    "match_options",
    "football_match_quality",
    "sports_connectivity",
)


def establish_authenticated_session(state: MutableMapping, user: dict[str, Any]) -> None:
    """Start a fresh browser session without retaining another user's private state."""
    for key in _USER_PRIVATE_KEYS:
        state.pop(key, None)
    state["current_user"] = user
    state["auth_session_id"] = secrets.token_urlsafe(24)
    state["auth_last_activity"] = time.time()


def authenticated_session_is_active(
    state: MutableMapping,
    *,
    now: float | None = None,
    idle_timeout: int = SESSION_IDLE_TIMEOUT_SECONDS,
) -> bool:
    if not state.get("current_user"):
        return False

    current_time = time.time() if now is None else float(now)
    last_activity = float(state.get("auth_last_activity", current_time) or current_time)
    if current_time - last_activity > idle_timeout:
        clear_authenticated_session(state)
        return False

    state.setdefault("auth_session_id", secrets.token_urlsafe(24))
    state["auth_last_activity"] = current_time
    return True


def clear_authenticated_session(state: MutableMapping) -> None:
    """Remove identity and user-specific state while keeping harmless preferences."""
    for key in (*_AUTH_KEYS, *_USER_PRIVATE_KEYS):
        state.pop(key, None)
    state["current_user"] = None
    state["screen"] = "home"
    state["public_screen"] = "landing"
