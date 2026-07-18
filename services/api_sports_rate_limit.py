from __future__ import annotations

import os
import time
from threading import Lock
from typing import Mapping


class ApiSportsRateLimiter:
    """Conservative, process-wide guard for API-Sports free-plan traffic."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._last_request_at = 0.0
        self._blocked_until = 0.0
        self._minimum_interval = max(
            6.1,
            float(os.getenv("API_SPORTS_MIN_INTERVAL_SECONDS", "6.2")),
        )

    def wait_for_slot(self) -> None:
        """Smooth requests below 10/minute instead of sending bursts."""
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(
                self._blocked_until - now,
                self._minimum_interval - (now - self._last_request_at),
                0.0,
            )
            if wait_seconds:
                time.sleep(wait_seconds)
            self._last_request_at = time.monotonic()

    def observe(self, headers: Mapping[str, str], status_code: int) -> None:
        """Honor the provider's minute budget without exposing credentials."""
        normalized = {str(key).lower(): value for key, value in headers.items()}
        try:
            minute_remaining = int(normalized.get("x-ratelimit-remaining", "-1"))
        except (TypeError, ValueError):
            minute_remaining = -1

        retry_after = 0.0
        try:
            retry_after = float(normalized.get("retry-after", "0") or 0)
        except (TypeError, ValueError):
            pass

        if status_code == 429 or minute_remaining == 0:
            with self._lock:
                self._blocked_until = max(
                    self._blocked_until,
                    time.monotonic() + max(retry_after, 60.0),
                )


api_sports_rate_limiter = ApiSportsRateLimiter()
