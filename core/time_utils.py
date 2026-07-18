from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return naive UTC for compatibility with existing database columns."""
    return datetime.now(UTC).replace(tzinfo=None)
