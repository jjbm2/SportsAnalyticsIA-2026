from __future__ import annotations

from datetime import date, datetime

from core.event_time import sports_timezone


def event_cache_hours(value: str | date | datetime, *, today: date | None = None) -> float:
    """Cache event lists according to how likely their schedule is to change."""
    if isinstance(value, datetime):
        event_day = value.date()
    elif isinstance(value, date):
        event_day = value
    else:
        event_day = date.fromisoformat(str(value)[:10])

    local_today = today or datetime.now(sports_timezone()).date()
    if event_day < local_today:
        return 24 * 14
    if event_day == local_today:
        return 0.25
    return 24 * 7
