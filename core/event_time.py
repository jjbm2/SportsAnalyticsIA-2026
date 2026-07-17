from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_SPORTS_TIMEZONE = "America/Mexico_City"


def sports_timezone() -> ZoneInfo:
    configured = (os.getenv("SPORTS_TIMEZONE") or DEFAULT_SPORTS_TIMEZONE).strip()
    try:
        return ZoneInfo(configured)
    except ZoneInfoNotFoundError:
        return ZoneInfo(DEFAULT_SPORTS_TIMEZONE)


def event_local_datetime(event: dict[str, Any]) -> datetime | None:
    raw_date = str(event.get("date") or "").strip()
    raw_time = str(event.get("time") or "").strip()
    if not raw_date:
        return None
    value = raw_date if "T" in raw_date else f"{raw_date}T{raw_time or '00:00:00'}"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    timezone = sports_timezone()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone)
    return parsed.astimezone(timezone)


def event_matches_local_date(event: dict[str, Any], selected_date: date) -> bool:
    parsed = event_local_datetime(event)
    if parsed is not None:
        return parsed.date() == selected_date
    return str(event.get("date") or "")[:10] == selected_date.isoformat()


def event_has_started(event: dict[str, Any], now: datetime | None = None) -> bool:
    raw_date = str(event.get("date") or "").strip()
    raw_time = str(event.get("time") or "").strip()
    if "T" not in raw_date and not raw_time:
        return False
    scheduled = event_local_datetime(event)
    if scheduled is None:
        return False
    current = now or datetime.now(sports_timezone())
    if current.tzinfo is None:
        current = current.replace(tzinfo=sports_timezone())
    else:
        current = current.astimezone(sports_timezone())
    return scheduled <= current
