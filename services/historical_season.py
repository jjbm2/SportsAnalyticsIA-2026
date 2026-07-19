from __future__ import annotations

import os
from typing import Any


def accessible_history_season(sport: str, requested_season: Any) -> int:
    """Return the newest season available to the configured API-Sports plan.

    Current fixtures and historical feeds have different access rules on the
    free plan. The value remains configurable so a paid plan can move the
    historical window forward without a code change.
    """
    sport_key = sport.upper().replace("-", "_").replace(" ", "_")
    configured = (
        os.getenv(f"API_{sport_key}_HISTORY_SEASON")
        or os.getenv("API_SPORTS_HISTORY_SEASON")
        or "2024"
    )
    allowed = _year(configured, 2024)
    requested = _year(requested_season, allowed)
    return min(requested, allowed)


def _year(value: Any, default: int) -> int:
    try:
        return int(str(value).strip().split("-", 1)[0])
    except (TypeError, ValueError):
        return default
