from __future__ import annotations

from typing import Any


FINISHED_STATUSES = {
    "FT",
    "AOT",
    "AP",
    "AW",
    "FINAL",
    "FINISHED",
    "COMPLETED",
    "ENDED",
    "AFTER OVERTIME",
}

LIVE_STATUSES = {
    "LIVE",
    "IN PLAY",
    "1H",
    "HT",
    "2H",
    "ET",
    "BT",
    "P",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "OT",
    "IN PROGRESS",
}

UNAVAILABLE_STATUSES = FINISHED_STATUSES | LIVE_STATUSES | {
    "CANC",
    "CANCELLED",
    "CANCELED",
    "POST",
    "POSTPONED",
    "SUSP",
    "SUSPENDED",
    "INT",
    "INTERRUPTED",
    "PST",
    "ABD",
}


def status_values(status: Any) -> set[str]:
    values: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                collect(nested)
            return
        if value is not None and str(value).strip():
            values.add(str(value).strip().upper())

    collect(status)
    return values


def is_finished_status(status: Any) -> bool:
    values = status_values(status)
    return any(
        value in FINISHED_STATUSES
        or "FINAL" in value
        or "FINISHED" in value
        or "COMPLETED" in value
        for value in values
    )


def is_live_status(status: Any) -> bool:
    values = status_values(status)
    return any(
        value in LIVE_STATUSES
        or "IN PROGRESS" in value
        or "IN PLAY" in value
        for value in values
    )


def is_available_for_pregame(status: Any) -> bool:
    values = status_values(status)
    return not any(
        value in UNAVAILABLE_STATUSES
        or "FINAL" in value
        or "FINISHED" in value
        or "COMPLETED" in value
        or "CANCEL" in value
        or "POSTPON" in value
        or "SUSPEND" in value
        or "IN PROGRESS" in value
        or "IN PLAY" in value
        for value in values
    )


def extract_final_score(game: dict[str, Any], sport: str) -> tuple[float | None, float | None]:
    if sport == "Fútbol":
        goals = game.get("goals") or {}
        home = _as_float(goals.get("home"))
        away = _as_float(goals.get("away"))
        if home is not None and away is not None:
            return home, away
        fulltime = ((game.get("score") or {}).get("fulltime") or {})
        return _as_float(fulltime.get("home")), _as_float(fulltime.get("away"))

    scores = game.get("scores") or {}
    teams = game.get("teams") or {}
    home = _first_score(scores.get("home"), teams.get("home"))
    away = _first_score(scores.get("away"), teams.get("away"))
    if home is None:
        home = _as_float(game.get("home_team_score"))
    if away is None:
        away = _as_float(game.get("visitor_team_score"))
    return home, away


def _first_score(score: Any, team: Any) -> float | None:
    candidates = []
    if isinstance(score, dict):
        candidates.extend([score.get("total"), score.get("points")])
    else:
        candidates.append(score)
    if isinstance(team, dict):
        candidates.append(team.get("score"))

    for candidate in candidates:
        parsed = _as_float(candidate)
        if parsed is not None:
            return parsed
    return None


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None
