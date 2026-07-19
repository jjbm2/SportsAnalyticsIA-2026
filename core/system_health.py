from __future__ import annotations

import os
from typing import Any

from sqlalchemy import text

from database.database import engine


def deployment_health() -> dict[str, Any]:
    """Return non-sensitive runtime status for the administrator dashboard."""
    database_ok = False
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        database_ok = False

    backend = "PostgreSQL persistente" if engine.dialect.name == "postgresql" else "SQLite local"
    return {
        "database_ok": database_ok,
        "database_backend": backend,
        "api_sports": bool((os.getenv("API_SPORTS_KEY") or "").strip()),
        "sportmonks": bool((os.getenv("SPORTMONKS_API_TOKEN") or "").strip()),
        "sportsdataio": bool((os.getenv("SPORTSDATA_API_KEY") or "").strip()),
        "balldontlie": bool(
            (os.getenv("BALLDONTLIE_API_KEY") or os.getenv("BALL_DONT_LIE_API_KEY") or "").strip()
        ),
    }
