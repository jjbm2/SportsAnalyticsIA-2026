from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from core.paths import CACHE_DIR

load_dotenv()


class BallDontLieNBAAPI:
    base_url = "https://api.balldontlie.io/v1"

    def __init__(self) -> None:
        self.api_key = os.getenv("BALLDONTLIE_API_KEY") or os.getenv("BALL_DONT_LIE_API_KEY")
        self.cache_dir = CACHE_DIR / "balldontlie_nba"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def get_teams(self, force_refresh: bool = False) -> dict[str, Any]:
        return self._get("teams", "teams", 24 * 30, force_refresh)

    def get_injuries(self, force_refresh: bool = False) -> dict[str, Any]:
        return self._get("player_injuries", "player_injuries", 2, force_refresh, {"per_page": 100})

    def _get(self, endpoint: str, cache_key: str, max_hours: int, force_refresh: bool,
             params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            return {"data": [], "_coverage": "unavailable"}
        cache_file = self.cache_dir / f"{cache_key}.json"
        cached = self._read_cache(cache_file, max_hours)
        if cached is not None and not force_refresh:
            return cached
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}", headers={"Authorization": self.api_key},
                params=params, timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload
        except (requests.RequestException, ValueError, OSError):
            stale = self._read_cache(cache_file, None)
            if stale is not None:
                return stale
            raise

    @staticmethod
    def _read_cache(path: Path, max_hours: int | None) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if max_hours is not None and datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) > timedelta(hours=max_hours):
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None
