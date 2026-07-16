from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from core.game_status import is_finished_status
from core.paths import CACHE_DIR


load_dotenv()


class SportmonksFootballAPI:
    """Proveedor complementario opcional para fixtures de fútbol."""

    base_url = "https://api.sportmonks.com/v3/football"

    def __init__(self) -> None:
        self.token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
        self.cache_dir = CACHE_DIR / "sportmonks_football"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.token)

    def get_games_by_date(self, fixture_date: str, force_refresh: bool = False) -> list[dict[str, Any]]:
        data = self._get_paginated(
            path=f"fixtures/date/{fixture_date}",
            params={"include": "league;participants;state;scores", "timezone": "UTC"},
            cache_key=f"fixtures_{fixture_date}",
            force_refresh=force_refresh,
            max_hours=6,
        )
        return [self._convert_fixture(item) for item in data if self._has_participants(item)]

    def get_recent_team_fixtures(
        self,
        team_id: int,
        last: int = 10,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        end = date.today()
        start = end - timedelta(days=550)
        data = self._get_paginated(
            path=f"fixtures/between/{start.isoformat()}/{end.isoformat()}/{team_id}",
            params={"include": "league;participants;state;scores", "timezone": "UTC", "order": "desc"},
            cache_key=f"team_{team_id}_last_{last}",
            force_refresh=force_refresh,
            max_hours=12,
        )
        converted = [self._convert_fixture(item) for item in data if self._has_participants(item)]
        finished = [
            item for item in converted
            if is_finished_status((item.get("fixture") or {}).get("status"))
        ]
        return finished[:last]

    def _get_paginated(
        self,
        path: str,
        params: dict[str, Any],
        cache_key: str,
        force_refresh: bool,
        max_hours: int,
    ) -> list[dict[str, Any]]:
        if not self.available:
            return []
        cache_file = self.cache_dir / f"{cache_key}.json"
        cached = self._read_cache(cache_file, max_hours if not force_refresh else None, allow_missing=True)
        if cached is not None and not force_refresh:
            return cached

        results: list[dict[str, Any]] = []
        page = 1
        try:
            while True:
                response = requests.get(
                    f"{self.base_url}/{path.lstrip('/')}",
                    params={**params, "api_token": self.token, "per_page": 50, "page": page},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                page_data = payload.get("data", [])
                if isinstance(page_data, list):
                    results.extend(item for item in page_data if isinstance(item, dict))
                pagination = payload.get("pagination") or {}
                if not pagination.get("has_more"):
                    break
                page += 1
            cache_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            return results
        except (requests.RequestException, ValueError, OSError):
            stale = self._read_cache(cache_file, None, allow_missing=True)
            if stale is not None:
                return stale
            raise

    @staticmethod
    def _read_cache(cache_file: Path, max_hours: int | None, allow_missing: bool = False) -> list[dict[str, Any]] | None:
        if not cache_file.exists():
            return None if allow_missing else []
        if max_hours is not None:
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age > timedelta(hours=max_hours):
                return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _has_participants(fixture: dict[str, Any]) -> bool:
        locations = {(item.get("meta") or {}).get("location") for item in fixture.get("participants", [])}
        return {"home", "away"}.issubset(locations)

    @classmethod
    def _convert_fixture(cls, fixture: dict[str, Any]) -> dict[str, Any]:
        participants = fixture.get("participants") or []
        home = next(item for item in participants if (item.get("meta") or {}).get("location") == "home")
        away = next(item for item in participants if (item.get("meta") or {}).get("location") == "away")
        league = fixture.get("league") or {}
        state = fixture.get("state") or {}
        scores = fixture.get("scores") or []
        current_scores = {
            (item.get("score") or {}).get("participant"): (item.get("score") or {}).get("goals")
            for item in scores
            if item.get("description") == "CURRENT"
        }
        return {
            "provider": "sportmonks",
            "fixture": {
                "id": f"sportmonks:{fixture.get('id')}",
                "date": str(fixture.get("starting_at") or "").replace(" ", "T") + "Z",
                "status": {
                    "short": state.get("state") or state.get("short_name") or "NS",
                    "long": state.get("name") or "Not Started",
                },
            },
            "league": {
                "id": league.get("id") or fixture.get("league_id"),
                "name": league.get("name") or "Competencia desconocida",
                "country": (league.get("country") or {}).get("name") or "",
            },
            "teams": {
                "home": {"id": home.get("id"), "name": home.get("name"), "logo": home.get("image_path") or ""},
                "away": {"id": away.get("id"), "name": away.get("name"), "logo": away.get("image_path") or ""},
            },
            "goals": {
                "home": current_scores.get(home.get("id")),
                "away": current_scores.get(away.get("id")),
            },
        }
