from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import requests

from core.event_cache_policy import event_cache_hours
from core.game_status import is_finished_status
from core.paths import CACHE_DIR
from services.http_client import build_retry_session


_LOCKS: dict[str, Lock] = {}
_LOCKS_GUARD = Lock()


def _cache_lock(path: Path) -> Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, Lock())


class SportsDataSoccerAPI:
    """Optional SportsDataIO soccer provider with shared disk caching."""

    base_url = "https://api.sportsdata.io/v4/soccer/scores/json"

    def __init__(self) -> None:
        self.api_key = (os.getenv("SPORTSDATA_API_KEY") or "").strip()
        raw_competitions = os.getenv("SPORTSDATA_SOCCER_COMPETITIONS", "3")
        self.competitions = [
            item.strip() for item in raw_competitions.split(",") if item.strip()
        ]
        self.http = build_retry_session()
        self.cache_dir = CACHE_DIR / "sportsdata_soccer"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.competitions)

    def get_games_by_date(
        self,
        fixture_date: str,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.available:
            return []
        games: list[dict[str, Any]] = []
        for competition in self.competitions:
            payload = self._get(
                path=f"GamesByDate/{competition}/{fixture_date}",
                cache_key=f"games_{competition}_{fixture_date}",
                max_hours=event_cache_hours(fixture_date),
                force_refresh=force_refresh,
            )
            raw_games = payload if isinstance(payload, list) else []
            games.extend(
                self._convert_game(item, competition)
                for item in raw_games
                if isinstance(item, dict)
            )
        return games

    def get_recent_team_fixtures(
        self,
        team_id: int,
        competition: str | int | None,
        last: int = 20,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        competition_key = str(competition or "").strip()
        if not self.available or not competition_key:
            return []
        payload = self._get(
            path=f"CompetitionDetails/{competition_key}",
            cache_key=f"competition_{competition_key}",
            max_hours=24,
            force_refresh=force_refresh,
        )
        if isinstance(payload, dict):
            raw_games = payload.get("Games") or payload.get("games") or []
        else:
            raw_games = payload if isinstance(payload, list) else []
        converted = [
            self._convert_game(item, competition_key)
            for item in raw_games
            if isinstance(item, dict)
            and team_id in {item.get("HomeTeamId"), item.get("AwayTeamId")}
        ]
        finished = [
            item for item in converted
            if is_finished_status((item.get("fixture") or {}).get("status"))
        ]
        finished.sort(
            key=lambda item: str((item.get("fixture") or {}).get("date") or ""),
            reverse=True,
        )
        return finished[:last]

    def _get(
        self,
        path: str,
        cache_key: str,
        max_hours: float,
        force_refresh: bool,
    ) -> Any:
        cache_file = self.cache_dir / f"{cache_key.replace('/', '_')}.json"
        if not force_refresh:
            cached = self._read_cache(cache_file, max_hours)
            if cached is not None:
                return cached
        with _cache_lock(cache_file):
            if not force_refresh:
                cached = self._read_cache(cache_file, max_hours)
                if cached is not None:
                    return cached
            try:
                response = self.http.get(
                    f"{self.base_url}/{path.lstrip('/')}",
                    headers={"Ocp-Apim-Subscription-Key": self.api_key},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                cache_file.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                return payload
            except (requests.RequestException, ValueError, OSError):
                stale = self._read_cache(cache_file, None)
                if stale is not None:
                    return stale
                raise RuntimeError("SportsDataIO provider request failed") from None

    @staticmethod
    def _read_cache(cache_file: Path, max_hours: float | None) -> Any | None:
        if not cache_file.exists():
            return None
        if max_hours is not None:
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age > timedelta(hours=max_hours):
                return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _convert_game(game: dict[str, Any], competition: str | int) -> dict[str, Any]:
        status = game.get("Status") or game.get("GameStatus") or "Scheduled"
        home_score = game.get("HomeTeamScore")
        away_score = game.get("AwayTeamScore")
        return {
            "provider": "sportsdataio",
            "fixture": {
                "id": f"sportsdata:{game.get('GameId')}",
                "date": game.get("DateTime") or game.get("Day") or "",
                "status": {"short": status, "long": status},
            },
            "league": {
                "id": str(competition),
                "name": game.get("CompetitionName") or f"SportsDataIO {competition}",
                "country": game.get("AreaName") or "",
                "season": game.get("Season"),
            },
            "teams": {
                "home": {
                    "id": game.get("HomeTeamId"),
                    "name": game.get("HomeTeamName") or game.get("HomeTeamKey") or "Local",
                    "logo": "",
                },
                "away": {
                    "id": game.get("AwayTeamId"),
                    "name": game.get("AwayTeamName") or game.get("AwayTeamKey") or "Visitante",
                    "logo": "",
                },
            },
            "goals": {"home": home_score, "away": away_score},
        }
