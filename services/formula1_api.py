from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from core.paths import CACHE_DIR


class Formula1API:
    base_url = "https://api.jolpi.ca/ergast/f1"

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR / "formula1"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_leagues(self, force_refresh: bool = False) -> dict[str, Any]:
        return {
            "response": [{
                "league": {"id": 1, "name": "Formula 1"},
                "country": {"name": "World"},
                "seasons": [{"year": date.today().year}],
            }],
            "results": 1,
            "errors": [],
            "_source": "local",
        }

    def get_games_by_date(
        self,
        date: str,
        league_id: int | None = None,
        season: int | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        selected_year = season or int(date[:4])
        races = self.get_season_races(selected_year, force_refresh)
        selected = [self._to_event(race) for race in races if race.get("date") == date]
        return {
            "response": selected,
            "results": len(selected),
            "errors": [],
            "_source": "cache_or_api",
        }

    def get_season_races(self, season: int, force_refresh: bool = False) -> list[dict[str, Any]]:
        payload = self._get_json(
            path=f"{season}/races/",
            cache_key=f"races_{season}",
            max_hours=24,
            force_refresh=force_refresh,
        )
        return (((payload.get("MRData") or {}).get("RaceTable") or {}).get("Races") or [])

    def get_results(self, season: int, force_refresh: bool = False) -> list[dict[str, Any]]:
        races_by_round: dict[str, dict[str, Any]] = {}
        offset = 0
        total = 1
        while offset < total:
            payload = self._get_json(
                path=f"{season}/results/",
                cache_key=f"results_{season}_offset_{offset}",
                max_hours=12,
                force_refresh=force_refresh,
                params={"limit": 100, "offset": offset},
            )
            mr_data = payload.get("MRData") or {}
            try:
                total = int(mr_data.get("total") or 0)
            except (TypeError, ValueError):
                total = 0
            page_races = ((mr_data.get("RaceTable") or {}).get("Races") or [])
            for race in page_races:
                round_key = str(race.get("round") or len(races_by_round))
                existing = races_by_round.setdefault(round_key, {**race, "Results": []})
                known = {item.get("number") for item in existing.get("Results", [])}
                existing["Results"].extend(
                    item for item in race.get("Results", [])
                    if item.get("number") not in known
                )
            offset += 100
            if not page_races:
                break
        return sorted(races_by_round.values(), key=lambda race: int(race.get("round") or 0))

    def _get_json(
        self,
        path: str,
        cache_key: str,
        max_hours: int,
        force_refresh: bool,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        cached = self._read_cache(cache_file, max_hours)
        if cached is not None and not force_refresh:
            return cached
        try:
            response = requests.get(
                f"{self.base_url}/{path.lstrip('/')}",
                params=params,
                timeout=30,
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
    def _read_cache(cache_file: Path, max_hours: int | None) -> dict[str, Any] | None:
        if not cache_file.exists():
            return None
        if max_hours is not None:
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age > timedelta(hours=max_hours):
                return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _to_event(race: dict[str, Any]) -> dict[str, Any]:
        circuit = race.get("Circuit") or {}
        location = circuit.get("Location") or {}
        race_date = str(race.get("date") or "")
        completed = race_date < date.today().isoformat()
        return {
            "provider": "jolpica",
            "race": {
                "id": f"f1:{race.get('season')}:{race.get('round')}",
                "name": race.get("raceName") or "Grand Prix",
                "round": int(race.get("round") or 0),
                "season": int(race.get("season") or date.today().year),
                "date": race_date,
                "time": race.get("time") or "",
                "status": "Completed" if completed else "Scheduled",
            },
            "circuit": {
                "id": circuit.get("circuitId"),
                "name": circuit.get("circuitName") or "Circuito",
                "city": location.get("locality") or "",
                "country": location.get("country") or "World",
            },
        }
