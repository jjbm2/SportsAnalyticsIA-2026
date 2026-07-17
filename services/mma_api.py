from __future__ import annotations

from datetime import date
from typing import Any

from services.base_sports_api import BaseSportsAPI
from core.event_cache_policy import event_cache_hours


class MMAAPI(BaseSportsAPI):
    def __init__(self) -> None:
        super().__init__(
            base_url="https://v1.mma.api-sports.io",
            sport_name="mma",
        )

    def get_leagues(self, force_refresh: bool = False) -> dict[str, Any]:
        return {
            "response": [{
                "league": {"id": "mma", "name": "MMA"},
                "country": {"name": "World"},
                "seasons": [{"year": date.today().year}],
            }],
            "results": 1,
            "errors": [],
        }

    def get_games_by_date(
        self,
        date: str,
        league_id: Any = None,
        season: int | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        payload = self.get(
            "fights",
            {"date": date},
            cache_key=date,
            force_refresh=force_refresh,
            max_hours=event_cache_hours(date),
        )
        fights = [self._normalize(item) for item in payload.get("response", []) if isinstance(item, dict)]
        payload["response"] = fights
        payload["results"] = len(fights)
        return payload

    @staticmethod
    def _normalize(item: dict[str, Any]) -> dict[str, Any]:
        fighters = item.get("fighters") or {}
        first = fighters.get("first") or fighters.get("home") or item.get("fighter1") or {}
        second = fighters.get("second") or fighters.get("away") or item.get("fighter2") or {}
        status_data = item.get("status") or {}
        status = status_data.get("short") if isinstance(status_data, dict) else status_data
        finished = str(status or "").upper() == "FT"
        winner = item.get("winner") or (item.get("result") or {}).get("winner") or {}
        winner_id = winner.get("id") if isinstance(winner, dict) else None
        category = item.get("category") or "MMA"
        category_name = category.get("name") if isinstance(category, dict) else str(category)
        return {
            "id": item.get("id"),
            "provider": "api_sports",
            "date": item.get("date"),
            "time": item.get("time"),
            "status": status_data or "NS",
            "league": {"id": "mma", "name": "MMA", "country": "World"},
            "teams": {
                "home": {"id": first.get("id"), "name": first.get("name"), "logo": first.get("logo") or first.get("photo")},
                "away": {"id": second.get("id"), "name": second.get("name"), "logo": second.get("logo") or second.get("photo")},
            },
            "scores": {
                "home": {"total": 1 if winner_id == first.get("id") else 0 if finished else None},
                "away": {"total": 1 if winner_id == second.get("id") else 0 if finished else None},
            },
            "analysis_context": {
                "kind": "mma",
                "weight_class": category_name,
                "home_profile": first,
                "away_profile": second,
            },
        }

    def get_fighter_record(self, fighter_id: Any, force_refresh: bool = False) -> dict[str, Any]:
        payload = self.get(
            "fighters/records",
            {"id": fighter_id},
            cache_key=f"record_{fighter_id}",
            force_refresh=force_refresh,
            max_hours=24,
        )
        response = payload.get("response", [])
        return response[0] if response else {}
