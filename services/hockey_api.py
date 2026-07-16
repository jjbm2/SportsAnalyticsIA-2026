from __future__ import annotations

from typing import Any

from services.base_sports_api import BaseSportsAPI


class HockeyAPI(BaseSportsAPI):
    def __init__(self) -> None:
        super().__init__(
            base_url="https://v1.hockey.api-sports.io",
            sport_name="hockey",
        )

    def get_leagues(self, force_refresh: bool = False) -> dict[str, Any]:
        return self.get("leagues", cache_key="all_leagues", force_refresh=force_refresh, max_hours=720)

    def get_games_by_date(
        self,
        date: str,
        league_id: int | None = None,
        season: int | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"date": date}
        cache_key = date
        if league_id is not None:
            params["league"] = league_id
            cache_key += f"_league_{league_id}"
        if season is not None:
            params["season"] = season
            cache_key += f"_season_{season}"
        payload = self.get("games", params, cache_key, force_refresh, 6)
        for game in payload.get("response", []):
            if not isinstance(game, dict):
                continue
            league = game.get("league") or {}
            game["provider"] = "api_sports"
            game["analysis_context"] = {
                "kind": "hockey",
                "league_id": league.get("id"),
                "season": league.get("season"),
            }
        return payload
