from __future__ import annotations

from typing import Any

from core.logger import logger
from services.balldontlie_nba_api import BallDontLieNBAAPI
from services.football_api import FootballAPI
from services.nfl_api import NFLAPI


class PlayerAvailabilityService:
    """Consulta disponibilidad solo donde el proveedor ofrece cobertura real."""

    def __init__(self, football_api: Any = None, nfl_api: Any = None, nba_api: Any = None) -> None:
        self._football_api = football_api
        self._nfl_api = nfl_api
        self._nba_api = nba_api

    def get_match_availability(
        self,
        sport: str,
        match: dict[str, Any],
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        neutral = self._neutral(sport)
        game_id = match.get("game_id")
        if not game_id or str(game_id).startswith("sportmonks:"):
            return neutral
        try:
            if sport == "Fútbol":
                return self._football(game_id, match, force_refresh)
            if sport == "NFL":
                return self._nfl(game_id, match, force_refresh)
            if sport == "Basketball":
                return self._basketball(match, force_refresh)
        except Exception as error:
            logger.warning("Disponibilidad de jugadores no disponible para %s: %s", sport, error)
        return neutral

    def _football(self, game_id: Any, match: dict[str, Any], force_refresh: bool) -> dict[str, Any]:
        api = self._football_api or FootballAPI()
        injuries = api.get(
            endpoint="injuries",
            params={"fixture": game_id},
            cache_key=f"injuries_fixture_{game_id}",
            force_refresh=force_refresh,
            max_hours=2,
        )
        lineups = api.get(
            endpoint="fixtures/lineups",
            params={"fixture": game_id},
            cache_key=f"lineups_fixture_{game_id}",
            force_refresh=force_refresh,
            max_hours=2,
        )
        return self._summarize(match, injuries, lineups)

    def _nfl(self, game_id: Any, match: dict[str, Any], force_refresh: bool) -> dict[str, Any]:
        api = self._nfl_api or NFLAPI()
        injuries = api.get(
            endpoint="injuries",
            params={"game": game_id},
            cache_key=f"injuries_game_{game_id}",
            force_refresh=force_refresh,
            max_hours=3,
        )
        return self._summarize(match, injuries, {"response": []})

    def _basketball(self, match: dict[str, Any], force_refresh: bool) -> dict[str, Any]:
        api = self._nba_api or BallDontLieNBAAPI()
        if not api.available or str(match.get("league") or "").strip().upper() != "NBA":
            return self._neutral("Basketball")
        teams = [item for item in api.get_teams(force_refresh).get("data", []) if isinstance(item, dict)]
        injuries = [item for item in api.get_injuries(force_refresh).get("data", []) if isinstance(item, dict)]
        team_ids = {
            self._normalize(item.get("full_name") or item.get("name")): item.get("id")
            for item in teams
        }

        def team_summary(team_name: Any) -> dict[str, int]:
            team_id = team_ids.get(self._normalize(team_name))
            reported = sum(
                1 for item in injuries
                if (item.get("player") or {}).get("team_id") == team_id
                and str(item.get("status") or "").strip().lower() not in {"available", "active"}
            )
            return {"reported_absences": reported, "confirmed_starters": 0}

        return {
            "coverage": "confirmed" if teams else "limited",
            "source": "balldontlie",
            "home": team_summary(match.get("home")),
            "away": team_summary(match.get("away")),
        }

    @staticmethod
    def _summarize(
        match: dict[str, Any],
        injuries_payload: dict[str, Any],
        lineups_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if injuries_payload.get("errors") or lineups_payload.get("errors"):
            return PlayerAvailabilityService._neutral(str(match.get("sport") or ""))
        injuries = [item for item in injuries_payload.get("response", []) if isinstance(item, dict)]
        lineups = [item for item in lineups_payload.get("response", []) if isinstance(item, dict)]
        home_id, away_id = match.get("home_id"), match.get("away_id")

        def team_summary(team_id: Any) -> dict[str, int]:
            team_injuries = [
                item for item in injuries
                if (item.get("team") or {}).get("id") == team_id
            ]
            lineup = next(
                (item for item in lineups if (item.get("team") or {}).get("id") == team_id),
                {},
            )
            return {
                "reported_absences": len(team_injuries),
                "confirmed_starters": len(lineup.get("startXI") or []),
            }

        coverage = "confirmed" if injuries or lineups else "limited"
        return {
            "coverage": coverage,
            "source": "api_sports",
            "home": team_summary(home_id),
            "away": team_summary(away_id),
        }

    @staticmethod
    def _neutral(sport: str) -> dict[str, Any]:
        return {
            "coverage": "unavailable",
            "source": None,
            "sport": sport,
            "home": {"reported_absences": 0, "confirmed_starters": 0},
            "away": {"reported_absences": 0, "confirmed_starters": 0},
        }

    @staticmethod
    def _normalize(value: Any) -> str:
        return " ".join(str(value or "").casefold().replace("-", " ").split())
