import os
from typing import Any

from core.game_status import is_finished_status

from services.football_api import FootballAPI
from services.sportmonks_football_api import SportmonksFootballAPI


class FootballDataService:
    def __init__(self):
        self.api = FootballAPI()
        self.sportmonks_api = SportmonksFootballAPI()

    def get_recent_team_fixtures(
        self,
        team_id: int,
        last: int = 10,
        force_refresh: bool = False,
        provider: str = "api_sports",
    ) -> list[dict[str, Any]]:
        if provider == "sportmonks":
            return self.sportmonks_api.get_recent_team_fixtures(
                team_id=team_id,
                last=last,
                force_refresh=force_refresh,
            )

        # API-Football free plans reject `last` and currently expose historical
        # seasons only through 2024. Fetch the newest allowed season once and
        # select the latest completed fixtures locally.
        history_season = int(os.getenv("API_FOOTBALL_HISTORY_SEASON", "2024"))
        data = self.api.get(
            endpoint="fixtures",
            params={
                "team": team_id,
                "season": history_season,
            },
            cache_key=f"team_{team_id}_season_{history_season}",
            force_refresh=force_refresh,
            # A closed historical season is immutable in normal operation.
            max_hours=24 * 30,
        )
        fixtures = [
            fixture for fixture in data.get("response", [])
            if isinstance(fixture, dict)
            and is_finished_status((fixture.get("fixture") or {}).get("status"))
        ]
        fixtures.sort(
            key=lambda fixture: str((fixture.get("fixture") or {}).get("date") or ""),
            reverse=True,
        )
        return fixtures[:last]

    def build_team_profile(
        self,
        team_id: int,
        last: int = 10,
        force_refresh: bool = False,
        provider: str = "api_sports",
    ) -> dict[str, Any]:
        fixtures = self.get_recent_team_fixtures(
            team_id=team_id,
            last=last,
            force_refresh=force_refresh,
            provider=provider,
        )

        played = 0
        goals_for = 0
        goals_against = 0

        for fixture in fixtures:
            teams = fixture.get("teams", {})
            goals = fixture.get("goals", {})

            home_team = teams.get("home", {})
            away_team = teams.get("away", {})

            home_id = home_team.get("id")
            away_id = away_team.get("id")

            home_goals = goals.get("home")
            away_goals = goals.get("away")

            if home_goals is None or away_goals is None:
                continue

            if team_id == home_id:
                goals_for += home_goals
                goals_against += away_goals
                played += 1

            elif team_id == away_id:
                goals_for += away_goals
                goals_against += home_goals
                played += 1

        if played == 0:
            return {
                "team_id": team_id,
                "played": 0,
                "avg_scored": 1.0,
                "avg_conceded": 1.0,
            }

        return {
            "team_id": team_id,
            "played": played,
            "avg_scored": goals_for / played,
            "avg_conceded": goals_against / played,
        }
