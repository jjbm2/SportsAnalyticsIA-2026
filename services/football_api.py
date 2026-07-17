from difflib import SequenceMatcher
import re
import unicodedata

from core.logger import logger
from core.event_time import sports_timezone
from services.base_sports_api import BaseSportsAPI
from services.sportmonks_football_api import SportmonksFootballAPI


class FootballAPI(BaseSportsAPI):
    def __init__(self):
        super().__init__(
            base_url="https://v3.football.api-sports.io",
            sport_name="football",
            require_api_key=False,
        )
        self.supplemental_api = SportmonksFootballAPI()

    def get_leagues(
        self,
        force_refresh: bool = False
    ):
        return self.get(
            endpoint="leagues",
            cache_key="all_leagues",
            force_refresh=force_refresh,
            max_hours=720
        )

    def get_games_by_date(
        self,
        date: str,
        league_id: int | None = None,
        season: int | None = None,
        force_refresh: bool = False
    ):
        params = {
            "date": date,
            "timezone": str(sports_timezone()),
        }

        cache_key = f"{date}_{str(sports_timezone()).replace('/', '_')}"

        if league_id is not None:
            params["league"] = league_id
            cache_key += f"_league_{league_id}"

        if season is not None:
            params["season"] = season
            cache_key += f"_season_{season}"

        primary_error: Exception | None = None
        try:
            primary = self.get(
                endpoint="fixtures",
                params=params,
                cache_key=cache_key,
                force_refresh=force_refresh,
                max_hours=6
            )
        except Exception as error:
            primary_error = error
            primary = {"response": [], "results": 0, "_source": "unavailable"}
            logger.warning("Proveedor principal de fútbol no disponible: %s", error)

        if not self.supplemental_api.available:
            if primary_error is not None:
                raise primary_error
            return primary

        try:
            supplemental = self.supplemental_api.get_games_by_date(
                fixture_date=date,
                force_refresh=force_refresh,
            )
            primary_games = [item for item in primary.get("response", []) if isinstance(item, dict)]
            for candidate in supplemental:
                if not any(self._same_fixture(candidate, existing) for existing in primary_games):
                    primary_games.append(candidate)
            primary["response"] = primary_games
            primary["results"] = len(primary_games)
            primary["_source"] = "sportmonks" if primary_error is not None else "combined"
        except Exception as error:
            logger.warning("Proveedor complementario de fútbol no disponible: %s", error)
            if primary_error is not None:
                raise primary_error

        return primary

    @classmethod
    def _same_fixture(cls, first: dict, second: dict) -> bool:
        first_fixture = first.get("fixture") or {}
        second_fixture = second.get("fixture") or {}
        first_day = str(first_fixture.get("date") or "")[:10]
        second_day = str(second_fixture.get("date") or "")[:10]
        if not first_day or first_day != second_day:
            return False
        first_teams = first.get("teams") or {}
        second_teams = second.get("teams") or {}
        return all(
            SequenceMatcher(None, cls._normalize_name((first_teams.get(side) or {}).get("name")), cls._normalize_name((second_teams.get(side) or {}).get("name"))).ratio() >= 0.78
            for side in ("home", "away")
        )

    @staticmethod
    def _normalize_name(value: str | None) -> str:
        text = unicodedata.normalize("NFKD", str(value or ""))
        text = "".join(char for char in text if not unicodedata.combining(char)).lower()
        return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())
