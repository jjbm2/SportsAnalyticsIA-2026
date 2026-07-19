from difflib import SequenceMatcher
import re
import unicodedata

from core.logger import logger
from core.event_time import sports_timezone
from core.event_cache_policy import event_cache_hours
from services.base_sports_api import BaseSportsAPI
from services.sportmonks_football_api import SportmonksFootballAPI
from services.sportsdata_soccer_api import SportsDataSoccerAPI


class FootballAPI(BaseSportsAPI):
    def __init__(self):
        super().__init__(
            base_url="https://v3.football.api-sports.io",
            sport_name="football",
            require_api_key=False,
        )
        self.supplemental_api = SportmonksFootballAPI()
        self.sportsdata_api = SportsDataSoccerAPI()

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

        # Provider priority: SportsDataIO, shared combined cache, API-Sports,
        # then SportMonks. Missing credentials remove that provider safely.
        games: list[dict] = []
        successful_providers: list[str] = []
        provider_warnings: list[dict[str, str]] = []
        combined_cache_file = self._cache_path("fixtures_combined", cache_key)

        if self.sportsdata_api.available:
            try:
                provider_games = self.sportsdata_api.get_games_by_date(
                    fixture_date=date,
                    force_refresh=force_refresh,
                )
                for candidate in provider_games:
                    if not any(self._same_fixture(candidate, existing) for existing in games):
                        games.append(candidate)
                successful_providers.append("sportsdataio")
            except Exception as error:
                logger.warning(
                    "Proveedor de fútbol %s no disponible (%s)",
                    "sportsdataio",
                    type(error).__name__,
                )
                provider_warnings.append({
                    "provider": "sportsdataio",
                    "reason": "provider_error",
                })

        if not force_refresh:
            combined_cache = self._read_cache(
                combined_cache_file,
                event_cache_hours(date),
            )
            if combined_cache is not None:
                for candidate in combined_cache.get("response", []):
                    if isinstance(candidate, dict) and not any(
                        self._same_fixture(candidate, existing) for existing in games
                    ):
                        games.append(candidate)
                return {
                    "response": games,
                    "results": len(games),
                    "_source": "combined_cache",
                    "_provider_warnings": provider_warnings,
                }

        if self.api_key:
            try:
                api_sports = self.get(
                    endpoint="fixtures",
                    params=params,
                    cache_key=cache_key,
                    force_refresh=force_refresh,
                    max_hours=event_cache_hours(date),
                )
                for candidate in api_sports.get("response", []):
                    if isinstance(candidate, dict) and not any(
                        self._same_fixture(candidate, existing) for existing in games
                    ):
                        games.append(candidate)
                successful_providers.append("api_sports")
            except Exception as error:
                logger.warning("API-Sports de fútbol no disponible (%s)", type(error).__name__)
                provider_warnings.append({
                    "provider": "api_sports",
                    "reason": getattr(error, "reason", "provider_error"),
                })

        if self.supplemental_api.available:
            try:
                provider_games = self.supplemental_api.get_games_by_date(
                    fixture_date=date,
                    force_refresh=force_refresh,
                )
                for candidate in provider_games:
                    if not any(self._same_fixture(candidate, existing) for existing in games):
                        games.append(candidate)
                successful_providers.append("sportmonks")
            except Exception as error:
                logger.warning(
                    "Proveedor de fútbol sportmonks no disponible (%s)",
                    type(error).__name__,
                )
                provider_warnings.append({
                    "provider": "sportmonks",
                    "reason": "provider_error",
                })

        source = (
            "combined" if len(successful_providers) > 1
            else successful_providers[0] if successful_providers
            else "unavailable"
        )
        result = {
            "response": games,
            "results": len(games),
            "_source": source,
            "_provider_warnings": provider_warnings,
        }
        if games:
            self._save_cache(combined_cache_file, result)
        return result

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
