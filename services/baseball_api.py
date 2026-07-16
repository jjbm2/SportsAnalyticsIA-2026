from services.base_sports_api import BaseSportsAPI


class BaseballAPI(BaseSportsAPI):
    def __init__(self):
        super().__init__(
            base_url="https://v1.baseball.api-sports.io",
            sport_name="baseball"
        )

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
            "date": date
        }

        cache_key = date

        if league_id is not None:
            params["league"] = league_id
            cache_key += f"_league_{league_id}"

        if season is not None:
            params["season"] = season
            cache_key += f"_season_{season}"

        return self.get(
            endpoint="games",
            params=params,
            cache_key=cache_key,
            force_refresh=force_refresh,
            max_hours=6
        )