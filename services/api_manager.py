from services.baseball_api import BaseballAPI
from services.basketball_api import BasketballAPI
from services.football_api import FootballAPI
from services.formula1_api import Formula1API
from services.hockey_api import HockeyAPI
from services.mma_api import MMAAPI
from services.nfl_api import NFLAPI


class APIManager:
    API_CLASSES = {
        "Fútbol": FootballAPI,
        "Béisbol": BaseballAPI,
        "Basketball": BasketballAPI,
        "NFL": NFLAPI,
        "Fórmula 1": Formula1API,
        "Hockey": HockeyAPI,
        "MMA": MMAAPI,
    }

    def __init__(self, sport: str):
        api_class = self.API_CLASSES.get(sport)

        if api_class is None:
            raise ValueError(
                f"El deporte '{sport}' no está soportado."
            )

        self.sport = sport
        self.api = api_class()

    def get_competitions(
        self,
        force_refresh: bool = False
    ):
        return self.api.get_leagues(
            force_refresh=force_refresh
        )

    def get_games_by_date(
        self,
        date: str,
        league_id: int | None = None,
        season: int | None = None,
        force_refresh: bool = False
    ):
        return self.api.get_games_by_date(
            date=date,
            league_id=league_id,
            season=season,
            force_refresh=force_refresh
        )
