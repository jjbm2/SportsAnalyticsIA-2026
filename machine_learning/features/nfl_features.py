from machine_learning.features.baseball_features import BaseballFeatures
from services.nfl_data_service import NFLDataService


class NFLFeatures(BaseballFeatures):
    """Variables de forma reciente adaptadas al puntaje de NFL."""

    def __init__(self) -> None:
        self.data_service = NFLDataService()

    @staticmethod
    def summarize_history(history, reference_date=None):
        if not history:
            history = [{"scored": 22.5, "allowed": 22.5, "won": 0.5}]
        return BaseballFeatures.summarize_history(history, reference_date)
