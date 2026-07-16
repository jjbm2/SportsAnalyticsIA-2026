from machine_learning.features.baseball_features import BaseballFeatures
from services.basketball_data_service import BasketballDataService


class BasketballFeatures(BaseballFeatures):
    """Features de forma genéricas adaptadas a puntos de basketball."""

    def __init__(self) -> None:
        self.data_service = BasketballDataService()

    @staticmethod
    def summarize_history(history, reference_date=None):
        if not history:
            history = [
                {"scored": 108.0, "allowed": 108.0, "won": 0.5}
            ]
        return BaseballFeatures.summarize_history(history, reference_date)
