from __future__ import annotations

from collections import defaultdict
from typing import Any


class EloRatings:
    def __init__(self, initial: float = 1500.0, k_factor: float = 24.0, home_advantage: float = 55.0) -> None:
        self.initial = initial
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: defaultdict[Any, float] = defaultdict(lambda: self.initial)

    def features(self, home_id: Any, away_id: Any) -> dict[str, float]:
        home = float(self.ratings[home_id])
        away = float(self.ratings[away_id])
        expected = 1.0 / (1.0 + 10 ** ((away - home - self.home_advantage) / 400.0))
        return {
            "home_elo": home,
            "away_elo": away,
            "diff_elo": home - away,
            "elo_home_probability": expected,
        }

    def update(self, home_id: Any, away_id: Any, home_score: float, away_score: float) -> None:
        expected = self.features(home_id, away_id)["elo_home_probability"]
        actual = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        change = self.k_factor * (actual - expected)
        self.ratings[home_id] += change
        self.ratings[away_id] -= change
