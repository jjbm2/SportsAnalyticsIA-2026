from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from machine_learning.features.nfl_features import NFLFeatures
from machine_learning.features.elo_features import EloRatings
from services.nfl_api import NFLAPI
from services.nfl_data_service import NFLDataService


class NFLDatasetBuilder:
    def __init__(self) -> None:
        self.api = NFLAPI()
        self.data_service = NFLDataService()

    def fetch_season_games(self, season: int, league_id: int = 1) -> list[dict[str, Any]]:
        payload = self.api.get(
            endpoint="games",
            params={"league": league_id, "season": season},
            cache_key=f"league_{league_id}_season_{season}_games",
            max_hours=168,
        )
        games = [item for item in payload.get("response", []) if isinstance(item, dict)]
        games.sort(key=self._game_date)
        return games

    def build_dataset(
        self,
        seasons: tuple[int, ...] = (2022, 2023, 2024),
        league_id: int = 1,
        min_history: int = 5,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for season in seasons:
            histories: defaultdict[int, list[dict[str, float]]] = defaultdict(list)
            elo = EloRatings()
            for game in self.fetch_season_games(season, league_id):
                if not self.data_service._is_finished_game(game):
                    continue
                teams = game.get("teams") or {}
                home_id = (teams.get("home") or {}).get("id")
                away_id = (teams.get("away") or {}).get("id")
                home, away = self.data_service._extract_scores(game)
                if None in (home_id, away_id, home, away):
                    continue
                pregame_elo = elo.features(home_id, away_id)
                if len(histories[home_id]) >= min_history and len(histories[away_id]) >= min_history:
                    game_date = self._game_date(game)
                    features = NFLFeatures.from_summaries(
                        NFLFeatures.summarize_history(histories[home_id], game_date),
                        NFLFeatures.summarize_history(histories[away_id], game_date),
                    )
                    features.update(elo.features(home_id, away_id))
                    rows.append({
                        "game_date": self._game_date(game).isoformat(),
                        "league_id": league_id,
                        "season": season,
                        "game_id": (game.get("game") or {}).get("id"),
                        "home_team_id": home_id,
                        "away_team_id": away_id,
                        "home_score": home,
                        "away_score": away,
                        "home_win_label": int(home > away),
                        "over_415_label": int(home + away > 41.5),
                        "home_over_205_label": int(home > 20.5),
                        **features,
                    })
                histories[home_id].append({"scored": home, "allowed": away, "won": float(home > away), "game_date": self._game_date(game), "is_home": True, "opponent_elo": pregame_elo["away_elo"], "expected_win_probability": pregame_elo["elo_home_probability"]})
                histories[away_id].append({"scored": away, "allowed": home, "won": float(away > home), "game_date": self._game_date(game), "is_home": False, "opponent_elo": pregame_elo["home_elo"], "expected_win_probability": 1.0 - pregame_elo["elo_home_probability"]})
                elo.update(home_id, away_id, home, away)
        dataset = pd.DataFrame(rows)
        if save_csv and not dataset.empty:
            path = Path("data/ml_datasets/nfl_2022_2024.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(path, index=False)
        return dataset

    @staticmethod
    def _game_date(game: dict[str, Any]) -> datetime:
        value = ((game.get("game") or {}).get("date") or {}).get("date")
        try:
            return datetime.fromisoformat(value or "")
        except ValueError:
            return datetime.min
