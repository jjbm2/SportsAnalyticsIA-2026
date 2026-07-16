from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from machine_learning.features.basketball_features import BasketballFeatures
from machine_learning.features.elo_features import EloRatings
from services.basketball_api import BasketballAPI
from services.basketball_data_service import BasketballDataService


class BasketballDatasetBuilder:
    def __init__(self) -> None:
        self.api = BasketballAPI()
        self.data_service = BasketballDataService()

    def fetch_season_games(self, league_id: int, season: str) -> list[dict[str, Any]]:
        payload = self.api.get(
            endpoint="games",
            params={"league": league_id, "season": season},
            cache_key=f"league_{league_id}_season_{season}_games",
            max_hours=168,
        )
        games = [game for game in payload.get("response", []) if isinstance(game, dict)]
        games.sort(key=lambda game: self._date(game.get("date")))
        return games

    def build_dataset(
        self,
        league_id: int = 12,
        seasons: tuple[str, ...] = ("2022-2023", "2023-2024", "2024-2025"),
        min_history: int = 8,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        rows = []
        for season in seasons:
            histories = defaultdict(list)
            elo = EloRatings()
            for game in self.fetch_season_games(league_id, season):
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
                    game_date = game.get("date")
                    features = BasketballFeatures.from_summaries(
                        BasketballFeatures.summarize_history(histories[home_id], game_date),
                        BasketballFeatures.summarize_history(histories[away_id], game_date),
                    )
                    features.update(elo.features(home_id, away_id))
                    rows.append({
                        "game_date": game.get("date"), "league_id": league_id,
                        "season": season, "game_id": game.get("id"),
                        "home_team_id": home_id, "away_team_id": away_id,
                        "home_score": home, "away_score": away,
                        "home_win_label": int(home > away),
                        "over_2195_label": int(home + away > 219.5),
                        "home_over_1095_label": int(home > 109.5), **features,
                    })
                histories[home_id].append({"scored": home, "allowed": away, "won": float(home > away), "game_date": game.get("date"), "is_home": True, "opponent_elo": pregame_elo["away_elo"], "expected_win_probability": pregame_elo["elo_home_probability"]})
                histories[away_id].append({"scored": away, "allowed": home, "won": float(away > home), "game_date": game.get("date"), "is_home": False, "opponent_elo": pregame_elo["home_elo"], "expected_win_probability": 1.0 - pregame_elo["elo_home_probability"]})
                elo.update(home_id, away_id, home, away)
        dataset = pd.DataFrame(rows)
        if save_csv and not dataset.empty:
            path = Path("data/ml_datasets/basketball_nba_2022_2025.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(path, index=False)
        return dataset

    @staticmethod
    def _date(value: str | None) -> datetime:
        try:
            return datetime.fromisoformat((value or "").replace("Z", "+00:00"))
        except ValueError:
            return datetime.min
