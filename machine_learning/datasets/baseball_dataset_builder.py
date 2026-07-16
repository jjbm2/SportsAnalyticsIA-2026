from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from machine_learning.features.baseball_features import BaseballFeatures
from machine_learning.features.elo_features import EloRatings
from services.baseball_api import BaseballAPI
from services.baseball_data_service import BaseballDataService


class BaseballDatasetBuilder:
    def __init__(self) -> None:
        self.api = BaseballAPI()
        self.data_service = BaseballDataService()

    def fetch_season_games(
        self,
        league_id: int,
        season: int,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        payload = self.api.get(
            endpoint="games",
            params={"league": league_id, "season": season},
            cache_key=f"league_{league_id}_season_{season}_games",
            force_refresh=force_refresh,
            max_hours=168,
        )
        games = [item for item in payload.get("response", []) if isinstance(item, dict)]
        games.sort(key=lambda item: self._parse_datetime(item.get("date")))
        return games

    def build_dataset(
        self,
        league_id: int = 1,
        seasons: tuple[int, ...] = (2023, 2024, 2025),
        min_history: int = 8,
        force_refresh: bool = False,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for season in seasons:
            histories: dict[int, list[dict[str, float]]] = defaultdict(list)
            elo = EloRatings()
            for game in self.fetch_season_games(league_id, season, force_refresh):
                if not self.data_service._is_finished_game(game):
                    continue
                teams = game.get("teams") or {}
                home_id = (teams.get("home") or {}).get("id")
                away_id = (teams.get("away") or {}).get("id")
                home_score, away_score = self.data_service._extract_scores(game)
                if None in (home_id, away_id, home_score, away_score):
                    continue
                pregame_elo = elo.features(home_id, away_id)

                if len(histories[home_id]) >= min_history and len(histories[away_id]) >= min_history:
                    game_date = game.get("date")
                    features = BaseballFeatures.from_summaries(
                        BaseballFeatures.summarize_history(histories[home_id], game_date),
                        BaseballFeatures.summarize_history(histories[away_id], game_date),
                    )
                    features.update(elo.features(home_id, away_id))
                    rows.append(
                        {
                            "game_date": game.get("date"),
                            "league_id": league_id,
                            "season": season,
                            "game_id": game.get("id"),
                            "home_team_id": home_id,
                            "away_team_id": away_id,
                            "home_score": home_score,
                            "away_score": away_score,
                            "home_win_label": int(home_score > away_score),
                            "over_85_label": int(home_score + away_score > 8.5),
                            "home_over_35_label": int(home_score > 3.5),
                            **features,
                        }
                    )

                histories[home_id].append(
                    {"scored": home_score, "allowed": away_score, "won": float(home_score > away_score), "game_date": game.get("date"), "is_home": True, "opponent_elo": pregame_elo["away_elo"], "expected_win_probability": pregame_elo["elo_home_probability"]}
                )
                histories[away_id].append(
                    {"scored": away_score, "allowed": home_score, "won": float(away_score > home_score), "game_date": game.get("date"), "is_home": False, "opponent_elo": pregame_elo["home_elo"], "expected_win_probability": 1.0 - pregame_elo["elo_home_probability"]}
                )
                elo.update(home_id, away_id, home_score, away_score)

        dataset = pd.DataFrame(rows)
        if save_csv and not dataset.empty:
            output_dir = Path("data/ml_datasets")
            output_dir.mkdir(parents=True, exist_ok=True)
            seasons_label = "_".join(str(season) for season in seasons)
            dataset.to_csv(
                output_dir / f"baseball_mlb_seasons_{seasons_label}.csv",
                index=False,
            )
        return dataset

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.min
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min
