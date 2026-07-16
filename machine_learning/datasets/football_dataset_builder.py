from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from machine_learning.features.football_features import FootballFeatures
from services.football_api import FootballAPI


class FootballDatasetBuilder:
    def __init__(self):
        self.api = FootballAPI()

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.min
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min

    @staticmethod
    def _is_finished_fixture(fixture: dict[str, Any]) -> bool:
        status = (fixture.get("fixture") or {}).get("status", {})
        short_status = str(status.get("short", "")).upper()
        long_status = str(status.get("long", "")).upper()

        final_statuses = {"FT", "AET", "PEN", "FINISHED", "MATCH FINISHED"}
        return (
            short_status in final_statuses
            or long_status in final_statuses
            or "FINISHED" in long_status
        )

    @staticmethod
    def _fixture_teams_and_goals(fixture: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})

        home_id = (teams.get("home") or {}).get("id")
        away_id = (teams.get("away") or {}).get("id")
        home_goals = goals.get("home")
        away_goals = goals.get("away")

        return home_id, away_id, home_goals, away_goals

    @staticmethod
    def _history_summary(history: list[dict[str, Any]]) -> dict[str, float]:
        return FootballFeatures.summarize_history(history)

    def fetch_league_fixtures(
        self,
        league_id: int,
        season: int,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        data = self.api.get(
            endpoint="fixtures",
            params={
                "league": league_id,
                "season": season,
            },
            cache_key=f"league_{league_id}_season_{season}_fixtures",
            force_refresh=force_refresh,
            max_hours=24,
        )

        fixtures = [fixture for fixture in data.get("response", []) if isinstance(fixture, dict)]
        fixtures.sort(key=lambda item: self._parse_datetime((item.get("fixture") or {}).get("date")))
        return fixtures

    def build_dataset(
        self,
        league_id: int,
        season: int,
        min_history: int = 5,
        force_refresh: bool = False,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        fixtures = self.fetch_league_fixtures(
            league_id=league_id,
            season=season,
            force_refresh=force_refresh,
        )

        histories: dict[int, list[dict[str, Any]]] = defaultdict(list)
        rows: list[dict[str, Any]] = []

        for fixture in fixtures:
            if not self._is_finished_fixture(fixture):
                continue

            home_id, away_id, home_goals, away_goals = self._fixture_teams_and_goals(fixture)
            if None in (home_id, away_id, home_goals, away_goals):
                continue

            if len(histories[home_id]) >= min_history and len(histories[away_id]) >= min_history:
                home_summary = self._history_summary(histories[home_id])
                away_summary = self._history_summary(histories[away_id])

                row = {
                    "game_date": (fixture.get("fixture") or {}).get("date"),
                    "league_id": league_id,
                    "season": season,
                    "fixture_id": (fixture.get("fixture") or {}).get("id"),
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result_label": 1 if home_goals > away_goals else 0 if home_goals == away_goals else -1,
                    "over_25_label": 1 if (home_goals + away_goals) > 2.5 else 0,
                    "btts_label": 1 if home_goals > 0 and away_goals > 0 else 0,
                    "home_avg_scored": home_summary["avg_scored"],
                    "home_avg_conceded": home_summary["avg_conceded"],
                    "away_avg_scored": away_summary["avg_scored"],
                    "away_avg_conceded": away_summary["avg_conceded"],
                    "home_avg_scored_last5": home_summary["avg_scored_last5"],
                    "home_avg_conceded_last5": home_summary["avg_conceded_last5"],
                    "away_avg_scored_last5": away_summary["avg_scored_last5"],
                    "away_avg_conceded_last5": away_summary["avg_conceded_last5"],
                    "home_win_rate_last5": home_summary["win_rate_last5"],
                    "home_draw_rate_last5": home_summary["draw_rate_last5"],
                    "away_win_rate_last5": away_summary["win_rate_last5"],
                    "away_draw_rate_last5": away_summary["draw_rate_last5"],
                    "home_clean_sheet_rate_last5": home_summary["clean_sheet_rate_last5"],
                    "away_clean_sheet_rate_last5": away_summary["clean_sheet_rate_last5"],
                    "home_failed_to_score_rate_last5": home_summary["failed_to_score_rate_last5"],
                    "away_failed_to_score_rate_last5": away_summary["failed_to_score_rate_last5"],
                    "home_advantage": 1.0,
                    **FootballFeatures._advanced_matchup_features(
                        home_summary, away_summary
                    ),
                    **FootballFeatures._season_context(home_summary, away_summary),
                }

                row["diff_avg_scored"] = row["home_avg_scored"] - row["away_avg_scored"]
                row["diff_avg_conceded"] = row["home_avg_conceded"] - row["away_avg_conceded"]
                row["diff_win_rate_last5"] = row["home_win_rate_last5"] - row["away_win_rate_last5"]
                row["diff_clean_sheet_rate_last5"] = (
                    row["home_clean_sheet_rate_last5"] - row["away_clean_sheet_rate_last5"]
                )
                row["diff_failed_to_score_rate_last5"] = (
                    row["home_failed_to_score_rate_last5"] - row["away_failed_to_score_rate_last5"]
                )
                for window in FootballFeatures.ROLLING_WINDOWS:
                    for metric in FootballFeatures.ROLLING_METRICS:
                        key = f"rolling_{window}_{metric}"
                        row[f"home_{key}"] = home_summary[key]
                        row[f"away_{key}"] = away_summary[key]
                    for metric in ("avg_goal_diff", "win_rate", "draw_rate"):
                        key = f"rolling_{window}_{metric}"
                        row[f"diff_{key}"] = home_summary[key] - away_summary[key]

                rows.append(row)

            histories[home_id].append(
                {
                    "scored": float(home_goals),
                    "conceded": float(away_goals),
                    "result": "W" if home_goals > away_goals else "D" if home_goals == away_goals else "L",
                    "is_home": True,
                }
            )
            histories[away_id].append(
                {
                    "scored": float(away_goals),
                    "conceded": float(home_goals),
                    "result": "W" if away_goals > home_goals else "D" if away_goals == home_goals else "L",
                    "is_home": False,
                }
            )

        df = pd.DataFrame(rows)

        if save_csv and not df.empty:
            output_dir = Path("data/ml_datasets")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"football_league_{league_id}_season_{season}.csv"
            df.to_csv(output_path, index=False)

        return df
