from __future__ import annotations

from datetime import datetime
from typing import Any

from services.nfl_api import NFLAPI
from services.historical_season import accessible_history_season


FINAL_STATUSES = {
    "FT",
    "AOT",
    "POST",
    "FINAL",
    "ENDED",
    "FINISHED",
    "COMPLETED",
}


class NFLDataService:
    def __init__(self):
        self.api = NFLAPI()

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.min
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return datetime.min

    @staticmethod
    def _is_finished_game(game: dict[str, Any]) -> bool:
        status = game.get("status") or (game.get("game") or {}).get("status") or {}
        if isinstance(status, dict):
            long_status = str(status.get("long", "")).upper()
            short_status = str(status.get("short", "")).upper()
            return (
                short_status in FINAL_STATUSES
                or long_status in FINAL_STATUSES
                or "FINAL" in long_status
                or "FINISHED" in long_status
                or "COMPLETED" in long_status
            )
        return False

    @staticmethod
    def _extract_scores(game: dict[str, Any]) -> tuple[float | None, float | None]:
        scores = game.get("scores") or {}
        teams = game.get("teams") or {}

        candidates_home = [
            ((scores.get("home") or {}).get("total") if isinstance(scores.get("home"), dict) else None),
            ((teams.get("home") or {}).get("score") if isinstance(teams.get("home"), dict) else None),
            scores.get("home"),
        ]
        candidates_away = [
            ((scores.get("away") or {}).get("total") if isinstance(scores.get("away"), dict) else None),
            ((teams.get("away") or {}).get("score") if isinstance(teams.get("away"), dict) else None),
            scores.get("away"),
        ]

        home_score = None
        away_score = None

        for value in candidates_home:
            if value is not None:
                try:
                    home_score = float(value)
                    break
                except (TypeError, ValueError):
                    continue

        for value in candidates_away:
            if value is not None:
                try:
                    away_score = float(value)
                    break
                except (TypeError, ValueError):
                    continue

        return home_score, away_score

    def get_recent_team_games(
        self,
        team_id: int,
        season: int | None = None,
        league_id: int | None = None,
        last: int = 10,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        history_season = accessible_history_season("nfl", season)
        params: dict[str, Any] = {"team": team_id}

        params["season"] = history_season
        if league_id is not None:
            params["league"] = league_id

        data = self.api.get(
            endpoint="games",
            params=params,
            cache_key=f"team_{team_id}_season_{history_season}_league_{league_id}_last_{last}",
            force_refresh=force_refresh,
            max_hours=12,
        )

        games = data.get("response", [])
        filtered = [game for game in games if isinstance(game, dict) and self._is_finished_game(game)]
        filtered.sort(
            key=lambda game: self._parse_datetime(
                game.get("date")
                or ((game.get("game") or {}).get("date") or {}).get("date")
            ),
            reverse=True,
        )

        return filtered[:last]

    def build_team_profile(
        self,
        team_id: int,
        season: int | None = None,
        league_id: int | None = None,
        last: int = 10,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        games = self.get_recent_team_games(
            team_id=team_id,
            season=season,
            league_id=league_id,
            last=last,
            force_refresh=force_refresh,
        )
        history_season = accessible_history_season("nfl", season)

        played = 0
        points_for = 0.0
        points_allowed = 0.0
        wins = 0
        team_scores: list[float] = []
        totals: list[float] = []

        for game in games:
            teams = game.get("teams") or {}
            home_team = teams.get("home") or {}
            away_team = teams.get("away") or {}

            home_id = home_team.get("id")
            away_id = away_team.get("id")

            home_score, away_score = self._extract_scores(game)
            if home_score is None or away_score is None:
                continue

            if team_id == home_id:
                team_score = home_score
                opp_score = away_score
            elif team_id == away_id:
                team_score = away_score
                opp_score = home_score
            else:
                continue

            played += 1
            points_for += team_score
            points_allowed += opp_score
            team_scores.append(team_score)
            totals.append(team_score + opp_score)

            if team_score > opp_score:
                wins += 1

        if played == 0:
            return {
                "team_id": team_id,
                "played": 0,
                "avg_scored": 22.5,
                "avg_allowed": 22.5,
                "win_rate": 0.5,
                "avg_total": 45.0,
                "score_std": 8.0,
                "history_season": history_season,
                "history_is_stale": history_season != self._season_year(season),
            }

        mean_score = points_for / played
        variance = sum((score - mean_score) ** 2 for score in team_scores) / max(1, played)
        score_std = variance ** 0.5

        return {
            "team_id": team_id,
            "played": played,
            "avg_scored": points_for / played,
            "avg_allowed": points_allowed / played,
            "win_rate": wins / played,
            "avg_total": sum(totals) / played,
            "score_std": score_std if score_std > 0 else 8.0,
            "history_season": history_season,
            "history_is_stale": history_season != self._season_year(season),
        }

    @staticmethod
    def _season_year(season: Any) -> int:
        try:
            return int(str(season).split("-", 1)[0])
        except (TypeError, ValueError):
            return datetime.now().year
