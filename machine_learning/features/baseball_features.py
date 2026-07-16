from __future__ import annotations

from typing import Any

import pandas as pd

from services.baseball_data_service import BaseballDataService
from machine_learning.features.elo_features import EloRatings


class BaseballFeatures:
    ROLLING_WINDOWS = (3, 5, 10, 20)
    ROLLING_METRICS = (
        "avg_scored", "avg_allowed", "avg_margin", "win_rate",
        "avg_total", "score_std", "allowed_std", "margin_std",
    )

    def __init__(self) -> None:
        self.data_service = BaseballDataService()

    @staticmethod
    def feature_columns() -> list[str]:
        sides = [
            "avg_scored",
            "avg_allowed",
            "win_rate",
            "avg_total",
            "score_std",
            "avg_scored_last5",
            "avg_allowed_last5",
            "win_rate_last5",
            "avg_margin",
            "avg_margin_last5",
            "scoring_trend",
            "allowed_trend",
            "win_rate_trend",
            "sample_strength",
            "rest_days",
            "back_to_back",
            "games_last_7",
            "games_last_14",
            "venue_home_win_rate",
            "venue_away_win_rate",
            "venue_home_avg_scored",
            "venue_away_avg_scored",
            "venue_home_avg_allowed",
            "venue_away_avg_allowed",
            "avg_opponent_elo",
            "recent_opponent_elo",
            "expected_win_rate",
            "performance_vs_expected",
        ]
        sides.extend(
            f"rolling_{window}_{metric}"
            for window in BaseballFeatures.ROLLING_WINDOWS
            for metric in BaseballFeatures.ROLLING_METRICS
        )
        rolling_differences = [
            f"diff_rolling_{window}_{metric}"
            for window in BaseballFeatures.ROLLING_WINDOWS
            for metric in ("avg_margin", "win_rate")
        ]
        return (
            [f"home_{name}" for name in sides]
            + [f"away_{name}" for name in sides]
            + [
                "diff_avg_scored",
                "diff_avg_allowed",
                "diff_win_rate",
                "diff_avg_total",
                "diff_win_rate_last5",
                "diff_avg_margin",
                "diff_avg_margin_last5",
                "diff_rest_days",
                "diff_games_last_7",
                "diff_games_last_14",
                "diff_venue_win_rate",
                "diff_venue_scoring",
                "diff_schedule_strength",
                "diff_expected_win_rate",
                "diff_performance_vs_expected",
                *rolling_differences,
                "home_advantage",
                "home_elo",
                "away_elo",
                "diff_elo",
                "elo_home_probability",
            ]
        )

    @staticmethod
    def summarize_history(
        history: list[dict[str, Any]],
        reference_date: Any = None,
    ) -> dict[str, float]:
        reference = pd.to_datetime(reference_date, errors="coerce", utc=True)
        if not pd.isna(reference):
            history = [
                item for item in history
                if pd.isna(parsed := pd.to_datetime(item.get("game_date"), errors="coerce", utc=True))
                or parsed < reference
            ]
        if not history:
            history = [{"scored": 4.5, "allowed": 4.5, "won": 0.5}]

        recent = history[-5:]
        scored = [item["scored"] for item in history]
        allowed = [item["allowed"] for item in history]
        mean_score = sum(scored) / len(scored)
        variance = sum((score - mean_score) ** 2 for score in scored) / len(scored)
        avg_allowed = sum(allowed) / len(allowed)
        win_rate = sum(item["won"] for item in history) / len(history)
        recent_scored = sum(item["scored"] for item in recent) / len(recent)
        recent_allowed = sum(item["allowed"] for item in recent) / len(recent)
        recent_win_rate = sum(item["won"] for item in recent) / len(recent)
        summary = {
            "avg_scored": mean_score,
            "avg_allowed": avg_allowed,
            "win_rate": win_rate,
            "avg_total": sum(a + b for a, b in zip(scored, allowed)) / len(history),
            "score_std": max(variance ** 0.5, 1.0),
            "avg_scored_last5": recent_scored,
            "avg_allowed_last5": recent_allowed,
            "win_rate_last5": recent_win_rate,
            "avg_margin": mean_score - avg_allowed,
            "avg_margin_last5": recent_scored - recent_allowed,
            "scoring_trend": recent_scored - mean_score,
            "allowed_trend": recent_allowed - avg_allowed,
            "win_rate_trend": recent_win_rate - win_rate,
            "sample_strength": min(len(history), 20) / 20,
        }
        summary.update(BaseballFeatures._schedule_summary(history, reference_date, summary))
        opponent_elos = [float(item.get("opponent_elo", 1500.0)) for item in history]
        expected_results = [float(item.get("expected_win_probability", 0.5)) for item in history]
        recent_opponents = opponent_elos[-5:]
        expected_win_rate = sum(expected_results) / len(expected_results)
        summary.update(
            {
                "avg_opponent_elo": sum(opponent_elos) / len(opponent_elos),
                "recent_opponent_elo": sum(recent_opponents) / len(recent_opponents),
                "expected_win_rate": expected_win_rate,
                "performance_vs_expected": win_rate - expected_win_rate,
            }
        )
        for window in BaseballFeatures.ROLLING_WINDOWS:
            window_history = history[-window:]
            window_scored = [item["scored"] for item in window_history]
            window_allowed = [item["allowed"] for item in window_history]
            window_margins = [a - b for a, b in zip(window_scored, window_allowed)]
            summary.update(
                {
                    f"rolling_{window}_avg_scored": sum(window_scored) / len(window_scored),
                    f"rolling_{window}_avg_allowed": sum(window_allowed) / len(window_allowed),
                    f"rolling_{window}_avg_margin": sum(window_margins) / len(window_margins),
                    f"rolling_{window}_win_rate": sum(item["won"] for item in window_history) / len(window_history),
                    f"rolling_{window}_avg_total": sum(a + b for a, b in zip(window_scored, window_allowed)) / len(window_history),
                    f"rolling_{window}_score_std": BaseballFeatures._std(window_scored),
                    f"rolling_{window}_allowed_std": BaseballFeatures._std(window_allowed),
                    f"rolling_{window}_margin_std": BaseballFeatures._std(window_margins),
                }
            )
        return summary

    @staticmethod
    def _schedule_summary(
        history: list[dict[str, Any]],
        reference_date: Any,
        fallback: dict[str, float],
    ) -> dict[str, float]:
        dated_history: list[tuple[pd.Timestamp, dict[str, Any]]] = []
        for item in history:
            parsed = pd.to_datetime(item.get("game_date"), errors="coerce", utc=True)
            if not pd.isna(parsed):
                dated_history.append((parsed, item))
        dated_history.sort(key=lambda entry: entry[0])

        reference = pd.to_datetime(reference_date, errors="coerce", utc=True)
        if pd.isna(reference):
            reference = dated_history[-1][0] + pd.Timedelta(days=1) if dated_history else None
        prior = [entry for entry in dated_history if reference is not None and entry[0] < reference]
        if prior:
            elapsed_days = max((reference - prior[-1][0]).total_seconds() / 86400, 0.0)
            rest_days = min(elapsed_days, 30.0)
            games_last_7 = sum(0 < (reference - date).total_seconds() <= 7 * 86400 for date, _ in prior)
            games_last_14 = sum(0 < (reference - date).total_seconds() <= 14 * 86400 for date, _ in prior)
        else:
            rest_days, games_last_7, games_last_14 = 7.0, 0, 0

        def venue_values(is_home: bool) -> tuple[float, float, float]:
            games = [item for _, item in prior if item.get("is_home") is is_home]
            if not games:
                return fallback["win_rate"], fallback["avg_scored"], fallback["avg_allowed"]
            count = len(games)
            return (
                sum(float(item["won"]) for item in games) / count,
                sum(float(item["scored"]) for item in games) / count,
                sum(float(item["allowed"]) for item in games) / count,
            )

        home_win, home_scored, home_allowed = venue_values(True)
        away_win, away_scored, away_allowed = venue_values(False)
        return {
            "rest_days": rest_days,
            "back_to_back": float(rest_days <= 1.5),
            "games_last_7": float(games_last_7),
            "games_last_14": float(games_last_14),
            "venue_home_win_rate": home_win,
            "venue_away_win_rate": away_win,
            "venue_home_avg_scored": home_scored,
            "venue_away_avg_scored": away_scored,
            "venue_home_avg_allowed": home_allowed,
            "venue_away_avg_allowed": away_allowed,
        }

    @staticmethod
    def _std(values: list[float]) -> float:
        mean = sum(values) / len(values)
        return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5

    @classmethod
    def from_summaries(
        cls,
        home: dict[str, float],
        away: dict[str, float],
    ) -> dict[str, float]:
        row = {f"home_{key}": value for key, value in home.items()}
        row.update({f"away_{key}": value for key, value in away.items()})
        row.update(
            {
                "diff_avg_scored": home["avg_scored"] - away["avg_scored"],
                "diff_avg_allowed": home["avg_allowed"] - away["avg_allowed"],
                "diff_win_rate": home["win_rate"] - away["win_rate"],
                "diff_avg_total": home["avg_total"] - away["avg_total"],
                "diff_win_rate_last5": home["win_rate_last5"] - away["win_rate_last5"],
                "diff_avg_margin": home["avg_margin"] - away["avg_margin"],
                "diff_avg_margin_last5": home["avg_margin_last5"] - away["avg_margin_last5"],
                "diff_rest_days": home["rest_days"] - away["rest_days"],
                "diff_games_last_7": home["games_last_7"] - away["games_last_7"],
                "diff_games_last_14": home["games_last_14"] - away["games_last_14"],
                "diff_venue_win_rate": home["venue_home_win_rate"] - away["venue_away_win_rate"],
                "diff_venue_scoring": home["venue_home_avg_scored"] - away["venue_away_avg_scored"],
                "diff_schedule_strength": home["avg_opponent_elo"] - away["avg_opponent_elo"],
                "diff_expected_win_rate": home["expected_win_rate"] - away["expected_win_rate"],
                "diff_performance_vs_expected": home["performance_vs_expected"] - away["performance_vs_expected"],
                "home_advantage": 1.0,
            }
        )
        for window in cls.ROLLING_WINDOWS:
            for metric in ("avg_margin", "win_rate"):
                key = f"rolling_{window}_{metric}"
                row[f"diff_{key}"] = home[key] - away[key]
        row.update(EloRatings().features("home", "away"))
        return row

    def build_live_feature_row(
        self,
        home_team_id: int,
        away_team_id: int,
        season: int | None = None,
        league_id: int | None = None,
        force_refresh: bool = False,
        match_date: Any = None,
        include_elo: bool = False,
    ) -> pd.DataFrame:
        home_games = self.data_service.get_recent_team_games(
            home_team_id, season, league_id, 20, force_refresh
        )
        away_games = self.data_service.get_recent_team_games(
            away_team_id, season, league_id, 20, force_refresh
        )
        elo_features = EloRatings().features(home_team_id, away_team_id)
        game_context: dict[Any, dict[Any, dict[str, float]]] = {}
        if include_elo:
            elo_features, game_context = self.build_live_elo_context(
                home_team_id, away_team_id, season, league_id, match_date, force_refresh
            )
        home = self._summarize_api_games(home_games, home_team_id, match_date, game_context)
        away = self._summarize_api_games(away_games, away_team_id, match_date, game_context)
        row = self.from_summaries(home, away)
        row.update(elo_features)
        return pd.DataFrame([row], columns=self.feature_columns())

    def build_live_elo(
        self, home_team_id: int, away_team_id: int, season: Any,
        league_id: Any, match_date: Any, force_refresh: bool,
    ) -> dict[str, float]:
        features, _ = self.build_live_elo_context(
            home_team_id, away_team_id, season, league_id, match_date, force_refresh
        )
        return features

    def build_live_elo_context(
        self, home_team_id: int, away_team_id: int, season: Any,
        league_id: Any, match_date: Any, force_refresh: bool,
    ) -> tuple[dict[str, float], dict[Any, dict[Any, dict[str, float]]]]:
        if season is None or league_id is None:
            return EloRatings().features(home_team_id, away_team_id), {}
        payload = self.data_service.api.get(
            endpoint="games",
            params={"league": league_id, "season": season},
            cache_key=f"league_{league_id}_season_{season}_games",
            force_refresh=force_refresh,
            max_hours=168,
        )
        games = [game for game in payload.get("response", []) if isinstance(game, dict)]
        games.sort(key=self._game_datetime)
        cutoff = self.data_service._parse_datetime(str(match_date)).timestamp() if match_date else None
        ratings = EloRatings()
        game_context: dict[Any, dict[Any, dict[str, float]]] = {}
        for game in games:
            game_date = self._game_datetime(game)
            if cutoff and game_date.timestamp() >= cutoff:
                break
            if not self.data_service._is_finished_game(game):
                continue
            teams = game.get("teams") or {}
            game_home = (teams.get("home") or {}).get("id")
            game_away = (teams.get("away") or {}).get("id")
            home_score, away_score = self.data_service._extract_scores(game)
            if None not in (game_home, game_away, home_score, away_score):
                pregame = ratings.features(game_home, game_away)
                game_context[self._game_id(game)] = {
                    game_home: {
                        "opponent_elo": pregame["away_elo"],
                        "expected_win_probability": pregame["elo_home_probability"],
                    },
                    game_away: {
                        "opponent_elo": pregame["home_elo"],
                        "expected_win_probability": 1.0 - pregame["elo_home_probability"],
                    },
                }
                ratings.update(game_home, game_away, home_score, away_score)
        return ratings.features(home_team_id, away_team_id), game_context

    def _game_datetime(self, game: dict[str, Any]):
        value = game.get("date") or ((game.get("game") or {}).get("date") or {}).get("date")
        return self.data_service._parse_datetime(value)

    @staticmethod
    def _game_id(game: dict[str, Any]) -> Any:
        return game.get("id") or (game.get("game") or {}).get("id")

    def _summarize_api_games(
        self,
        games: list[dict[str, Any]],
        team_id: int,
        reference_date: Any = None,
        game_context: dict[Any, dict[Any, dict[str, float]]] | None = None,
    ) -> dict[str, float]:
        history: list[dict[str, float]] = []
        for game in reversed(games):
            teams = game.get("teams") or {}
            home_id = (teams.get("home") or {}).get("id")
            away_id = (teams.get("away") or {}).get("id")
            home_score, away_score = self.data_service._extract_scores(game)
            if home_score is None or away_score is None:
                continue
            if team_id == home_id:
                scored, allowed = home_score, away_score
            elif team_id == away_id:
                scored, allowed = away_score, home_score
            else:
                continue
            strength = (game_context or {}).get(self._game_id(game), {}).get(team_id, {})
            history.append(
                {
                    "scored": scored,
                    "allowed": allowed,
                    "won": float(scored > allowed),
                    "game_date": self._game_datetime(game),
                    "is_home": team_id == home_id,
                    "opponent_elo": strength.get("opponent_elo", 1500.0),
                    "expected_win_probability": strength.get("expected_win_probability", 0.5),
                }
            )
        return self.summarize_history(history, reference_date)
