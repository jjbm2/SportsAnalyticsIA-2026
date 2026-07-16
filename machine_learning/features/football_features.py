from __future__ import annotations

from typing import Any

from datetime import datetime

import pandas as pd

from services.football_data_service import FootballDataService


class FootballFeatures:
    ROLLING_WINDOWS = (3, 5, 10, 20)
    ROLLING_METRICS = (
        "avg_scored", "avg_conceded", "win_rate", "draw_rate",
        "clean_sheet_rate", "failed_to_score_rate", "avg_total",
        "avg_goal_diff", "goal_diff_std",
    )

    def __init__(self):
        self.data_service = FootballDataService()

    @staticmethod
    def _average(values: list[float], default: float = 0.0) -> float:
        if not values:
            return default
        return sum(values) / len(values)

    @staticmethod
    def _rate(values: list[int], default: float = 0.0) -> float:
        if not values:
            return default
        return sum(values) / len(values)

    def summarize_recent_form(
        self,
        fixtures: list[dict[str, Any]],
        team_id: int,
        last_n: int = 5,
    ) -> dict[str, float]:
        history: list[dict[str, float | str]] = []

        ordered = sorted(fixtures, key=self._fixture_date)
        for fixture in ordered[-last_n:]:
            teams = fixture.get("teams", {})
            goals = fixture.get("goals", {})

            home_team = teams.get("home", {})
            away_team = teams.get("away", {})

            home_id = home_team.get("id")
            away_id = away_team.get("id")

            home_goals = goals.get("home")
            away_goals = goals.get("away")

            if home_goals is None or away_goals is None:
                continue

            if team_id == home_id:
                scored = float(home_goals)
                conceded = float(away_goals)
            elif team_id == away_id:
                scored = float(away_goals)
                conceded = float(home_goals)
            else:
                continue

            history.append({
                "scored": scored,
                "conceded": conceded,
                "result": "W" if scored > conceded else "D" if scored == conceded else "L",
                "is_home": team_id == home_id,
            })
        return self.summarize_history(history)

    @staticmethod
    def _fixture_date(fixture: dict[str, Any]) -> datetime:
        value = str((fixture.get("fixture") or {}).get("date") or "")
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min

    @classmethod
    def summarize_history(cls, history: list[dict[str, Any]]) -> dict[str, float]:
        if not history:
            history = [{"scored": 1.0, "conceded": 1.0, "result": "D", "is_home": None}]
        recent = history[-5:]
        scored_all = [float(item["scored"]) for item in history]
        conceded_all = [float(item["conceded"]) for item in history]
        recent_scored = [float(item["scored"]) for item in recent]
        recent_conceded = [float(item["conceded"]) for item in recent]
        points = [cls._points(item["result"]) for item in history]
        recent_points = [cls._points(item["result"]) for item in recent]
        home_items = [item for item in history if item.get("is_home") is True]
        away_items = [item for item in history if item.get("is_home") is False]
        overall_ppg = cls._average(points, 1.0)
        summary = {
            "matches_played": float(len(history)),
            "recent_sample_size": float(len(recent)),
            "avg_scored_last5": cls._average(recent_scored, 1.0),
            "avg_conceded_last5": cls._average(recent_conceded, 1.0),
            "points_last5": float(sum(recent_points)),
            "points_per_game_last5": cls._average(recent_points, 1.0),
            "goals_scored_last5": float(sum(recent_scored)),
            "goals_conceded_last5": float(sum(recent_conceded)),
            "win_rate_last5": cls._rate([int(item["result"] == "W") for item in recent], 0.33),
            "draw_rate_last5": cls._rate([int(item["result"] == "D") for item in recent], 0.33),
            "clean_sheet_rate_last5": cls._rate([int(float(item["conceded"]) == 0) for item in recent], 0.2),
            "failed_to_score_rate_last5": cls._rate([int(float(item["scored"]) == 0) for item in recent], 0.2),
            "avg_scored": cls._average(scored_all, 1.0),
            "avg_conceded": cls._average(conceded_all, 1.0),
            "clean_sheet_rate": cls._rate([int(value == 0) for value in conceded_all], 0.2),
            "btts_rate": cls._rate([int(a > 0 and b > 0) for a, b in zip(scored_all, conceded_all)], 0.5),
            "btts_rate_last5": cls._rate([int(a > 0 and b > 0) for a, b in zip(recent_scored, recent_conceded)], 0.5),
            "scored_std": cls._std(scored_all),
            "scored_std_last5": cls._std(recent_scored),
            "win_streak": float(cls._streak(history, "W")),
            "loss_streak": float(cls._streak(history, "L")),
            "home_points_per_game": cls._venue_ppg(home_items, overall_ppg),
            "away_points_per_game": cls._venue_ppg(away_items, overall_ppg),
            "home_sample_rate": len(home_items) / len(history),
            "away_sample_rate": len(away_items) / len(history),
        }
        for window in cls.ROLLING_WINDOWS:
            items = history[-window:]
            scored = [float(item["scored"]) for item in items]
            conceded = [float(item["conceded"]) for item in items]
            differences = [a - b for a, b in zip(scored, conceded)]
            prefix = f"rolling_{window}_"
            summary.update({
                prefix + "avg_scored": cls._average(scored),
                prefix + "avg_conceded": cls._average(conceded),
                prefix + "win_rate": cls._rate([int(item["result"] == "W") for item in items]),
                prefix + "draw_rate": cls._rate([int(item["result"] == "D") for item in items]),
                prefix + "clean_sheet_rate": cls._rate([int(value == 0) for value in conceded]),
                prefix + "failed_to_score_rate": cls._rate([int(value == 0) for value in scored]),
                prefix + "avg_total": cls._average([a + b for a, b in zip(scored, conceded)]),
                prefix + "avg_goal_diff": cls._average(differences),
                prefix + "goal_diff_std": cls._std(differences),
            })
        return summary

    @staticmethod
    def _std(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5

    @staticmethod
    def _points(result: Any) -> float:
        return 3.0 if result == "W" else 1.0 if result == "D" else 0.0

    @classmethod
    def _venue_ppg(cls, items: list[dict[str, Any]], fallback: float) -> float:
        return cls._average([cls._points(item["result"]) for item in items], fallback)

    @staticmethod
    def _streak(history: list[dict[str, Any]], result: str) -> int:
        streak = 0
        for item in reversed(history):
            if item["result"] != result:
                break
            streak += 1
        return streak

    def build_live_feature_row(
        self,
        home_team_id: int,
        away_team_id: int,
        force_refresh: bool = False,
        provider: str = "api_sports",
    ) -> pd.DataFrame:
        home_profile = self.data_service.build_team_profile(
            team_id=home_team_id,
            last=20,
            force_refresh=force_refresh,
            provider=provider,
        )
        away_profile = self.data_service.build_team_profile(
            team_id=away_team_id,
            last=20,
            force_refresh=force_refresh,
            provider=provider,
        )

        home_fixtures = self.data_service.get_recent_team_fixtures(
            team_id=home_team_id,
            last=20,
            force_refresh=force_refresh,
            provider=provider,
        )
        away_fixtures = self.data_service.get_recent_team_fixtures(
            team_id=away_team_id,
            last=20,
            force_refresh=force_refresh,
            provider=provider,
        )

        home_form = self.summarize_recent_form(
            fixtures=home_fixtures,
            team_id=home_team_id,
            last_n=20,
        )
        away_form = self.summarize_recent_form(
            fixtures=away_fixtures,
            team_id=away_team_id,
            last_n=20,
        )

        row = {
            "home_avg_scored": home_profile["avg_scored"],
            "home_avg_conceded": home_profile["avg_conceded"],
            "away_avg_scored": away_profile["avg_scored"],
            "away_avg_conceded": away_profile["avg_conceded"],
            "home_avg_scored_last5": home_form["avg_scored_last5"],
            "home_avg_conceded_last5": home_form["avg_conceded_last5"],
            "away_avg_scored_last5": away_form["avg_scored_last5"],
            "away_avg_conceded_last5": away_form["avg_conceded_last5"],
            "home_win_rate_last5": home_form["win_rate_last5"],
            "home_draw_rate_last5": home_form["draw_rate_last5"],
            "away_win_rate_last5": away_form["win_rate_last5"],
            "away_draw_rate_last5": away_form["draw_rate_last5"],
            "home_clean_sheet_rate_last5": home_form["clean_sheet_rate_last5"],
            "away_clean_sheet_rate_last5": away_form["clean_sheet_rate_last5"],
            "home_failed_to_score_rate_last5": home_form["failed_to_score_rate_last5"],
            "away_failed_to_score_rate_last5": away_form["failed_to_score_rate_last5"],
            **self._advanced_matchup_features(home_form, away_form),
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
        row["home_advantage"] = 1.0
        row.update(self._season_context(home_form, away_form))
        for window in self.ROLLING_WINDOWS:
            for metric in self.ROLLING_METRICS:
                key = f"rolling_{window}_{metric}"
                row[f"home_{key}"] = home_form[key]
                row[f"away_{key}"] = away_form[key]
            for metric in ("avg_goal_diff", "win_rate", "draw_rate"):
                key = f"rolling_{window}_{metric}"
                row[f"diff_{key}"] = home_form[key] - away_form[key]

        return pd.DataFrame([row])

    @staticmethod
    def _season_context(home: dict[str, float], away: dict[str, float]) -> dict[str, float]:
        played = min(float(home["matches_played"]), float(away["matches_played"]))
        progress = min(1.0, played / 38.0)
        return {
            "normalized_matchweek": progress,
            "season_phase_early": float(progress < 0.33),
            "season_phase_middle": float(0.33 <= progress < 0.67),
            "season_phase_late": float(progress >= 0.67),
            "matches_played_diff": float(home["matches_played"] - away["matches_played"]),
        }

    @staticmethod
    def _advanced_matchup_features(
        home: dict[str, float], away: dict[str, float]
    ) -> dict[str, float]:
        keys = (
            "matches_played", "recent_sample_size",
            "points_last5", "points_per_game_last5", "goals_scored_last5",
            "goals_conceded_last5", "win_streak", "loss_streak",
            "clean_sheet_rate", "btts_rate", "btts_rate_last5",
            "scored_std", "scored_std_last5", "home_points_per_game",
            "away_points_per_game", "home_sample_rate", "away_sample_rate",
        )
        row = {f"home_{key}": home[key] for key in keys}
        row.update({f"away_{key}": away[key] for key in keys})
        row.update({
            "diff_points_last5": home["points_last5"] - away["points_last5"],
            "diff_btts_rate": home["btts_rate"] - away["btts_rate"],
            "diff_scored_std": home["scored_std"] - away["scored_std"],
            "home_attack_vs_away_defense": home["avg_scored"] - away["avg_conceded"],
            "home_defense_vs_away_attack": home["avg_conceded"] - away["avg_scored"],
            "recent_home_attack_vs_away_defense": home["avg_scored_last5"] - away["avg_conceded_last5"],
            "recent_home_defense_vs_away_attack": home["avg_conceded_last5"] - away["avg_scored_last5"],
            "real_home_advantage": home["home_points_per_game"] - away["away_points_per_game"],
        })
        return row

    @staticmethod
    def feature_columns() -> list[str]:
        columns = [
            "home_avg_scored",
            "home_avg_conceded",
            "away_avg_scored",
            "away_avg_conceded",
            "home_avg_scored_last5",
            "home_avg_conceded_last5",
            "away_avg_scored_last5",
            "away_avg_conceded_last5",
            "home_win_rate_last5",
            "home_draw_rate_last5",
            "away_win_rate_last5",
            "away_draw_rate_last5",
            "home_clean_sheet_rate_last5",
            "away_clean_sheet_rate_last5",
            "home_failed_to_score_rate_last5",
            "away_failed_to_score_rate_last5",
            "diff_avg_scored",
            "diff_avg_conceded",
            "diff_win_rate_last5",
            "diff_clean_sheet_rate_last5",
            "diff_failed_to_score_rate_last5",
            "home_advantage",
            "normalized_matchweek", "season_phase_early", "season_phase_middle",
            "season_phase_late", "matches_played_diff",
        ]
        advanced = (
            "points_last5", "points_per_game_last5", "goals_scored_last5",
            "goals_conceded_last5", "win_streak", "loss_streak",
            "clean_sheet_rate", "btts_rate", "btts_rate_last5",
            "scored_std", "scored_std_last5", "home_points_per_game",
            "away_points_per_game", "home_sample_rate", "away_sample_rate",
        )
        columns.extend(f"{side}_{key}" for side in ("home", "away") for key in advanced)
        columns.extend([
            "diff_points_last5", "diff_btts_rate", "diff_scored_std",
            "home_attack_vs_away_defense", "home_defense_vs_away_attack",
            "recent_home_attack_vs_away_defense", "recent_home_defense_vs_away_attack",
            "real_home_advantage",
        ])
        columns.extend(
            f"{side}_rolling_{window}_{metric}"
            for side in ("home", "away")
            for window in FootballFeatures.ROLLING_WINDOWS
            for metric in FootballFeatures.ROLLING_METRICS
        )
        columns.extend(
            f"diff_rolling_{window}_{metric}"
            for window in FootballFeatures.ROLLING_WINDOWS
            for metric in ("avg_goal_diff", "win_rate", "draw_rate")
        )
        return columns
