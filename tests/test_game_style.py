from __future__ import annotations

import unittest

from core.game_style import (
    BALANCED,
    CHAOTIC,
    DEFENSIVE,
    OFFENSIVE,
    UNBALANCED,
    apply_game_style_to_markets,
    classify_game_style,
)


def features(goals: float, btts: float, variance: float, points_diff: float = 0.0) -> dict:
    return {
        "home_avg_scored": goals / 2,
        "home_avg_conceded": goals / 2,
        "away_avg_scored": goals / 2,
        "away_avg_conceded": goals / 2,
        "home_btts_rate": btts,
        "away_btts_rate": btts,
        "home_scored_std": variance,
        "away_scored_std": variance,
        "diff_points_last5": points_diff,
    }


QUALITY = {
    "match_quality_score": 0.72,
    "match_quality_components": {"consistency": 0.8},
}


class GameStyleTests(unittest.TestCase):
    def test_defensive(self) -> None:
        self.assertEqual(classify_game_style(features(1.8, 0.35, 0.6), QUALITY)["game_style"], DEFENSIVE)

    def test_offensive(self) -> None:
        self.assertEqual(classify_game_style(features(3.2, 0.70, 0.8), QUALITY)["game_style"], OFFENSIVE)

    def test_balanced(self) -> None:
        self.assertEqual(classify_game_style(features(2.5, 0.52, 0.8), QUALITY)["game_style"], BALANCED)

    def test_chaotic_takes_priority(self) -> None:
        self.assertEqual(classify_game_style(features(3.4, 0.75, 1.6, 9), QUALITY)["game_style"], CHAOTIC)

    def test_unbalanced(self) -> None:
        self.assertEqual(classify_game_style(features(2.5, 0.50, 0.8, 9), QUALITY)["game_style"], UNBALANCED)

    def test_interpretation_changes_but_probability_does_not(self) -> None:
        style = classify_game_style(features(3.4, 0.75, 1.6), QUALITY)
        markets = [{"probability": 64.0, "confidence_score": 70.0, "explanation": "Señal original."}]
        apply_game_style_to_markets(markets, style)
        self.assertEqual(markets[0]["probability"], 64.0)
        self.assertEqual(markets[0]["confidence_score"], 60.0)
        self.assertIn("Partido caótico", markets[0]["explanation"])


if __name__ == "__main__":
    unittest.main()
