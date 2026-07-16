from __future__ import annotations

import unittest

from core.match_quality import calculate_match_quality


def market(probability: float, confidence: float, agreement: float) -> dict:
    return {
        "market_type": "home_win",
        "probability": probability,
        "confidence_score": confidence,
        "extra_data_json": {"confidence_components": {"model_agreement": agreement}},
    }


class MatchQualityTests(unittest.TestCase):
    def test_high_quality_match_is_recommended(self) -> None:
        result = calculate_match_quality(
            [market(78, 88, 92), market(70, 82, 89)],
            {
                "home_matches_played": 30,
                "away_matches_played": 28,
                "home_scored_std": 0.45,
                "away_scored_std": 0.55,
            },
            quality_gate={"result": True, "over_2_5": True},
        )
        self.assertGreaterEqual(result["match_quality_score"], 0.75)
        self.assertEqual(result["match_quality_label"], "Partido recomendado")

    def test_low_data_and_variance_reduce_quality(self) -> None:
        result = calculate_match_quality(
            [market(52, 35, 40)],
            {
                "home_matches_played": 2,
                "away_matches_played": 3,
                "home_scored_std": 2.8,
                "away_scored_std": 2.4,
            },
            quality_gate={"result": True},
        )
        self.assertLess(result["match_quality_score"], 0.60)
        self.assertEqual(result["match_quality_label"], "Calidad baja")

    def test_fallback_is_supported_and_cannot_claim_high_quality(self) -> None:
        result = calculate_match_quality(
            [market(80, 85, 65)],
            {
                "home_matches_played": 25,
                "away_matches_played": 25,
                "home_scored_std": 0.3,
                "away_scored_std": 0.3,
            },
            quality_gate={"result": False, "over_2_5": False, "btts": False},
        )
        self.assertLess(result["match_quality_score"], 0.75)


if __name__ == "__main__":
    unittest.main()
