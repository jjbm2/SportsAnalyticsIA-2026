import unittest

from core.limited_prediction import LIMITED_WARNING, build_limited_prediction


class LimitedPredictionTests(unittest.TestCase):
    def test_is_symmetric_and_explicitly_low_confidence(self):
        result = build_limited_prediction("Basketball", "Local", "Visitante", 5_000)

        winner_markets = result["markets_to_save"][:2]
        self.assertEqual([market["probability"] for market in winner_markets], [50.0, 50.0])
        self.assertTrue(all(market["confidence"] == "Baja" for market in result["markets_to_save"]))
        self.assertTrue(all(market["risk"] == "Alto" for market in result["markets_to_save"]))
        self.assertTrue(all(market["confidence_score"] <= 20.0 for market in result["markets_to_save"]))
        self.assertIs(result["context_json"]["limited_history"], True)
        self.assertIn("no es 100% confiable", LIMITED_WARNING)

    def test_does_not_invent_cross_sport_markets(self):
        result = build_limited_prediction("MMA", "Peleador A", "Peleador B", 5_000)

        self.assertEqual(
            [market["market_type"] for market in result["markets_to_save"]],
            ["home_win", "away_win"],
        )


if __name__ == "__main__":
    unittest.main()
