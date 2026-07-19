from __future__ import annotations

import ast
import sqlite3
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from core.analysis_transparency import classify_analysis
from core.constants import DEFAULT_SIMULATIONS, MIN_SIMULATIONS
from core.game_status import extract_final_score, is_finished_status
from core.event_cache_policy import event_cache_hours
from core.paths import DATABASE_PATH
from database.model_metrics_repository import ModelMetricsRepository
from database.database import _database_url, _engine_options
from engines.formula1_prediction_engine import Formula1PredictionEngine
from engines.football_prediction_engine import FootballPredictionEngine
from engines.specialty_prediction_engines import (
    HockeyPredictionEngine,
    MMAPredictionEngine,
)
from machine_learning.evaluation.post_match_evaluator import evaluate_markets
from machine_learning.evaluation.model_error_analyzer import build_error_rows, detect_failure_patterns
from machine_learning.model_promotion import evaluate_promotion
from machine_learning.model_registry import ModelRegistry
from machine_learning.features.baseball_features import BaseballFeatures
from machine_learning.features.elo_features import EloRatings
from machine_learning.features.formula1_features import Formula1Features
from machine_learning.model_quality import (
    expected_calibration_error,
    multiclass_probability_metrics,
    validated_ml_weight,
)
from machine_learning.calibration import FootballProbabilityCalibrator
from machine_learning.shadow_validation import ShadowValidationService
from machine_learning.continuous_learning import ContinuousLearningService
from machine_learning.probability_ensemble import ProbabilityEnsemble
from machine_learning.backtesting import WalkForwardBacktester
from services.football_api import FootballAPI
from services.football_data_service import FootballDataService
from services.player_availability_service import PlayerAvailabilityService
from services.post_match_service import PostMatchService
from services.sportmonks_football_api import SportmonksFootballAPI
from services.sportsdata_soccer_api import SportsDataSoccerAPI
from services.http_client import TRANSIENT_STATUS_CODES, build_retry_session
from services.base_sports_api import BaseSportsAPI, ProviderResponseError, classify_provider_error
from machine_learning.predictors.baseball_predictor import BaseballPredictor
from machine_learning.predictors.basketball_predictor import BasketballPredictor
from services.provider_health import check_sports_connectivity
from core.league_filters import filter_games_by_league_view
from core.market_risk import apply_probability_risk, probability_risk_profile
from core.market_visibility import visible_markets
from core.prediction_confidence import enrich_football_markets


class _Formula1Results:
    @staticmethod
    def get_results(season: int, force_refresh: bool = False) -> list[dict]:
        del season, force_refresh
        drivers = [
            ("driver-a", "Ana", "Uno", "Equipo A", 1),
            ("driver-b", "Beto", "Dos", "Equipo B", 2),
            ("driver-c", "Caro", "Tres", "Equipo C", 3),
        ]
        return [
            {
                "round": str(round_number),
                "Results": [
                    {
                        "position": str(position),
                        "Driver": {
                            "driverId": driver_id,
                            "givenName": given_name,
                            "familyName": family_name,
                        },
                        "Constructor": {"name": constructor},
                    }
                    for driver_id, given_name, family_name, constructor, position in drivers
                ],
            }
            for round_number in range(1, 5)
        ]


class RegressionTests(unittest.TestCase):
    def test_live_predictors_cap_history_to_validated_season(self) -> None:
        self.assertEqual(BaseballPredictor._history_season(2026, 2024), 2024)
        self.assertEqual(BaseballPredictor._history_season(2023, 2024), 2023)
        self.assertEqual(
            BasketballPredictor._history_season("2025-2026", "2024-2025"),
            "2024-2025",
        )
        self.assertEqual(
            BasketballPredictor._history_season("2023-2024", "2024-2025"),
            "2023-2024",
        )
    def test_event_cache_policy_keeps_today_fresh_and_future_weekly(self) -> None:
        today = date(2026, 7, 16)
        self.assertEqual(event_cache_hours(today, today=today), 0.25)
        self.assertEqual(event_cache_hours(today + timedelta(days=1), today=today), 168)
        self.assertEqual(event_cache_hours(today - timedelta(days=1), today=today), 336)
    def test_postgres_pool_is_bounded_for_small_web_instance(self) -> None:
        options = _engine_options("postgresql+psycopg://example")

        self.assertEqual(options["pool_size"], 2)
        self.assertEqual(options["max_overflow"], 1)
        self.assertEqual(options["pool_timeout"], 15)
        self.assertEqual(options["connect_args"]["connect_timeout"], 10)

    def test_sqlite_keeps_local_timeout_without_postgres_pool_options(self) -> None:
        options = _engine_options("sqlite:///local.db")

        self.assertEqual(options["connect_args"]["timeout"], 30)
        self.assertNotIn("pool_size", options)

    def test_provider_application_error_is_not_cached_as_games(self) -> None:
        cache_dir = Path("data/test_provider_error_cache")
        cache_file = cache_dir / "games_unique.json"
        cache_file.unlink(missing_ok=True)
        with patch.dict("os.environ", {"API_SPORTS_KEY": "configured"}):
            api = BaseSportsAPI("https://provider.invalid", "test")
        api.cache_dir = cache_dir
        api.cache_dir.mkdir(parents=True, exist_ok=True)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"errors": {"token": "rejected"}, "response": []}
        api.http.get = Mock(return_value=response)

        try:
            with self.assertRaises(ProviderResponseError):
                api.get("games", cache_key="unique", force_refresh=True)
            self.assertFalse(cache_file.exists())
        finally:
            cache_file.unlink(missing_ok=True)
            cache_dir.rmdir()

    def test_provider_error_classification_is_safe_and_actionable(self) -> None:
        self.assertEqual(
            classify_provider_error({"access": "Your account is suspended"}),
            "account_suspended",
        )
        self.assertEqual(classify_provider_error({"rateLimit": "Exceeded"}), "quota_exceeded")
        self.assertEqual(classify_provider_error({"token": "Invalid"}), "credential_rejected")
        self.assertEqual(
            classify_provider_error({"plan": "Free plans do not have access to this season"}),
            "plan_restriction",
        )

    def test_free_plan_history_uses_allowed_season_instead_of_last_parameter(self) -> None:
        api = Mock()
        api.get.return_value = {
            "response": [
                {
                    "fixture": {"date": "2024-01-01", "status": {"short": "FT"}},
                    "goals": {"home": 1, "away": 0},
                },
                {
                    "fixture": {"date": "2024-02-01", "status": {"short": "FT"}},
                    "goals": {"home": 2, "away": 1},
                },
                {
                    "fixture": {"date": "2024-03-01", "status": {"short": "NS"}},
                    "goals": {"home": None, "away": None},
                },
            ]
        }
        service = object.__new__(FootballDataService)
        service.api = api
        service.sportmonks_api = Mock()

        with patch.dict("os.environ", {"API_FOOTBALL_HISTORY_SEASON": "2024"}):
            fixtures = service.get_recent_team_fixtures(541, last=1)

        params = api.get.call_args.kwargs["params"]
        self.assertEqual(params, {"team": 541, "season": 2024})
        self.assertNotIn("last", params)
        self.assertEqual(fixtures[0]["fixture"]["date"], "2024-02-01")

    def test_simultaneous_users_share_one_provider_request(self) -> None:
        cache_dir = Path("data/test_concurrent_provider_cache")
        cache_file = cache_dir / "games_shared.json"
        cache_file.unlink(missing_ok=True)
        with patch.dict("os.environ", {"API_SPORTS_KEY": "configured"}):
            api = BaseSportsAPI("https://provider.invalid", "test")
        api.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"errors": [], "response": [{"id": 1}]}

        def delayed_response(*args, **kwargs):
            del args, kwargs
            time.sleep(0.1)
            return response

        api.http.get = Mock(side_effect=delayed_response)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: api.get("games", cache_key="shared"), range(2)))
            self.assertEqual(api.http.get.call_count, 1)
            self.assertEqual([item["response"] for item in results], [[{"id": 1}], [{"id": 1}]])
        finally:
            cache_file.unlink(missing_ok=True)
            cache_dir.rmdir()

    def test_provider_health_distinguishes_empty_schedule_from_error(self) -> None:
        class FakeManager:
            def __init__(self, sport: str):
                self.sport = sport

            def get_games_by_date(self, **_: object) -> dict:
                if self.sport == "Error":
                    raise RuntimeError("secret provider details")
                return {"response": [] if self.sport == "Vacío" else [{"id": 1}], "errors": []}

        results = check_sports_connectivity(
            ["Con eventos", "Vacío", "Error"],
            fixture_date="2026-07-16",
            manager_factory=FakeManager,
        )

        self.assertEqual([item["status"] for item in results], ["connected", "connected", "error"])
        self.assertEqual([item["events"] for item in results], [1, 0, 0])
        self.assertNotIn("secret", results[2]["detail"])

    def test_provider_health_preserves_primary_suspension_with_empty_fallback(self) -> None:
        manager = Mock()
        manager.get_games_by_date.return_value = {
            "response": [],
            "_source": "sportmonks",
            "_provider_warnings": [
                {"provider": "api_sports", "reason": "account_suspended"}
            ],
        }
        result = check_sports_connectivity(
            ["Fútbol"], manager_factory=lambda sport: manager
        )
        self.assertEqual(result[0]["status"], "error")
        self.assertIn("suspendida", result[0]["detail"])

    def test_streamlit_entrypoint_defers_heavy_ml_imports(self) -> None:
        tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        top_level_modules = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_modules.add(node.module)

        self.assertNotIn("pandas", top_level_modules)
        self.assertNotIn("machine_learning.predictors.football_predictor", top_level_modules)
        self.assertNotIn("machine_learning.continuous_learning", top_level_modules)
        self.assertNotIn("machine_learning.shadow_validation", top_level_modules)

    def test_http_retries_are_bounded_to_transient_get_failures(self) -> None:
        session = build_retry_session()
        retries = session.get_adapter("https://").max_retries

        self.assertEqual(retries.total, 2)
        self.assertEqual(retries.allowed_methods, frozenset({"GET"}))
        self.assertEqual(tuple(retries.status_forcelist), TRANSIENT_STATUS_CODES)

    def test_neon_database_url_uses_psycopg_driver(self) -> None:
        with patch.dict(
            "os.environ",
            {"DATABASE_URL": "postgresql://user:password@example.neon.tech/app?sslmode=require"},
        ):
            url = _database_url()

        self.assertTrue(url.startswith("postgresql+psycopg://"))
        self.assertIn("sslmode=require", url)

    def test_football_can_start_with_sportmonks_only(self) -> None:
        with patch.dict(
            "os.environ",
            {"API_SPORTS_KEY": "", "SPORTMONKS_API_TOKEN": "configured-token"},
        ):
            api = FootballAPI()

        self.assertFalse(api.api_key)
        self.assertTrue(api.supplemental_api.available)

    def test_sportsdataio_maps_games_without_exposing_its_key(self) -> None:
        secret = "sportsdata-private-key"
        with patch.dict(
            "os.environ",
            {
                "SPORTSDATA_API_KEY": secret,
                "SPORTSDATA_SOCCER_COMPETITIONS": "3",
            },
        ):
            api = SportsDataSoccerAPI()
        api.cache_dir = Path("data/test_sportsdata_cache")
        api.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = api.cache_dir / "games_3_2099-01-01.json"
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [{
            "GameId": 99,
            "DateTime": "2099-01-01T18:00:00",
            "Status": "Scheduled",
            "CompetitionName": "Champions League",
            "HomeTeamId": 10,
            "HomeTeamName": "Local",
            "AwayTeamId": 20,
            "AwayTeamName": "Visitante",
            "HomeTeamScore": None,
            "AwayTeamScore": None,
        }]
        api.http.get = Mock(return_value=response)
        try:
            games = api.get_games_by_date("2099-01-01", force_refresh=True)
            call = api.http.get.call_args
            self.assertEqual(
                call.kwargs["headers"],
                {"Ocp-Apim-Subscription-Key": secret},
            )
            self.assertNotIn(secret, call.args[0])
            self.assertEqual(games[0]["provider"], "sportsdataio")
            self.assertEqual(games[0]["teams"]["away"]["name"], "Visitante")
        finally:
            cache_file.unlink(missing_ok=True)
            api.cache_dir.rmdir()

    def test_football_merges_sportsdataio_as_a_supplement(self) -> None:
        with patch.dict("os.environ", {"API_SPORTS_KEY": "configured"}):
            api = FootballAPI()
        primary = {
            "response": [{
                "fixture": {"id": 1, "date": "2026-07-18T10:00:00"},
                "teams": {
                    "home": {"name": "A"},
                    "away": {"name": "B"},
                },
            }],
            "results": 1,
            "_source": "api",
        }
        api.get = Mock(return_value=primary)
        api.supplemental_api = Mock(available=False)
        api.sportsdata_api = Mock(available=True)
        api.sportsdata_api.get_games_by_date.return_value = [{
            "provider": "sportsdataio",
            "fixture": {"id": "sportsdata:2", "date": "2026-07-18T12:00:00"},
            "teams": {
                "home": {"name": "C"},
                "away": {"name": "D"},
            },
        }]

        result = api.get_games_by_date("2026-07-18")

        self.assertEqual(result["results"], 2)
        self.assertEqual(result["_source"], "combined")

    def test_confidence_uses_history_consistency_agreement_and_quality(self) -> None:
        markets = [{
            "market_type": "home_win", "selection": "Local", "probability": 70.0,
            "extra_data_json": {"category": "Resultado"},
        }]
        features = {
            "home_matches_played": 20, "away_matches_played": 20,
            "home_scored_std": 0.5, "away_scored_std": 0.6,
            "diff_points_last5": 5, "recent_home_attack_vs_away_defense": 0.8,
            "real_home_advantage": 0.7,
        }
        enrich_football_markets(
            markets, features, {"home_win": 72.0}, {"home_win_probability": 68.0},
            {"result": True},
        )
        self.assertGreater(markets[0]["confidence_score"], 60)
        self.assertIn("+5.0 puntos", markets[0]["explanation"])
        self.assertIn("confidence_components", markets[0]["extra_data_json"])

    def test_confidence_discloses_small_historical_sample(self) -> None:
        markets = [{"market_type": "btts", "selection": "Sí", "probability": 65.0}]
        enrich_football_markets(
            markets, {"home_matches_played": 2, "away_matches_played": 3}, {}, {}, {}
        )
        self.assertIn("Confianza limitada", markets[0]["explanation"])
    def test_football_calibrator_is_separate_and_reduces_overconfidence(self) -> None:
        np = __import__("numpy")
        probabilities = np.tile(np.linspace(0.05, 0.95, 20), 6)
        targets = (probabilities > 0.65).astype(int)
        bundle, report = FootballProbabilityCalibrator.fit_market(
            probabilities, targets, [0, 1]
        )
        self.assertIsNotNone(bundle)
        self.assertGreater(report["brier_improvement"], 0)
        calibrator = FootballProbabilityCalibrator("v1", {"over_2_5": bundle})
        adjusted = calibrator.calibrate("over_2_5", [0.2, 0.8], [0, 1])
        self.assertAlmostEqual(float(adjusted.sum()), 1.0)

    def test_football_calibrator_rejects_different_model_version(self) -> None:
        from pathlib import Path

        directory = Path("data/test_calibrator")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "calibrator.joblib"
        report = directory / "report.json"
        try:
            FootballProbabilityCalibrator("v1", {}).save(path, report, {})
            loaded = FootballProbabilityCalibrator.load_compatible(
                path, {"model_version": "v2"}
            )
            self.assertIsNone(loaded)
        finally:
            path.unlink(missing_ok=True)
            report.unlink(missing_ok=True)
    def test_formula1_features_use_only_supplied_prior_results(self) -> None:
        history = [
            {"position": 2.0, "points": 18.0},
            {"position": 4.0, "points": 12.0},
            {"position": 1.0, "points": 25.0},
        ]
        row = Formula1Features.build(history, [3.0, 4.0], [2.0], 4, 24)
        self.assertEqual(set(row), set(Formula1Features.feature_columns()))
        self.assertAlmostEqual(row["avg_position_3"], 7 / 3)
        self.assertAlmostEqual(row["podium_rate_5"], 2 / 3)

    def test_formula1_predictor_falls_back_without_active_model(self) -> None:
        from machine_learning.predictors.formula1_predictor import Formula1Predictor

        predictor = Formula1Predictor(model_dir=Path("data/test_missing_formula1_model"))
        self.assertFalse(predictor.available)

    def test_analysis_transparency_requires_a_qualified_ml_market(self) -> None:
        self.assertEqual(
            classify_analysis("Football ML + Poisson", {"quality_gate": {"result": True}}),
            "hybrid_ai",
        )
        self.assertEqual(
            classify_analysis("Football ML + Poisson", {"quality_gate": {"result": False}}),
            "statistical",
        )
        self.assertEqual(classify_analysis("Poisson + Monte Carlo"), "statistical")

    def test_sportmonks_scores_are_mapped_by_participant_id(self) -> None:
        converted = SportmonksFootballAPI._convert_fixture(
            {
                "id": 50,
                "starting_at": "2026-07-15 20:00:00",
                "state": {"state": "FINISHED", "name": "Finished"},
                "participants": [
                    {"id": 10, "name": "Local", "meta": {"location": "home"}},
                    {"id": 20, "name": "Visitante", "meta": {"location": "away"}},
                ],
                "scores": [
                    {"description": "CURRENT", "score": {"participant": 10, "goals": 2}},
                    {"description": "CURRENT", "score": {"participant": 20, "goals": 1}},
                ],
            }
        )
        self.assertEqual(converted["goals"], {"home": 2, "away": 1})
        self.assertTrue(is_finished_status(converted["fixture"]["status"]))

    def test_sportmonks_failure_never_exposes_token(self) -> None:
        secret = "private-token-value"
        api = SportmonksFootballAPI()
        api.token = secret
        api.cache_dir = Path("data/test_sportmonks_security")
        api.http = Mock()
        api.http.get.side_effect = requests.RequestException(
            f"request failed: https://provider.test?api_token={secret}"
        )

        with self.assertRaisesRegex(RuntimeError, "provider request failed") as raised:
            api.get_games_by_date("2099-01-01", force_refresh=True)

        self.assertNotIn(secret, str(raised.exception))

    def test_nested_provider_status_is_detected_as_finished(self) -> None:
        status = {"state": {"short": "FT", "name": "Finished"}}
        self.assertTrue(is_finished_status(status))

    def test_balldontlie_final_score_is_extracted(self) -> None:
        score = extract_final_score(
            {"home_team_score": 112, "visitor_team_score": 108},
            "Basketball",
        )
        self.assertEqual(score, (112.0, 108.0))

    def test_player_availability_does_not_treat_missing_coverage_as_healthy(self) -> None:
        service = PlayerAvailabilityService()
        result = service.get_match_availability(
            "Basketball",
            {"game_id": 1, "home_id": 10, "away_id": 20},
        )
        self.assertEqual(result["coverage"], "unavailable")
        self.assertEqual(result["home"]["reported_absences"], 0)

    def test_player_availability_summarizes_confirmed_football_data(self) -> None:
        api = Mock()
        api.get.side_effect = [
            {"response": [{"team": {"id": 10}, "player": {"id": 1}}], "errors": []},
            {"response": [{"team": {"id": 20}, "startXI": [{}, {}]}], "errors": []},
        ]
        result = PlayerAvailabilityService(football_api=api).get_match_availability(
            "Fútbol",
            {"game_id": 99, "home_id": 10, "away_id": 20},
        )
        self.assertEqual(result["coverage"], "confirmed")
        self.assertEqual(result["home"]["reported_absences"], 1)
        self.assertEqual(result["away"]["confirmed_starters"], 2)

    def test_basketball_availability_matches_teams_without_adjusting_model(self) -> None:
        class NBAAvailability:
            available = True

            @staticmethod
            def get_teams(force_refresh=False):
                return {"data": [{"id": 1, "full_name": "Boston Celtics"}, {"id": 2, "full_name": "Atlanta Hawks"}]}

            @staticmethod
            def get_injuries(force_refresh=False):
                return {"data": [{"player": {"team_id": 1}, "status": "Out"}]}

        result = PlayerAvailabilityService(nba_api=NBAAvailability()).get_match_availability(
            "Basketball",
            {"game_id": 99, "league": "NBA", "home": "Boston Celtics", "away": "Atlanta Hawks"},
        )
        self.assertEqual(result["coverage"], "confirmed")
        self.assertEqual(result["home"]["reported_absences"], 1)
        self.assertEqual(result["away"]["reported_absences"], 0)

    def test_primary_view_keeps_top_baseball_and_basketball_events(self) -> None:
        games = [
            {"league": "MLB", "country": "USA"},
            {"league": "NBA - Las Vegas Summer League", "country": "USA"},
            {"league": "Regional Development League", "country": "USA"},
        ]
        self.assertEqual(len(filter_games_by_league_view(games, "Béisbol", "Principales")), 1)
        self.assertEqual(len(filter_games_by_league_view(games, "Basketball", "Principales")), 1)

    def test_all_leagues_places_primary_events_first(self) -> None:
        games = [
            {"league": "Regional League", "country": "USA", "label": "Regional"},
            {"league": "NBA", "country": "USA", "label": "NBA"},
            {"league": "EuroLeague", "country": "Europe", "label": "EuroLeague"},
        ]
        ordered = filter_games_by_league_view(games, "Basketball", "Todas")
        self.assertEqual([game["league"] for game in ordered], ["EuroLeague", "NBA", "Regional League"])

    def test_football_uses_supplemental_provider_when_primary_fails(self) -> None:
        api = FootballAPI()
        api.get = Mock(side_effect=ConnectionError("primary unavailable"))
        api.supplemental_api = Mock()
        api.supplemental_api.available = True
        api.supplemental_api.get_games_by_date.return_value = [
            {
                "fixture": {"id": "sportmonks:1", "date": "2026-07-15T20:00:00Z"},
                "teams": {
                    "home": {"name": "Equipo A"},
                    "away": {"name": "Equipo B"},
                },
            }
        ]

        result = api.get_games_by_date("2026-07-15")

        self.assertEqual(result["_source"], "sportmonks")
        self.assertEqual(result["results"], 1)

    def test_simulation_policy_is_kept_internal_and_safe(self) -> None:
        self.assertGreaterEqual(MIN_SIMULATIONS, 5_000)
        self.assertGreaterEqual(DEFAULT_SIMULATIONS, MIN_SIMULATIONS)

    def test_football_rejects_neutral_profiles_without_real_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "suficiente historial real"):
            FootballPredictionEngine.validate_team_profiles(
                {"played": 0, "avg_scored": 1.0, "avg_conceded": 1.0},
                {"played": 0, "avg_scored": 1.0, "avg_conceded": 1.0},
            )

    def test_football_accepts_profiles_with_real_history(self) -> None:
        FootballPredictionEngine.validate_team_profiles(
            {"played": 8, "avg_scored": 1.8, "avg_conceded": 0.9},
            {"played": 7, "avg_scored": 1.1, "avg_conceded": 1.4},
        )

    def test_formula1_analysis_without_network(self) -> None:
        engine = Formula1PredictionEngine()
        engine.api = _Formula1Results()
        result = engine.analyze_match(
            {"season": 2026, "round": 6, "date": "2026-08-01"},
            DEFAULT_SIMULATIONS,
        )
        self.assertEqual(result["model_name"], "F1 Recent Form + Monte Carlo")
        self.assertGreaterEqual(len(result["markets_to_save"]), 6)

    def test_hockey_analysis_without_network(self) -> None:
        engine = HockeyPredictionEngine()
        engine.api_sports.get = lambda *args, **kwargs: {
            "response": [
                {"team": {"id": 1}, "games": {"played": 10, "wins": {"total": 7}}},
                {"team": {"id": 2}, "games": {"played": 10, "wins": {"total": 4}}},
            ]
        }
        result = engine.analyze_match(
            {
                "home": "Local",
                "away": "Visitante",
                "home_id": 1,
                "away_id": 2,
                "date": "2026-08-01",
                "analysis_context": {"league_id": 57, "season": 2026},
            },
            DEFAULT_SIMULATIONS,
        )
        self.assertEqual(len(result["markets_to_save"]), 2)

    def test_mma_analysis_without_network(self) -> None:
        result = MMAPredictionEngine().analyze_match(
            {
                "home": "Peleador A",
                "away": "Peleador B",
                "analysis_context": {
                    "home_profile": {"record_wins": 12, "record_losses": 2},
                    "away_profile": {"record_wins": 8, "record_losses": 4},
                },
            },
            DEFAULT_SIMULATIONS,
        )
        self.assertEqual(len(result["markets_to_save"]), 2)

    def test_sqlite_integrity_read_only(self) -> None:
        with sqlite3.connect(f"file:{DATABASE_PATH.as_posix()}?mode=ro", uri=True) as connection:
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")

    def test_accumulated_metrics_keep_sports_separate(self) -> None:
        summary = ModelMetricsRepository().get_performance_summary()
        if summary is None:
            self.skipTest("Todavía no existen evaluaciones post-partido")
        self.assertGreaterEqual(summary["reviews"], 1)
        self.assertGreaterEqual(summary["evaluated_markets"], 1)
        self.assertTrue(0 <= summary["accuracy"] <= 1)
        self.assertTrue(0 <= summary["mean_brier_score"] <= 1)
        self.assertTrue(all(item["sport"] for item in summary["sports"]))
        self.assertTrue(all(item["sport"] for item in summary["models"]))
        self.assertTrue(all(item["model_version"] for item in summary["models"]))
        self.assertTrue(all(item["sport"] for item in summary["markets"]))

    def test_formula1_markets_are_evaluated_by_driver_selection(self) -> None:
        markets = [
            {"market_type": "f1_win", "selection": "Ana Uno", "probability": 45},
            {"market_type": "f1_win", "selection": "Beto Dos", "probability": 30},
            {"market_type": "f1_podium", "selection": "Beto Dos", "probability": 70},
        ]
        evaluation = evaluate_markets(
            markets,
            0,
            0,
            selection_outcomes={
                "f1_win": {"ana uno"},
                "f1_podium": {"ana uno", "beto dos", "caro tres"},
            },
            actual_outcome="winner:Ana Uno",
        )
        self.assertEqual(evaluation["evaluated_markets"], 3)
        self.assertEqual(evaluation["correct_markets"], 2)
        self.assertEqual(evaluation["actual_outcome"], "winner:Ana Uno")

    def test_low_probability_is_correct_when_event_does_not_happen(self) -> None:
        evaluation = evaluate_markets(
            [{"market_type": "home_win", "selection": "Local", "probability": 40}],
            home_score=90,
            away_score=102,
        )
        self.assertEqual(evaluation["correct_markets"], 1)
        self.assertFalse(evaluation["market_evaluations"][0]["is_pick"])

    def test_football_selects_informative_goal_line_not_easiest_line(self) -> None:
        lines = {
            "1.5": {"over": 70.0, "under": 30.0},
            "2.5": {"over": 58.0, "under": 42.0},
            "3.5": {"over": 25.0, "under": 75.0},
            "4.5": {"over": 10.0, "under": 90.0},
        }
        total = FootballPredictionEngine.select_goal_line(lines)
        self.assertEqual(total["market_type"], "over_2_5_goals")
        self.assertEqual(total["probability"], 58.0)

    def test_dynamic_football_total_is_evaluated(self) -> None:
        evaluation = evaluate_markets(
            [{"market_type": "under_2_5_goals", "selection": "Under 2.5 goles", "probability": 60}],
            home_score=1,
            away_score=1,
        )
        self.assertEqual(evaluation["evaluated_markets"], 1)
        self.assertEqual(evaluation["correct_markets"], 1)

    def test_football_market_risk_changes_with_probability(self) -> None:
        low_risk = FootballPredictionEngine._market("Test", "a", "A", 80)
        high_risk = FootballPredictionEngine._market("Test", "b", "B", 45)
        self.assertEqual(low_risk["risk"], "Bajo")
        self.assertEqual(high_risk["risk"], "Muy alto")

    def test_shared_market_risk_is_consistent_across_sports(self) -> None:
        markets = [{"probability": 78.0}, {"probability": 54.0}, {"probability": 40.0}]
        apply_probability_risk(markets)
        self.assertEqual([item["risk"] for item in markets], ["Bajo", "Alto", "Muy alto"])
        self.assertEqual(probability_risk_profile(66.0), ("Media-Alta", "Medio"))

    def test_public_markets_show_all_valid_signals_with_1x2_first(self) -> None:
        markets = [
            {"market_type": "home_win", "probability": 42, "selection": "Local"},
            {"market_type": "draw", "probability": 28, "selection": "Empate"},
            {"market_type": "away_win", "probability": 30, "selection": "Visitante"},
            {"market_type": "btts", "probability": 49.9, "selection": "Ambos"},
            {"market_type": "under_3_5_goals", "probability": 67, "selection": "Under"},
        ]
        filtered = visible_markets(markets)
        self.assertEqual(
            {market["market_type"] for market in filtered},
            {"home_win", "draw", "away_win", "btts", "under_3_5_goals"},
        )
        self.assertEqual(
            [market["market_type"] for market in filtered[:3]],
            ["home_win", "draw", "away_win"],
        )

    def test_football_expanded_markets_include_double_chance_and_team_goals(self) -> None:
        lines = {
            line: {"over": 50.0, "under": 50.0}
            for line in ("1.5", "2.5", "3.5", "4.5")
        }
        markets = FootballPredictionEngine.build_market_options(
            "Local", "Visitante", 40, 30, 30, lines, 55, 70, 65,
            {
                "home_over_1_5": 58,
                "away_over_1_5": 42,
                "home_clean_sheet": 35,
                "away_clean_sheet": 30,
            },
        )
        market_types = {item["market_type"] for item in markets}
        self.assertEqual(len(markets), 22)
        self.assertIn("double_chance_home_draw", market_types)
        self.assertIn("away_over_0_5_goals", market_types)
        self.assertIn("home_over_1_5_goals", market_types)
        self.assertIn("away_clean_sheet", market_types)

    def test_football_advanced_features_use_only_supplied_history(self) -> None:
        from machine_learning.features.football_features import FootballFeatures

        history = [
            {"scored": 2, "conceded": 0, "result": "W", "is_home": True},
            {"scored": 1, "conceded": 1, "result": "D", "is_home": False},
            {"scored": 3, "conceded": 1, "result": "W", "is_home": True},
        ]
        summary = FootballFeatures.summarize_history(history)
        self.assertEqual(summary["points_last5"], 7.0)
        self.assertEqual(summary["goals_scored_last5"], 6.0)
        self.assertEqual(summary["win_streak"], 1.0)
        self.assertEqual(summary["loss_streak"], 0.0)
        self.assertAlmostEqual(summary["clean_sheet_rate"], 1 / 3)
        self.assertAlmostEqual(summary["btts_rate"], 2 / 3)
        self.assertAlmostEqual(summary["home_points_per_game"], 3.0)
        self.assertEqual(len(FootballFeatures.feature_columns()), len(set(FootballFeatures.feature_columns())))

    def test_model_promotion_requires_enough_evidence(self) -> None:
        decision = evaluate_promotion(
            {"evaluated": 100, "accuracy": 0.60, "mean_brier_score": 0.24},
            {"evaluated": 12, "accuracy": 0.72, "mean_brier_score": 0.18},
        )
        self.assertEqual(decision["decision"], "insufficient_data")
        self.assertFalse(decision["automatic_change"])

    def test_paired_model_summary_weights_market_evidence(self) -> None:
        review_a = Mock(
            evaluated_markets=2, correct_markets=1, mean_brier_score=0.20,
            details_json=[{}, {}],
        )
        review_b = Mock(
            evaluated_markets=4, correct_markets=3, mean_brier_score=0.10,
            details_json=[{}, {}, {}, {}],
        )
        summary = ModelMetricsRepository._summarize_paired(
            [(review_a, Mock()), (review_b, Mock())], "candidate"
        )
        self.assertEqual(summary["paired_matches"], 2)
        self.assertEqual(summary["evaluated"], 6)
        self.assertAlmostEqual(summary["accuracy"], 4 / 6)
        self.assertAlmostEqual(summary["mean_brier_score"], (0.20 * 2 + 0.10 * 4) / 6)

    def test_post_match_service_skips_existing_review(self) -> None:
        service = PostMatchService()
        service.predictions = Mock()
        service.reviews = Mock()
        service.error_analysis = Mock()
        service.predictions.list_runs_by_match_id.return_value = [{
            "id": 7, "sport": "Basketball", "match_id": "game-1", "context_json": {}
        }]
        service.reviews.get_review.return_value = {
            "id": 9, "market_evaluations": [], "actual_outcome": "home_win"
        }
        processed = service._evaluate_match("Basketball", "game-1", 100, 90)
        self.assertEqual(processed, 0)
        service.predictions.list_markets_by_run.assert_not_called()
        service.reviews.save_review.assert_not_called()

    def test_model_error_analysis_records_context_and_detects_pattern(self) -> None:
        run = {
            "id": 4, "match_id": "fixture-4", "sport": "Fútbol",
            "context_json": {"analysis_context": {"league": "Premier League"}},
        }
        evaluation = {"market_evaluations": [{
            "market_type": "home_win", "selection": "Local", "probability": 78.0,
            "actual": False, "is_pick": True, "correct": False,
            "absolute_error": 0.78, "brier_score": 0.6084,
        }]}
        row = build_error_rows(run, 12, evaluation)[0]
        self.assertEqual(row["league"], "Premier League")
        self.assertEqual(row["match_type"], "strong_favorite")
        self.assertAlmostEqual(row["probability_difference"], 0.78)
        self.assertIn("high_confidence_miss", row["pattern_tags"])
        patterns = detect_failure_patterns([row, dict(row), dict(row)], minimum_samples=3)
        self.assertEqual(patterns[0]["patterns"], [
            "systematic_failure", "overconfidence", "false_positive_bias"
        ])

    def test_post_match_accepts_sportmonks_list_cache_payload(self) -> None:
        service = PostMatchService()
        service._evaluate_match = Mock(return_value=1)
        payload = [
            {
                "fixture": {"id": 55, "status": {"short": "FT"}},
                "goals": {"home": 2, "away": 1},
            }
        ]
        games = service._payload_games(payload)
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]["fixture"]["id"], 55)

    def test_post_match_process_games_uses_status_and_embedded_score(self) -> None:
        service = PostMatchService()
        service._evaluate_match = Mock(return_value=1)
        processed = service.process_games(
            "Basketball",
            [
                {
                    "game_id": "nba-7",
                    "status": "Final",
                    "home_team_score": 112,
                    "visitor_team_score": 108,
                }
            ],
        )
        self.assertEqual(processed, 1)
        service._evaluate_match.assert_called_once_with(
            sport="Basketball",
            match_id="nba-7",
            home_score=112.0,
            away_score=108.0,
        )

    def test_model_promotion_recommends_only_measurable_improvement(self) -> None:
        decision = evaluate_promotion(
            {"evaluated": 100, "accuracy": 0.60, "mean_brier_score": 0.24},
            {"evaluated": 100, "accuracy": 0.62, "mean_brier_score": 0.23},
        )
        self.assertEqual(decision["decision"], "promote")
        self.assertFalse(decision["automatic_change"])

        hold = evaluate_promotion(
            {"evaluated": 100, "accuracy": 0.60, "mean_brier_score": 0.24},
            {"evaluated": 100, "accuracy": 0.605, "mean_brier_score": 0.239},
        )
        self.assertEqual(hold["decision"], "hold")

    def test_shadow_validation_ignores_unsupported_sports(self) -> None:
        service = ShadowValidationService()
        self.assertIsNone(service.find_candidate("MMA"))

    def test_formula1_candidate_is_available_to_shadow_validation(self) -> None:
        service = ShadowValidationService()
        candidate = service.find_candidate("Fórmula 1")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.parent.name, "formula1")

    def test_shadow_validation_prefers_quality_over_recency(self) -> None:
        stronger = {
            "qualified_markets": ["home_win"],
            "metrics": {"home_win": {"roc_auc": 0.70, "accuracy_lift": 0.08, "brier_score": 0.21}},
        }
        newer_but_weaker = {
            "qualified_markets": ["home_win"],
            "metrics": {"home_win": {"roc_auc": 0.60, "accuracy_lift": 0.02, "brier_score": 0.24}},
        }
        self.assertGreater(
            ShadowValidationService._candidate_score(stronger),
            ShadowValidationService._candidate_score(newer_but_weaker),
        )

    def test_model_registry_is_read_only_without_confirmation(self) -> None:
        registry = ModelRegistry()
        self.assertIsInstance(registry.list_versions("baseball"), list)
        with self.assertRaises(PermissionError):
            registry.promote("baseball", "does-not-exist")
        with self.assertRaises(PermissionError):
            registry.rollback("baseball")

    def test_richer_form_features_are_finite_and_backward_compatible(self) -> None:
        history = [
            {"scored": 4.0 + index % 3, "allowed": 3.0 + index % 2, "won": float(index % 2 == 0)}
            for index in range(12)
        ]
        summary = BaseballFeatures.summarize_history(history)
        for feature in (
            "avg_margin", "avg_margin_last5", "scoring_trend",
            "allowed_trend", "win_rate_trend", "sample_strength",
        ):
            self.assertIn(feature, summary)
            self.assertIsInstance(summary[feature], float)
        row = BaseballFeatures.from_summaries(summary, summary)
        self.assertEqual(set(row), set(BaseballFeatures.feature_columns()))
        self.assertEqual(row["diff_avg_margin"], 0.0)
        self.assertGreaterEqual(len(BaseballFeatures.feature_columns()), 100)
        self.assertEqual(len(BaseballFeatures.feature_columns()), len(set(BaseballFeatures.feature_columns())))

    def test_schedule_and_venue_features_use_only_prior_games(self) -> None:
        history = [
            {"scored": 5.0, "allowed": 2.0, "won": 1.0, "game_date": "2026-07-01T20:00:00Z", "is_home": True},
            {"scored": 3.0, "allowed": 4.0, "won": 0.0, "game_date": "2026-07-06T20:00:00Z", "is_home": False},
            {"scored": 20.0, "allowed": 0.0, "won": 1.0, "game_date": "2026-07-20T20:00:00Z", "is_home": True},
        ]
        summary = BaseballFeatures.summarize_history(history, "2026-07-08T20:00:00Z")
        self.assertEqual(summary["rest_days"], 2.0)
        self.assertEqual(summary["games_last_7"], 2.0)
        self.assertEqual(summary["games_last_14"], 2.0)
        self.assertEqual(summary["venue_home_win_rate"], 1.0)
        self.assertEqual(summary["venue_away_win_rate"], 0.0)
        self.assertEqual(summary["avg_scored"], 4.0)

    def test_schedule_strength_compares_results_with_pregame_expectation(self) -> None:
        history = [
            {"scored": 5.0, "allowed": 2.0, "won": 1.0, "opponent_elo": 1600.0, "expected_win_probability": 0.4},
            {"scored": 4.0, "allowed": 3.0, "won": 1.0, "opponent_elo": 1550.0, "expected_win_probability": 0.6},
        ]
        summary = BaseballFeatures.summarize_history(history)
        self.assertEqual(summary["avg_opponent_elo"], 1575.0)
        self.assertEqual(summary["expected_win_rate"], 0.5)
        self.assertEqual(summary["performance_vs_expected"], 0.5)

    def test_validated_ml_weight_is_conservative_and_bounded(self) -> None:
        weak = {"metrics": {"market": {"accuracy_lift": 0.0, "roc_auc": 0.5}}}
        strong = {"metrics": {"market": {"accuracy_lift": 0.20, "roc_auc": 0.85}}}
        self.assertEqual(validated_ml_weight(weak, "market"), 0.5)
        self.assertEqual(validated_ml_weight(strong, "market"), 0.65)

    def test_market_quality_gate_can_reject_negligible_improvement(self) -> None:
        from machine_learning.model_quality import market_model_is_qualified

        metadata = {
            "metrics": {
                "market": {"accuracy_lift": 0.008, "roc_auc": 0.56},
            }
        }
        self.assertFalse(
            market_model_is_qualified(
                metadata, "market", minimum_lift=0.01, minimum_auc=0.55
            )
        )

    def test_calibration_error_is_zero_for_perfect_probabilities(self) -> None:
        self.assertEqual(expected_calibration_error([0, 1], [0.0, 1.0]), 0.0)

    def test_multiclass_metrics_measure_auc_brier_and_calibration(self) -> None:
        metrics = multiclass_probability_metrics(
            [-1, 0, 1, -1, 0, 1],
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90],
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
                [0.10, 0.10, 0.80],
            ],
            [-1, 0, 1],
        )
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertLess(metrics["brier_score"], 0.1)
        self.assertLess(metrics["calibration_error"], 0.25)

    def test_automatic_promotion_requires_all_three_quality_metrics(self) -> None:
        complete = {
            "metrics": {
                "home_win": {
                    "roc_auc": 0.70,
                    "brier_score": 0.20,
                    "calibration_error": 0.04,
                }
            },
            "qualified_markets": ["home_win"],
        }
        missing_calibration = {
            "metrics": {
                "home_win": {"roc_auc": 0.72, "brier_score": 0.19}
            },
            "qualified_markets": ["home_win"],
        }
        comparisons = ModelRegistry._comparable_market_metrics(complete, complete)
        blocked = ModelRegistry._comparable_market_metrics(complete, missing_calibration)
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(blocked, [])

    def test_football_flat_metrics_are_comparable_when_complete(self) -> None:
        metadata = {
            "metrics": {
                "over_auc": 0.62,
                "over_brier_score": 0.22,
                "over_calibration_error": 0.05,
            },
            "qualified_markets": ["over_2_5"],
        }
        comparisons = ModelRegistry._comparable_market_metrics(metadata, metadata)
        self.assertEqual(comparisons[0][0], "over_2_5")
        self.assertEqual(comparisons[0][1]["roc_auc"], 0.62)

    def test_incremental_football_example_builds_real_training_labels(self) -> None:
        row = ContinuousLearningService._football_training_row({
            "features": {"feature_a": 1.5},
            "home_score": 2,
            "away_score": 1,
            "evaluated_at": "2026-07-16T10:00:00",
        })
        self.assertEqual(row["result_label"], 1)
        self.assertEqual(row["over_25_label"], 1)
        self.assertEqual(row["btts_label"], 1)
        self.assertEqual(row["feature_a"], 1.5)

    def test_incremental_training_labels_cover_other_main_sports(self) -> None:
        example = {
            "features": {"feature_a": 2.0}, "home_score": 112,
            "away_score": 105, "evaluated_at": "2026-07-16T10:00:00",
        }
        basketball = ContinuousLearningService._training_row("basketball", example)
        self.assertEqual(basketball["season"], "2025-2026")
        self.assertEqual(basketball["home_win_label"], 1)
        self.assertEqual(basketball["over_2195_label"], 0)

    def test_probability_ensemble_combines_independent_predictions(self) -> None:
        first = Mock(classes_=[0, 1])
        second = Mock(classes_=[0, 1])
        first.fit.return_value = first
        second.fit.return_value = second
        first.predict_proba.return_value = __import__("numpy").array([[0.2, 0.8]])
        second.predict_proba.return_value = __import__("numpy").array([[0.6, 0.4]])
        ensemble = ProbabilityEnsemble([first, second], weights=[0.75, 0.25]).fit([[0]], [1])
        probability = ensemble.predict_proba([[0]])[0][1]
        self.assertAlmostEqual(probability, 0.7)

    def test_probability_ensemble_excludes_rejected_model(self) -> None:
        from unittest.mock import MagicMock

        ml, monte_carlo = MagicMock(), MagicMock()
        ml.fit.return_value = ml
        monte_carlo.fit.return_value = monte_carlo
        ml.classes_ = monte_carlo.classes_ = __import__("numpy").array([0, 1])
        ml.predict_proba.return_value = __import__("numpy").array([[0.1, 0.9]])
        monte_carlo.predict_proba.return_value = __import__("numpy").array([[0.7, 0.3]])
        ensemble = ProbabilityEnsemble(
            [ml, monte_carlo], estimator_names=["ml", "monte_carlo"],
            quality_profiles={
                "ml": {"qualified": False},
                "monte_carlo": {"qualified": True, "global": {"roc_auc": 0.55, "brier_score": 0.23, "evaluated": 200}},
            },
        ).fit([[0]], [1])
        self.assertAlmostEqual(ensemble.predict_proba([[0]])[0][1], 0.3)

    def test_probability_ensemble_changes_weight_by_league_and_market(self) -> None:
        from unittest.mock import MagicMock

        models = [MagicMock(), MagicMock()]
        for model in models:
            model.fit.return_value = model
            model.classes_ = __import__("numpy").array([0, 1])
        profiles = {
            "ml": {"qualified": True, "leagues": {
                "Premier League": {"markets": {"result": {"roc_auc": 0.68, "brier_score": 0.19, "accuracy_lift": 0.06, "evaluated": 300}}},
                "MLS": {"markets": {"result": {"roc_auc": 0.51, "brier_score": 0.27, "accuracy_lift": 0.0, "evaluated": 80}}},
            }},
            "monte_carlo": {"qualified": True, "leagues": {
                "Premier League": {"markets": {"result": {"roc_auc": 0.56, "brier_score": 0.23, "accuracy_lift": 0.01, "evaluated": 300}}},
                "MLS": {"markets": {"result": {"roc_auc": 0.62, "brier_score": 0.21, "accuracy_lift": 0.04, "evaluated": 200}}},
            }},
        }
        ensemble = ProbabilityEnsemble(
            models, estimator_names=["ml", "monte_carlo"], quality_profiles=profiles
        ).fit([[0]], [1])
        premier = ensemble.weights_for_context("Premier League", "result")
        mls = ensemble.weights_for_context("MLS", "result")
        self.assertGreater(premier["ml"], premier["monte_carlo"])
        self.assertGreater(mls["monte_carlo"], mls["ml"])

    def test_backtest_sorting_contract_rejects_missing_temporal_columns(self) -> None:
        import pandas as pd
        schema = __import__(
            "machine_learning.backtesting.walk_forward", fromlist=["SCHEMAS"]
        ).SCHEMAS["football"]
        with self.assertRaises(ValueError):
            WalkForwardBacktester._validate(pd.DataFrame({"result_label": [1]}), schema)

    def test_backtest_ml_gate_requires_prior_predictions(self) -> None:
        quality = WalkForwardBacktester._raw_market_quality([], "result")
        self.assertFalse(quality["qualified"])
        self.assertEqual(quality["reason"], "insufficient_history")

    def test_football_backtest_fallback_is_normalized(self) -> None:
        import pandas as pd

        fallback = WalkForwardBacktester._football_fallback(pd.Series({
            "home_avg_scored": 1.8, "away_avg_conceded": 1.5,
            "away_avg_scored": 1.1, "home_avg_conceded": 0.9,
        }))
        self.assertAlmostEqual(sum(fallback["result"].values()), 1.0)
        self.assertTrue(0.0 <= fallback["over_2_5"] <= 1.0)
        self.assertTrue(0.0 <= fallback["btts"] <= 1.0)

    def test_football_legacy_model_requires_baseline_evidence(self) -> None:
        predictor = object.__new__(__import__(
            "machine_learning.predictors.football_predictor",
            fromlist=["FootballPredictor"],
        ).FootballPredictor)
        predictor.metadata = {"metrics": {"result_accuracy": 0.60}}
        self.assertFalse(predictor._legacy_result_is_qualified())
        predictor.metadata = {
            "metrics": {
                "result_accuracy": 0.60,
                "result_baseline_accuracy": 0.55,
                "result_accuracy_lift": 0.05,
            }
        }
        self.assertTrue(predictor._legacy_result_is_qualified())

    def test_elo_uses_only_previous_results_and_updates_both_teams(self) -> None:
        ratings = EloRatings()
        before = ratings.features(1, 2)
        self.assertGreater(before["elo_home_probability"], 0.5)
        ratings.update(1, 2, 3, 1)
        after = ratings.features(1, 2)
        self.assertGreater(after["home_elo"], before["home_elo"])
        self.assertLess(after["away_elo"], before["away_elo"])
        self.assertGreater(after["elo_home_probability"], before["elo_home_probability"])

    def test_live_history_uses_chronological_opponent_context(self) -> None:
        features = BaseballFeatures()
        games = [
            {
                "id": 10,
                "date": "2026-07-01T00:00:00Z",
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "scores": {"home": {"total": 5}, "away": {"total": 3}},
            }
        ]
        context = {10: {1: {"opponent_elo": 1580.0, "expected_win_probability": 0.42}}}
        summary = features._summarize_api_games(games, 1, "2026-07-02T00:00:00Z", context)
        self.assertEqual(summary["avg_opponent_elo"], 1580.0)
        self.assertEqual(summary["expected_win_rate"], 0.42)



if __name__ == "__main__":
    unittest.main()
