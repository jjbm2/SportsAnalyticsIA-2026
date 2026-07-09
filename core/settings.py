from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

APP_NAME = "SportsAnalyticsAI"
APP_VERSION = "0.1.0"

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
DATABASE_DIR = DATA_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"

DATABASE_PATH = DATABASE_DIR / "sports_analytics.db"

DEFAULT_SIMULATIONS = 100_000