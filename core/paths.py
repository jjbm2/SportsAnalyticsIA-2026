import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = Path(
    os.getenv("SPORTSANALYTICS_DATA_DIR", str(BASE_DIR / "data"))
).expanduser()
DATABASE_DIR = DATA_DIR / "database"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

LOGO_PATH = ASSETS_DIR / "logo.svg"
DATABASE_PATH = DATABASE_DIR / "sports_analytics.db"
