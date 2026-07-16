import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from core.paths import DATABASE_DIR, DATABASE_PATH

DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def _database_url() -> str:
    configured = (os.getenv("DATABASE_URL") or "").strip()
    if not configured:
        return f"sqlite:///{DATABASE_PATH}"
    if configured.startswith("postgresql://"):
        return configured.replace("postgresql://", "postgresql+psycopg://", 1)
    if configured.startswith("postgres://"):
        return configured.replace("postgres://", "postgresql+psycopg://", 1)
    return configured


DATABASE_URL = _database_url()
engine_options = {
    "echo": False,
    "future": True,
    "pool_pre_ping": True,
}
if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"timeout": 30}
else:
    engine_options["pool_recycle"] = 300

engine = create_engine(DATABASE_URL, **engine_options)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True
)

Base = declarative_base()


def get_session():
    return SessionLocal()


def create_database():
    # Register every mapped table even when database initialization is called
    # before repositories import their model classes.
    from database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_payment_proof_columns()


def _ensure_payment_proof_columns() -> None:
    """Idempotent migration for installations created before proof uploads."""
    inspector = inspect(engine)
    if "payment_requests" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("payment_requests")}
    datetime_type = "TIMESTAMP" if engine.dialect.name == "postgresql" else "DATETIME"
    with engine.begin() as connection:
        if "proof_path" not in columns:
            connection.execute(text("ALTER TABLE payment_requests ADD COLUMN proof_path VARCHAR"))
        if "proof_uploaded_at" not in columns:
            connection.execute(text(
                f"ALTER TABLE payment_requests ADD COLUMN proof_uploaded_at {datetime_type}"
            ))
        if "receipt_path" in columns:
            connection.execute(text(
                "UPDATE payment_requests SET proof_path = receipt_path "
                "WHERE proof_path IS NULL AND receipt_path IS NOT NULL"
            ))
