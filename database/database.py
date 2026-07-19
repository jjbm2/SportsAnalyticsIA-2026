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


def _engine_options(database_url: str) -> dict:
    options = {
        "echo": False,
        "future": True,
        "pool_pre_ping": True,
    }
    if database_url.startswith("sqlite"):
        options["connect_args"] = {"timeout": 30}
    else:
        options.update({
            "pool_size": 2,
            "max_overflow": 1,
            "pool_timeout": 15,
            "pool_recycle": 300,
            "pool_use_lifo": True,
            "connect_args": {"connect_timeout": 10},
        })
    return options


engine_options = _engine_options(DATABASE_URL)

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
    _ensure_user_moderation_columns()


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


def _ensure_user_moderation_columns() -> None:
    """Add reversible account suspension fields to existing databases."""
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("users")}
    datetime_type = "TIMESTAMP" if engine.dialect.name == "postgresql" else "DATETIME"
    boolean_type = "BOOLEAN" if engine.dialect.name == "postgresql" else "BOOLEAN"
    with engine.begin() as connection:
        if "is_banned" not in columns:
            connection.execute(text(
                f"ALTER TABLE users ADD COLUMN is_banned {boolean_type} NOT NULL DEFAULT FALSE"
            ))
        if "banned_at" not in columns:
            connection.execute(text(f"ALTER TABLE users ADD COLUMN banned_at {datetime_type}"))
        if "ban_reason" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN ban_reason VARCHAR"))
