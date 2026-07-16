from datetime import datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import relationship

from database.database import Base


class Sport(Base):
    __tablename__ = "sports"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)

    competitions = relationship("Competition", back_populates="sport")


class Competition(Base):
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True, index=True)
    sport_id = Column(Integer, ForeignKey("sports.id"), nullable=False)
    name = Column(String, nullable=False)
    country = Column(String, nullable=True)
    season = Column(String, nullable=True)
    api_id = Column(String, nullable=True)
    last_update = Column(DateTime, default=datetime.utcnow)

    sport = relationship("Sport", back_populates="competitions")
    teams = relationship("Team", back_populates="competition")
    matches = relationship("Match", back_populates="competition")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=True)
    name = Column(String, nullable=False)
    country = Column(String, nullable=True)
    logo = Column(String, nullable=True)
    api_id = Column(String, nullable=True)

    competition = relationship("Competition", back_populates="teams")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    match_date = Column(DateTime, nullable=True)
    status = Column(String, default="scheduled")
    venue = Column(String, nullable=True)
    referee = Column(String, nullable=True)
    weather = Column(String, nullable=True)
    api_id = Column(String, nullable=True)

    competition = relationship("Competition", back_populates="matches")


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id = Column(Integer, primary_key=True, index=True)
    sport = Column(String, nullable=False, index=True)
    match_id = Column(String, nullable=True, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    simulations = Column(Integer, nullable=False)
    status = Column(String, default="completed")
    context_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    markets = relationship(
        "PredictionMarket",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class PredictionMarket(Base):
    __tablename__ = "prediction_markets"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("prediction_runs.id"), nullable=False, index=True)
    market_type = Column(String, nullable=False, index=True)
    selection = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    confidence = Column(String, nullable=True)
    risk = Column(String, nullable=True)
    extra_data_json = Column(JSON, nullable=True)

    run = relationship("PredictionRun", back_populates="markets")


class PostMatchReview(Base):
    __tablename__ = "post_match_reviews"

    id = Column(Integer, primary_key=True, index=True)
    prediction_run_id = Column(
        Integer,
        ForeignKey("prediction_runs.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    match_id = Column(String, nullable=False, index=True)
    sport = Column(String, nullable=False, index=True)
    home_score = Column(Float, nullable=False)
    away_score = Column(Float, nullable=False)
    actual_outcome = Column(String, nullable=False)
    evaluated_markets = Column(Integer, nullable=False, default=0)
    correct_markets = Column(Integer, nullable=False, default=0)
    accuracy = Column(Float, nullable=True)
    mean_absolute_error = Column(Float, nullable=True)
    mean_brier_score = Column(Float, nullable=True)
    details_json = Column(JSON, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ModelErrorAnalysis(Base):
    __tablename__ = "model_error_analysis"
    __table_args__ = (
        UniqueConstraint(
            "prediction_run_id", "market_type", "selection",
            name="uq_model_error_run_market_selection",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    prediction_run_id = Column(Integer, ForeignKey("prediction_runs.id"), nullable=False, index=True)
    post_match_review_id = Column(Integer, ForeignKey("post_match_reviews.id"), nullable=False, index=True)
    match_id = Column(String, nullable=False, index=True)
    sport = Column(String, nullable=False, index=True)
    league = Column(String, nullable=False, default="unknown", index=True)
    market_type = Column(String, nullable=False, index=True)
    match_type = Column(String, nullable=False, default="standard", index=True)
    selection = Column(String, nullable=False)
    predicted_probability = Column(Float, nullable=False)
    predicted_event = Column(Boolean, nullable=False)
    actual_event = Column(Boolean, nullable=False)
    probability_difference = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=False)
    correct = Column(Boolean, nullable=False, index=True)
    pattern_tags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    plan = Column(String, nullable=False, default="free")
    is_admin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UserUsage(Base):
    __tablename__ = "user_usage"
    __table_args__ = (
        UniqueConstraint("user_id", "sport", "date", name="uq_user_usage_day_sport"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    sport = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    predictions_count = Column(Integer, nullable=False, default=0)
    extra_predictions = Column(Integer, nullable=False, default=0)


class PaymentRequest(Base):
    __tablename__ = "payment_requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    plan = Column(String, nullable=False)
    billing_cycle = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, nullable=False, default="pending", index=True)
    receipt_path = Column(String, nullable=True)
    proof_path = Column(String, nullable=True)
    proof_uploaded_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    plan = Column(String, nullable=False)
    billing_cycle = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    active = Column(Boolean, nullable=False, default=True, index=True)
    payment_request_id = Column(Integer, ForeignKey("payment_requests.id"), nullable=True)
