from __future__ import annotations

import math
import unicodedata
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Combine only qualified estimators, with weights adapted to context."""

    def __init__(
        self,
        estimators: list[Any],
        weights: list[float] | None = None,
        estimator_names: list[str] | None = None,
        quality_profiles: dict[str, Any] | None = None,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.estimator_names = estimator_names
        self.quality_profiles = quality_profiles
        self.context_: dict[str, str | None] = {"league": None, "market": None}

    def fit(self, X: Any, y: Any) -> "ProbabilityEnsemble":
        if not self.estimators:
            raise ValueError("El ensamble requiere al menos un estimador.")
        for estimator in self.estimators:
            estimator.fit(X, y)
        self.classes_ = np.asarray(self.estimators[0].classes_)
        if any(not np.array_equal(self.classes_, estimator.classes_) for estimator in self.estimators):
            raise ValueError("Los estimadores del ensamble no comparten clases.")
        names = self.estimator_names or [f"model_{index}" for index in range(len(self.estimators))]
        if len(names) != len(self.estimators) or len(set(names)) != len(names):
            raise ValueError("Los nombres de los estimadores no son válidos.")
        weights = self.weights or [1.0] * len(self.estimators)
        if len(weights) != len(self.estimators) or any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("Los pesos del ensamble no son válidos.")
        self.estimator_names_ = list(names)
        self.base_weights_ = np.asarray(weights, dtype=float)
        self.normalized_weights_ = self.base_weights_ / float(self.base_weights_.sum())
        return self

    def set_context(self, league: str | None = None, market: str | None = None) -> "ProbabilityEnsemble":
        """Select the evidence scope used by subsequent predictions."""
        self.context_ = {"league": league, "market": market}
        return self

    def weights_for_context(
        self, league: str | None = None, market: str | None = None
    ) -> dict[str, float]:
        if not hasattr(self, "base_weights_"):
            raise ValueError("El ensamble debe entrenarse antes de calcular pesos.")
        if not self.quality_profiles:
            return dict(zip(self.estimator_names_, self.normalized_weights_))

        scores = []
        for name, base_weight in zip(self.estimator_names_, self.base_weights_):
            profile = self.quality_profiles.get(name, {})
            evidence = self._context_evidence(profile, league, market)
            scores.append(float(base_weight) * self._quality_score(profile, evidence, market))
        total = float(sum(scores))
        if total <= 0:
            raise ValueError("No hay modelos aprobados para esta liga y mercado.")
        return {
            name: score / total for name, score in zip(self.estimator_names_, scores)
        }

    def predict_proba(self, X: Any) -> np.ndarray:
        probabilities = [estimator.predict_proba(X) for estimator in self.estimators]
        context_weights = self.weights_for_context(**self.context_)
        weights = [context_weights[name] for name in self.estimator_names_]
        return np.average(np.asarray(probabilities), axis=0, weights=weights)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    @classmethod
    def _context_evidence(
        cls, profile: dict[str, Any], league: str | None, market: str | None
    ) -> dict[str, Any]:
        evidence = profile.get("global", profile.get("metrics", {}))
        leagues = profile.get("leagues", {})
        normalized_league = cls._normalize(league)
        league_profile = next(
            (value for key, value in leagues.items() if cls._normalize(key) == normalized_league),
            None,
        )
        if league_profile:
            evidence = league_profile
        markets = evidence.get("markets", {}) if isinstance(evidence, dict) else {}
        if market and market in markets:
            evidence = markets[market]
        return evidence if isinstance(evidence, dict) else {}

    @staticmethod
    def _quality_score(
        profile: dict[str, Any], evidence: dict[str, Any], market: str | None
    ) -> float:
        qualified_markets = profile.get("qualified_markets")
        if profile.get("qualified") is False or (
            qualified_markets is not None and market and market not in qualified_markets
        ) or evidence.get("qualified") is False:
            return 0.0

        # Profiles with no quality evidence retain legacy behavior. Once evidence
        # exists, it must show at least non-negative discrimination/lift.
        if not evidence:
            return 1.0
        try:
            auc = float(evidence.get("roc_auc", evidence.get("auc", 0.5)))
            lift = float(evidence.get("accuracy_lift", 0.0))
            brier = float(evidence.get("brier_score", 0.25))
            samples = int(evidence.get("evaluated", evidence.get("samples", 0)))
        except (TypeError, ValueError):
            return 0.0
        if auc < 0.5 or lift < 0.0 or brier < 0.0 or brier > 1.0:
            return 0.0
        evidence_factor = min(1.0, math.sqrt(max(samples, 1) / 200.0))
        discrimination = 1.0 + max(0.0, auc - 0.5) * 4.0
        lift_factor = 1.0 + min(0.5, max(0.0, lift) * 4.0)
        calibration_factor = max(0.20, 1.0 - brier)
        return evidence_factor * discrimination * lift_factor * calibration_factor

    @staticmethod
    def _normalize(value: str | None) -> str:
        raw = unicodedata.normalize("NFKD", str(value or ""))
        return " ".join("".join(char for char in raw if not unicodedata.combining(char)).lower().split())
