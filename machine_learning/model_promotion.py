from __future__ import annotations

from typing import Any


MIN_EVALUATED_MARKETS = 30
MIN_ACCURACY_GAIN = 0.01
MIN_BRIER_GAIN = 0.005
MAX_METRIC_REGRESSION = 0.002


def evaluate_promotion(
    active: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return an auditable recommendation without changing model files."""
    if not active or not candidate:
        return _decision("insufficient_data", "Faltan métricas comparables.")

    active_count = int(active.get("evaluated", 0))
    candidate_count = int(candidate.get("evaluated", 0))
    if min(active_count, candidate_count) < MIN_EVALUATED_MARKETS:
        return _decision(
            "insufficient_data",
            f"Se requieren al menos {MIN_EVALUATED_MARKETS} mercados evaluados por versión.",
        )

    accuracy_gain = float(candidate.get("accuracy", 0.0)) - float(active.get("accuracy", 0.0))
    brier_gain = float(active.get("mean_brier_score", 1.0)) - float(candidate.get("mean_brier_score", 1.0))
    promote = (
        accuracy_gain >= MIN_ACCURACY_GAIN and brier_gain >= -MAX_METRIC_REGRESSION
    ) or (
        brier_gain >= MIN_BRIER_GAIN and accuracy_gain >= -MAX_METRIC_REGRESSION
    )
    return {
        "decision": "promote" if promote else "hold",
        "reason": (
            "El candidato mejora métricas sin una regresión material."
            if promote
            else "El candidato todavía no demuestra una mejora suficiente."
        ),
        "accuracy_gain": accuracy_gain,
        "brier_gain": brier_gain,
        "active_evaluated": active_count,
        "candidate_evaluated": candidate_count,
        "automatic_change": False,
    }


def _decision(decision: str, reason: str) -> dict[str, Any]:
    return {
        "decision": decision,
        "reason": reason,
        "accuracy_gain": None,
        "brier_gain": None,
        "automatic_change": False,
    }
