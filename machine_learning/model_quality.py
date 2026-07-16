from __future__ import annotations

from typing import Any

import numpy as np


def market_model_is_qualified(
    metadata: dict[str, Any],
    market: str,
    *,
    minimum_lift: float = 0.0,
    minimum_auc: float = 0.5,
) -> bool:
    """Return True only when a binary model beats its validation baseline."""
    metrics = metadata.get("metrics", {}).get(market, {})
    try:
        return (
            float(metrics.get("accuracy_lift", float("-inf"))) > minimum_lift
            and float(metrics.get("roc_auc", 0.0)) >= minimum_auc
        )
    except (TypeError, ValueError):
        return False


def metric_is_qualified(
    metadata: dict[str, Any], metric: str, *, minimum: float
) -> bool:
    """Return True when a scalar validation metric reaches a safe threshold."""
    try:
        return float(metadata.get("metrics", {}).get(metric, 0.0)) >= minimum
    except (TypeError, ValueError):
        return False


def validated_ml_weight(
    metadata: dict[str, Any],
    market: str,
    *,
    minimum: float = 0.50,
    maximum: float = 0.65,
) -> float:
    """Translate out-of-sample lift and AUC into a conservative blend weight."""
    metrics = metadata.get("metrics", {}).get(market, {})
    try:
        lift = max(0.0, float(metrics.get("accuracy_lift", 0.0)))
        auc_edge = max(0.0, float(metrics.get("roc_auc", 0.5)) - 0.5)
    except (TypeError, ValueError):
        return minimum
    quality = (2.0 * lift) + auc_edge
    return min(maximum, max(minimum, minimum + quality * 0.4))


def expected_calibration_error(
    target: Any, probability: Any, bins: int = 10
) -> float:
    """Measure probability calibration; lower values are better."""
    y = np.asarray(target, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if y.size == 0 or y.size != p.size:
        raise ValueError("Target y probabilidades deben tener el mismo tamaño.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(y.size)
    error = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (p >= edges[index]) & (
            p <= edges[index + 1] if upper_inclusive else p < edges[index + 1]
        )
        if not mask.any():
            continue
        error += (float(mask.sum()) / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)
