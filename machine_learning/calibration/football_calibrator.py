from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class FootballProbabilityCalibrator:
    """Post-process football probabilities without changing the base models."""

    def __init__(self, base_model_version: str, markets: dict[str, dict[str, Any]]):
        self.base_model_version = str(base_model_version)
        self.markets = markets

    @staticmethod
    def model_version(metadata: dict[str, Any]) -> str:
        return str(metadata.get("model_version") or metadata.get("trained_at") or "unknown")

    @classmethod
    def fit_market(
        cls,
        probabilities: np.ndarray,
        targets: np.ndarray,
        classes: list[int],
        fit_fraction: float = 0.6,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        probabilities = np.asarray(probabilities, dtype=float)
        targets = np.asarray(targets)
        if probabilities.ndim == 1:
            probabilities = np.column_stack((1.0 - probabilities, probabilities))
        if len(targets) < 50 or probabilities.shape != (len(targets), len(classes)):
            raise ValueError("Se requieren al menos 50 predicciones temporales completas.")
        split = max(30, min(len(targets) - 20, int(len(targets) * fit_fraction)))
        fit_p, validation_p = probabilities[:split], probabilities[split:]
        fit_y, validation_y = targets[:split], targets[split:]
        before = cls._metrics(validation_p, validation_y, classes)
        candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for method in ("sigmoid", "isotonic"):
            fitted = cls._fit_one_vs_rest(fit_p, fit_y, classes, method)
            adjusted = cls._apply(fitted, validation_p, classes)
            metrics = cls._metrics(adjusted, validation_y, classes)
            candidates.append((metrics["brier_score"], fitted, metrics))
        _, best, after = min(candidates, key=lambda item: item[0])
        report = {
            "fit_rows": split,
            "validation_rows": len(targets) - split,
            "method": best["method"],
            "before": before,
            "after": after,
            "brier_improvement": before["brier_score"] - after["brier_score"],
        }
        if report["brier_improvement"] <= 0:
            return None, report
        best["classes"] = list(classes)
        return best, report

    @staticmethod
    def _fit_one_vs_rest(
        probabilities: np.ndarray, targets: np.ndarray, classes: list[int], method: str
    ) -> dict[str, Any]:
        estimators = []
        for index, label in enumerate(classes):
            binary_target = (targets == label).astype(int)
            if binary_target.min() == binary_target.max():
                raise ValueError(f"La clase {label} no tiene ejemplos positivos y negativos.")
            if method == "sigmoid":
                estimator = LogisticRegression(C=1e6, solver="lbfgs")
                estimator.fit(probabilities[:, [index]], binary_target)
            else:
                estimator = IsotonicRegression(out_of_bounds="clip")
                estimator.fit(probabilities[:, index], binary_target)
            estimators.append(estimator)
        return {"method": method, "estimators": estimators}

    @staticmethod
    def _apply(bundle: dict[str, Any], probabilities: np.ndarray, classes: list[int]) -> np.ndarray:
        adjusted = []
        for index, estimator in enumerate(bundle["estimators"]):
            values = probabilities[:, index]
            if bundle["method"] == "sigmoid":
                adjusted.append(estimator.predict_proba(values.reshape(-1, 1))[:, 1])
            else:
                adjusted.append(estimator.predict(values))
        matrix = np.column_stack(adjusted)
        totals = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, totals, out=probabilities.copy(), where=totals > 0)

    @classmethod
    def _metrics(cls, probabilities: np.ndarray, targets: np.ndarray, classes: list[int]) -> dict[str, Any]:
        expected = np.column_stack([(targets == label).astype(float) for label in classes])
        if len(classes) == 2:
            brier = float(np.mean((probabilities[:, 1] - expected[:, 1]) ** 2))
        else:
            brier = float(np.mean(np.sum((probabilities - expected) ** 2, axis=1)))
        curves = {}
        for index, label in enumerate(classes):
            observed, predicted = calibration_curve(
                expected[:, index], probabilities[:, index], n_bins=8, strategy="quantile"
            )
            curves[str(label)] = {
                "predicted": predicted.tolist(), "observed": observed.tolist()
            }
        return {"brier_score": brier, "calibration_curve": curves}

    def calibrate(self, market: str, probabilities: np.ndarray, classes: list[int]) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float).reshape(1, -1)
        bundle = self.markets.get(market)
        if not bundle or list(bundle.get("classes", [])) != list(classes):
            return values[0]
        return self._apply(bundle, values, classes)[0]

    def save(self, path: Path, report_path: Path, reports: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        report_path.write_text(json.dumps({
            "base_model_version": self.base_model_version,
            "markets": reports,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_compatible(cls, path: Path, metadata: dict[str, Any]) -> FootballProbabilityCalibrator | None:
        if not path.exists():
            return None
        try:
            calibrator = joblib.load(path)
            if not isinstance(calibrator, cls):
                return None
            return calibrator if calibrator.base_model_version == cls.model_version(metadata) else None
        except Exception:
            return None
