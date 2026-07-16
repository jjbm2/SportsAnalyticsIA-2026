from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database.model_metrics_repository import ModelMetricsRepository
from machine_learning.model_promotion import evaluate_promotion


MODEL_FILES = {
    "football": {
        "result_model.joblib": "football_result_model.joblib",
        "over25_model.joblib": "football_over25_model.joblib",
        "btts_model.joblib": "football_btts_model.joblib",
        "metadata.json": "football_metadata.json",
    },
    "baseball": {
        "home_win_model.joblib": "baseball_home_win_model.joblib",
        "over85_model.joblib": "baseball_over85_model.joblib",
        "home_over35_model.joblib": "baseball_home_over35_model.joblib",
        "metadata.json": "baseball_metadata.json",
    },
    "basketball": {
        "home_win_model.joblib": "basketball_home_win_model.joblib",
        "over_2195_model.joblib": "basketball_over_2195_model.joblib",
        "home_over_1095_model.joblib": "basketball_home_over_1095_model.joblib",
        "metadata.json": "basketball_metadata.json",
    },
    "nfl": {
        "home_win_model.joblib": "nfl_home_win_model.joblib",
        "over_415_model.joblib": "nfl_over_415_model.joblib",
        "home_over_205_model.joblib": "nfl_home_over_205_model.joblib",
        "metadata.json": "nfl_metadata.json",
    },
    "formula1": {
        "podium_model.joblib": "formula1_podium_model.joblib",
        "metadata.json": "formula1_metadata.json",
    },
}

SPORT_LABELS = {
    "football": "Fútbol",
    "baseball": "Béisbol",
    "basketball": "Basketball",
    "nfl": "NFL",
    "formula1": "Fórmula 1",
}


class ModelRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("machine_learning/models_store")

    def list_versions(self, sport: str) -> list[dict[str, Any]]:
        self._validate_sport(sport)
        active = self.active_version(sport)
        versions = []
        for path in (self.root / "versions" / sport).glob("*/metadata.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            version = str(metadata.get("model_version") or path.parent.name)
            versions.append({
                "version": version,
                "status": metadata.get("status", "unknown"),
                "active": version == active,
                "path": str(path.parent),
            })
        return sorted(versions, key=lambda item: item["version"], reverse=True)

    def active_version(self, sport: str) -> str:
        self._validate_sport(sport)
        path = self.root / MODEL_FILES[sport]["metadata.json"]
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("model_version") or "legacy")
        except (OSError, ValueError):
            return "legacy"

    def recommendation(self, sport: str, version: str) -> dict[str, Any]:
        active, candidate = ModelMetricsRepository().get_paired_model_performance(
            SPORT_LABELS[sport], self.active_version(sport), str(version)
        )
        return evaluate_promotion(active, candidate)

    def safe_automatic_recommendation(self, sport: str, version: str) -> dict[str, Any]:
        """Require AUC, Brier and calibration evidence before auto-promotion."""
        self._validate_sport(sport)
        candidate_path = self.root / "versions" / sport / str(version) / "metadata.json"
        active_path = self.root / MODEL_FILES[sport]["metadata.json"]
        try:
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            active = json.loads(active_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._safe_decision("insufficient_data", "Faltan metadatos comparables.")

        comparisons = self._comparable_market_metrics(active, candidate)
        if not comparisons:
            return self._safe_decision(
                "insufficient_data",
                "Faltan AUC, Brier o calibración comparables por mercado.",
            )
        improvements = []
        for market, active_metrics, candidate_metrics in comparisons:
            auc_gain = candidate_metrics["roc_auc"] - active_metrics["roc_auc"]
            brier_change = candidate_metrics["brier_score"] - active_metrics["brier_score"]
            calibration_change = (
                candidate_metrics["calibration_error"]
                - active_metrics["calibration_error"]
            )
            improvements.append({
                "market": market,
                "auc_gain": auc_gain,
                "brier_change": brier_change,
                "calibration_change": calibration_change,
            })
        safe = any(
            item["auc_gain"] >= 0.02
            and item["brier_change"] <= 0.0
            and item["calibration_change"] <= 0.0
            for item in improvements
        ) and all(
            item["brier_change"] <= 0.0 and item["calibration_change"] <= 0.0
            for item in improvements
        )
        return {
            "decision": "promote" if safe else "hold",
            "reason": (
                "El candidato supera AUC sin degradar Brier ni calibración."
                if safe else
                "El candidato no cumple simultáneamente AUC, Brier y calibración."
            ),
            "comparisons": improvements,
            "automatic_change": safe,
        }

    def promote(self, sport: str, version: str, *, confirm: bool = False) -> Path:
        self._validate_sport(sport)
        if not confirm:
            raise PermissionError("La promoción requiere confirmación explícita.")
        decision = self.recommendation(sport, version)
        if decision["decision"] != "promote":
            raise ValueError(f'Promoción bloqueada: {decision["reason"]}')
        candidate = self.root / "versions" / sport / str(version)
        self._validate_candidate(sport, candidate)
        backup = self._backup_active(sport)
        try:
            self._install(sport, candidate)
        except Exception:
            self._restore(sport, backup)
            raise
        return backup

    def promote_automatically_if_safe(self, sport: str, version: str) -> Path:
        decision = self.safe_automatic_recommendation(sport, version)
        if decision["decision"] != "promote":
            raise ValueError(f'Promoción automática bloqueada: {decision["reason"]}')
        candidate = self.root / "versions" / sport / str(version)
        self._validate_candidate(sport, candidate)
        backup = self._backup_active(sport)
        try:
            self._install(sport, candidate)
        except Exception:
            self._restore(sport, backup)
            raise
        return backup

    def rollback(self, sport: str, backup: str | None = None, *, confirm: bool = False) -> Path:
        self._validate_sport(sport)
        if not confirm:
            raise PermissionError("La restauración requiere confirmación explícita.")
        base = self.root / "backups" / sport
        available = list(base.iterdir()) if base.exists() else []
        target = base / backup if backup else max((path for path in available if path.is_dir()), default=None)
        if target is None or not target.exists():
            raise FileNotFoundError("No existe un respaldo para restaurar.")
        self._restore(sport, target)
        return target

    def _backup_active(self, sport: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = self.root / "backups" / sport / timestamp
        backup.mkdir(parents=True, exist_ok=False)
        for active_name in MODEL_FILES[sport].values():
            source = self.root / active_name
            if source.exists():
                shutil.copy2(source, backup / active_name)
        return backup

    def _install(self, sport: str, candidate: Path) -> None:
        for candidate_name, active_name in MODEL_FILES[sport].items():
            source = candidate / candidate_name
            pending = self.root / f"{active_name}.pending"
            shutil.copy2(source, pending)
            os.replace(pending, self.root / active_name)
        metadata_path = self.root / MODEL_FILES[sport]["metadata.json"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "active"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def _restore(self, sport: str, backup: Path) -> None:
        for active_name in MODEL_FILES[sport].values():
            source = backup / active_name
            if source.exists():
                shutil.copy2(source, self.root / active_name)

    @staticmethod
    def _validate_candidate(sport: str, candidate: Path) -> None:
        missing = [name for name in MODEL_FILES[sport] if not (candidate / name).exists()]
        if missing:
            raise FileNotFoundError(f"Candidato incompleto: {', '.join(missing)}")
        metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "candidate_qualified":
            raise ValueError("La versión no está calificada como candidata.")

    @staticmethod
    def _validate_sport(sport: str) -> None:
        if sport not in MODEL_FILES:
            raise ValueError(f"Deporte no soportado: {sport}")

    @staticmethod
    def _safe_decision(decision: str, reason: str) -> dict[str, Any]:
        return {"decision": decision, "reason": reason, "comparisons": [], "automatic_change": False}

    @staticmethod
    def _comparable_market_metrics(
        active: dict[str, Any], candidate: dict[str, Any]
    ) -> list[tuple[str, dict[str, float], dict[str, float]]]:
        active_metrics = active.get("metrics") or {}
        candidate_metrics = candidate.get("metrics") or {}
        qualified = set(candidate.get("qualified_markets") or candidate_metrics.keys())
        comparisons = []
        for market in qualified:
            left = active_metrics.get(market)
            right = candidate_metrics.get(market)
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue
            required = ("roc_auc", "brier_score", "calibration_error")
            try:
                left_values = {key: float(left[key]) for key in required}
                right_values = {key: float(right[key]) for key in required}
            except (KeyError, TypeError, ValueError):
                continue
            comparisons.append((str(market), left_values, right_values))
        if comparisons:
            return comparisons

        # Football stores binary-market validation metrics in a flat legacy shape.
        prefixes = {"over_2_5": "over", "btts": "btts"}
        for market in qualified:
            prefix = prefixes.get(str(market))
            if not prefix:
                continue
            keys = {
                "roc_auc": f"{prefix}_auc",
                "brier_score": f"{prefix}_brier_score",
                "calibration_error": f"{prefix}_calibration_error",
            }
            try:
                left_values = {name: float(active_metrics[key]) for name, key in keys.items()}
                right_values = {name: float(candidate_metrics[key]) for name, key in keys.items()}
            except (KeyError, TypeError, ValueError):
                continue
            comparisons.append((str(market), left_values, right_values))
        return comparisons
