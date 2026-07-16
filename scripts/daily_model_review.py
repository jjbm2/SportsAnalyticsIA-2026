from __future__ import annotations

import json
from typing import Any

from database.model_metrics_repository import ModelMetricsRepository
from machine_learning.model_registry import MODEL_FILES, ModelRegistry
from machine_learning.shadow_validation import ShadowValidationService
from services.post_match_service import PostMatchService


def run_daily_review() -> dict[str, Any]:
    new_reviews = PostMatchService().process_cached_results()
    metrics = ModelMetricsRepository().get_performance_summary()
    registry = ModelRegistry()
    shadow = ShadowValidationService()
    recommendations: dict[str, Any] = {}
    for sport_label, sport_key in shadow.sport_dirs.items():
        if sport_key not in MODEL_FILES:
            continue
        candidate = shadow.find_candidate(sport_label)
        if candidate is None:
            continue
        recommendations[sport_label] = {
            "candidate_version": candidate.name,
            **registry.recommendation(sport_key, candidate.name),
        }
    return {
        "new_reviews": new_reviews,
        "performance": metrics,
        "recommendations": recommendations,
        "models_changed": False,
    }


def main() -> None:
    print(json.dumps(run_daily_review(), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
