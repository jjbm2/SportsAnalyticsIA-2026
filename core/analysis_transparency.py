from __future__ import annotations

from typing import Any


def classify_analysis(model_name: str, context: dict[str, Any] | None = None) -> str:
    """Clasifica el método ejecutado usando su control de calidad real."""
    normalized_name = str(model_name or "").lower()
    quality_gate = (context or {}).get("quality_gate")
    if "ml" in normalized_name and isinstance(quality_gate, dict):
        return "hybrid_ai" if any(bool(value) for value in quality_gate.values()) else "statistical"
    return "hybrid_ai" if "ml" in normalized_name else "statistical"


def analysis_method_copy(analysis_type: str) -> tuple[str, str]:
    if analysis_type == "hybrid_ai":
        return "IA híbrida validada", "Combina modelos entrenados con simulación estadística."
    return "Modelo estadístico", "Utiliza datos históricos y simulación; no se presenta como IA entrenada."
