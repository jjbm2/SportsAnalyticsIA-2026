from __future__ import annotations

from typing import Any


class Formula1Features:
    @staticmethod
    def feature_columns() -> list[str]:
        return [
            "avg_position_3", "avg_position_5", "avg_position_10",
            "position_std_5", "podium_rate_5", "top10_rate_5",
            "avg_points_5", "starts_strength", "constructor_avg_position_5",
            "circuit_avg_position", "season_progress",
        ]

    @classmethod
    def build(
        cls,
        driver_history: list[dict[str, float]],
        constructor_history: list[float],
        circuit_history: list[float],
        round_number: int,
        total_rounds: int = 24,
    ) -> dict[str, float]:
        positions = [float(item["position"]) for item in driver_history] or [12.0]
        points = [float(item.get("points", 0.0)) for item in driver_history] or [0.0]

        def average(values: list[float], window: int) -> float:
            recent = values[-window:]
            return sum(recent) / len(recent)

        recent_positions = positions[-5:]
        mean_recent = sum(recent_positions) / len(recent_positions)
        variance = sum((value - mean_recent) ** 2 for value in recent_positions) / len(recent_positions)
        constructor_recent = (constructor_history or [12.0])[-5:]
        circuit_recent = circuit_history or [mean_recent]
        return {
            "avg_position_3": average(positions, 3),
            "avg_position_5": average(positions, 5),
            "avg_position_10": average(positions, 10),
            "position_std_5": variance ** 0.5,
            "podium_rate_5": sum(value <= 3 for value in recent_positions) / len(recent_positions),
            "top10_rate_5": sum(value <= 10 for value in recent_positions) / len(recent_positions),
            "avg_points_5": average(points, 5),
            "starts_strength": min(len(driver_history), 40) / 40,
            "constructor_avg_position_5": sum(constructor_recent) / len(constructor_recent),
            "circuit_avg_position": sum(circuit_recent) / len(circuit_recent),
            "season_progress": min(max(round_number / max(total_rounds, 1), 0.0), 1.0),
        }

    @staticmethod
    def result_identity(result: dict[str, Any]) -> tuple[str | None, str | None]:
        driver = result.get("Driver") or {}
        constructor = result.get("Constructor") or {}
        return driver.get("driverId"), constructor.get("constructorId")
