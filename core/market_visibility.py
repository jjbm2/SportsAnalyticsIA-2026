from __future__ import annotations

from typing import Any


CATEGORY_ORDER = {
    "Resultado": 0,
    "Doble oportunidad": 1,
    "Total de goles": 2,
    "Ambos anotan": 3,
    "Goles por equipo": 4,
    "Portería en cero": 5,
}
MARKET_ORDER = {"home_win": 0, "draw": 1, "away_win": 2}


def visible_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every valid market, ordered with 1X2 and primary groups first."""
    visible = []
    for market in markets:
        try:
            probability = float(market.get("probability"))
        except (TypeError, ValueError):
            continue
        if 0.0 <= probability <= 100.0:
            visible.append({**market, "probability": probability})
    return sorted(
        visible,
        key=lambda item: (
            CATEGORY_ORDER.get(
                str((item.get("extra_data_json") or {}).get("category") or "Otros"),
                99,
            ),
            MARKET_ORDER.get(str(item.get("market_type") or ""), 99),
            -item["probability"],
        ),
    )
