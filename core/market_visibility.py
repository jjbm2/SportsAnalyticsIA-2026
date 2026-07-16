from __future__ import annotations

from typing import Any


ALWAYS_VISIBLE_MARKETS = {"home_win", "draw", "away_win"}


def visible_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return strong public signals without changing the persisted market set."""
    visible = []
    for market in markets:
        try:
            probability = float(market.get("probability"))
        except (TypeError, ValueError):
            continue
        if market.get("market_type") in ALWAYS_VISIBLE_MARKETS or probability >= 50.0:
            visible.append({**market, "probability": probability})
    return sorted(
        visible,
        key=lambda item: (
            str((item.get("extra_data_json") or {}).get("category") or "Otros"),
            -item["probability"],
        ),
    )
