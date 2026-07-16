from __future__ import annotations

import re
import unicodedata
from typing import Any


PRIMARY_LEAGUE_TERMS: dict[str, tuple[str, ...]] = {
    "Fútbol": (
        "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
        "liga mx", "eredivisie", "primeira liga", "champions league",
        "europa league", "conference league", "copa libertadores",
        "copa sudamericana", "world cup", "copa mundial", "euro championship",
        "uefa euro", "copa america", "nations league", "club world cup",
        "mundial de clubes", "international friendly", "friendlies", "friendly",
        "amistosos", "amistoso", "women s super league", "liga f", "nwsl",
        "frauen bundesliga", "division 1 feminine", "serie a women",
        "women s champions league", "women s world cup", "women s euro",
        "copa america femenina",
    ),
    "Basketball": (
        "nba", "wnba", "euroleague", "euro league", "ncaa",
        "summer league", "las vegas summer league", "california classic",
        "salt lake city summer league",
    ),
    "Béisbol": (
        "mlb", "major league baseball", "lmb", "liga mexicana de beisbol",
        "liga mexicana del pacifico", "npb", "nippon professional baseball",
        "kbo", "korea baseball organization", "world baseball classic",
        "serie del caribe", "caribbean series", "spring training",
        "cactus league", "grapefruit league", "preseason", "pre season",
    ),
    "NFL": (
        "nfl", "national football league", "ncaa", "college football",
        "college", "preseason", "pre season",
    ),
    "Fórmula 1": (
        "formula 1", "f1", "grand prix",
    ),
    "Hockey": (
        "nhl", "national hockey league", "ahl", "khl", "shl",
        "liiga", "del", "iihf", "world championship", "olympic",
        "champions hockey league",
    ),
    "MMA": (
        "mma", "ufc", "ultimate fighting championship", "bellator",
        "pfl", "one championship",
    ),
}


def normalize_league_text(value: Any) -> str:
    """Normaliza texto para comparaciones locales tolerantes a formato."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def is_primary_league(sport: str, league_name: Any, country: Any = "") -> bool:
    searchable = normalize_league_text(f"{league_name} {country}")
    return any(
        normalize_league_text(term) in searchable
        for term in PRIMARY_LEAGUE_TERMS.get(sport, ())
    )


def filter_games_by_league_view(
    games: list[dict[str, Any]],
    sport: str,
    view: str,
) -> list[dict[str, Any]]:
    """Filtra juegos ya cargados; nunca consulta servicios externos."""
    visible = list(games) if view == "Todas" else [
        game for game in games
        if is_primary_league(sport, game.get("league"), game.get("country"))
    ]
    return sorted(
        visible,
        key=lambda game: (
            0 if is_primary_league(sport, game.get("league"), game.get("country")) else 1,
            normalize_league_text(game.get("league")),
            normalize_league_text(game.get("label")),
        ),
    )
