from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from core.paths import CACHE_DIR


def cleanup_expired_cache(
    max_age_days: int = 7,
    now: datetime | None = None,
) -> int:
    """Elimina únicamente respuestas JSON reconstruibles y antiguas."""
    if max_age_days < 1 or not CACHE_DIR.exists():
        return 0

    cutoff = (now or datetime.now()) - timedelta(days=max_age_days)
    deleted = 0

    for cache_file in CACHE_DIR.rglob("*.json"):
        if not _is_inside_cache(cache_file):
            continue
        try:
            modified = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if modified < cutoff:
                cache_file.unlink()
                deleted += 1
        except OSError:
            continue

    return deleted


def _is_inside_cache(path: Path) -> bool:
    try:
        path.resolve().relative_to(CACHE_DIR.resolve())
        return True
    except ValueError:
        return False
