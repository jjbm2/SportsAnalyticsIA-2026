from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from core.paths import CACHE_DIR


def cleanup_expired_cache(
    max_age_days: int = 14,
    max_size_mb: int = 150,
    now: datetime | None = None,
) -> int:
    """Delete only rebuildable JSON cache, bounded by age and disk budget."""
    if max_age_days < 1 or max_size_mb < 1 or not CACHE_DIR.exists():
        return 0

    cutoff = (now or datetime.now()) - timedelta(days=max_age_days)
    deleted = 0

    retained: list[Path] = []
    for cache_file in CACHE_DIR.rglob("*.json"):
        if not _is_inside_cache(cache_file):
            continue
        try:
            modified = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if modified < cutoff:
                cache_file.unlink()
                deleted += 1
            else:
                retained.append(cache_file)
        except OSError:
            continue

    budget_bytes = max_size_mb * 1024 * 1024
    sized_files: list[tuple[float, int, Path]] = []
    for cache_file in retained:
        try:
            stat = cache_file.stat()
            sized_files.append((stat.st_mtime, stat.st_size, cache_file))
        except OSError:
            continue

    current_size = sum(item[1] for item in sized_files)
    for _, file_size, cache_file in sorted(sized_files):
        if current_size <= budget_bytes:
            break
        try:
            cache_file.unlink()
            current_size -= file_size
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
