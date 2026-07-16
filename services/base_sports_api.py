import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from core.paths import CACHE_DIR

load_dotenv()


class BaseSportsAPI:
    def __init__(self, base_url: str, sport_name: str, require_api_key: bool = True):
        self.api_key = (os.getenv("API_SPORTS_KEY") or "").strip()

        if require_api_key and not self.api_key:
            raise ValueError("Falta configurar API_SPORTS_KEY")

        self.base_url = base_url.rstrip("/")
        self.sport_name = sport_name.lower()

        self.headers = {"x-apisports-key": self.api_key} if self.api_key else {}

        self.cache_dir = CACHE_DIR / self.sport_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _safe_cache_name(self, value: str) -> str:
        return (
            value.replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
            .replace(" ", "_")
        )

    def _cache_path(
        self,
        endpoint: str,
        cache_key: str
    ) -> Path:
        endpoint_name = endpoint.replace("/", "_")
        safe_key = self._safe_cache_name(cache_key)

        return self.cache_dir / f"{endpoint_name}_{safe_key}.json"

    def _read_cache(
        self,
        cache_file: Path,
        max_hours: int | None,
    ) -> dict[str, Any] | None:
        if not cache_file.exists():
            return None

        modified = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - modified

        if max_hours is not None and age > timedelta(hours=max_hours):
            return None

        try:
            with cache_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            data["_source"] = "cache"
            return data

        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(
        self,
        cache_file: Path,
        data: dict[str, Any]
    ) -> None:
        with cache_file.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        cache_key: str = "default",
        force_refresh: bool = False,
        max_hours: int = 24
    ) -> dict[str, Any]:
        params = params or {}

        if not self.api_key:
            raise ValueError("Proveedor API-Sports no configurado")

        cache_file = self._cache_path(
            endpoint=endpoint,
            cache_key=cache_key
        )

        if not force_refresh:
            cached_data = self._read_cache(
                cache_file=cache_file,
                max_hours=max_hours
            )

            if cached_data is not None:
                return cached_data

        try:
            response = requests.get(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
        except requests.RequestException:
            stale_data = self._read_cache(cache_file=cache_file, max_hours=None)
            if stale_data is not None:
                stale_data["_source"] = "stale_cache"
                return stale_data
            raise

        data = response.json()
        data["_source"] = "api"

        self._save_cache(
            cache_file=cache_file,
            data=data
        )

        return data
