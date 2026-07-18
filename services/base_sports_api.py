import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from threading import Lock
from dotenv import load_dotenv

from core.paths import CACHE_DIR
from services.http_client import build_retry_session
from services.api_sports_rate_limit import api_sports_rate_limiter

load_dotenv()


_CACHE_LOCKS: dict[str, Lock] = {}
_CACHE_LOCKS_GUARD = Lock()


def _cache_lock(cache_file: Path) -> Lock:
    key = str(cache_file.resolve())
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(key, Lock())


class ProviderResponseError(RuntimeError):
    """Raised when a provider returns HTTP 200 with an application error."""

    def __init__(self, message: str, *, reason: str = "provider_error") -> None:
        super().__init__(message)
        self.reason = reason


def classify_provider_error(errors: Any) -> str:
    """Classify provider errors without propagating sensitive response text."""
    text = str(errors or "").lower()
    if "suspend" in text:
        return "account_suspended"
    if "plan" in text or "subscription" in text or "season" in text:
        return "plan_restriction"
    if any(token in text for token in ("rate", "limit", "quota", "request")):
        return "quota_exceeded"
    if any(token in text for token in ("access", "key", "token", "auth", "permission")):
        return "credential_rejected"
    return "provider_error"


class BaseSportsAPI:
    def __init__(self, base_url: str, sport_name: str, require_api_key: bool = True):
        self.api_key = (os.getenv("API_SPORTS_KEY") or "").strip()

        if require_api_key and not self.api_key:
            raise ValueError("Falta configurar API_SPORTS_KEY")

        self.base_url = base_url.rstrip("/")
        self.sport_name = sport_name.lower()

        self.headers = {"x-apisports-key": self.api_key} if self.api_key else {}
        self.http = build_retry_session()

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
        max_hours: float = 24
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

        with _cache_lock(cache_file):
            if not force_refresh:
                cached_data = self._read_cache(cache_file=cache_file, max_hours=max_hours)
                if cached_data is not None:
                    return cached_data

            try:
                if self.base_url.endswith("api-sports.io"):
                    api_sports_rate_limiter.wait_for_slot()
                response = self.http.get(
                    f"{self.base_url}/{endpoint.lstrip('/')}",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                if self.base_url.endswith("api-sports.io"):
                    api_sports_rate_limiter.observe(response.headers, response.status_code)
                response.raise_for_status()
            except requests.RequestException:
                stale_data = self._read_cache(cache_file=cache_file, max_hours=None)
                if stale_data is not None:
                    stale_data["_source"] = "stale_cache"
                    return stale_data
                raise

            data = response.json()
            if data.get("errors"):
                stale_data = self._read_cache(cache_file=cache_file, max_hours=None)
                if stale_data is not None:
                    stale_data["_source"] = "stale_cache"
                    return stale_data
                raise ProviderResponseError(
                    f"{self.sport_name} provider rejected the request",
                    reason=classify_provider_error(data.get("errors")),
                )
            data["_source"] = "api"

            self._save_cache(
                cache_file=cache_file,
                data=data
            )

            return data
