from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Callable

from services.api_manager import APIManager


def check_sports_connectivity(
    sports: list[str],
    *,
    fixture_date: str | None = None,
    manager_factory: Callable[[str], Any] = APIManager,
) -> list[dict[str, Any]]:
    """Run an explicit admin-only connectivity check without exposing secrets."""
    selected_date = fixture_date or date.today().isoformat()

    def check(sport: str) -> dict[str, Any]:
        try:
            payload = manager_factory(sport).get_games_by_date(
                date=selected_date,
                force_refresh=True,
            )
            provider_errors = payload.get("errors")
            if provider_errors:
                return _result(sport, "error", 0, "El proveedor rechazó la consulta")
            events = payload.get("response") or []
            warnings = payload.get("_provider_warnings") or []
            warning_reasons = {
                str(item.get("reason")) for item in warnings if isinstance(item, dict)
            }
            if not events and "account_suspended" in warning_reasons:
                return _result(sport, "error", 0, "La cuenta del proveedor está suspendida")
            if not events and "quota_exceeded" in warning_reasons:
                return _result(sport, "error", 0, "Límite de consultas alcanzado")
            if not events and "credential_rejected" in warning_reasons:
                return _result(sport, "error", 0, "Credencial rechazada o sin permisos")
            return _result(
                sport,
                "connected",
                len(events),
                "Eventos disponibles" if events else "Conectado, sin eventos para esta fecha",
            )
        except Exception as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            detail = "Proveedor temporalmente no disponible"
            reason = getattr(error, "reason", None)
            if reason == "account_suspended":
                detail = "La cuenta del proveedor está suspendida"
            elif reason == "quota_exceeded" or status_code == 429:
                detail = "Límite de consultas alcanzado"
            elif reason == "credential_rejected" or status_code in {401, 403}:
                detail = "Credencial rechazada o sin permisos"
            return _result(sport, "error", 0, detail)

    workers = max(1, min(4, len(sports)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check, sports))
    return results


def _result(sport: str, status: str, events: int, detail: str) -> dict[str, Any]:
    return {
        "sport": sport,
        "status": status,
        "events": int(events),
        "detail": detail,
    }
