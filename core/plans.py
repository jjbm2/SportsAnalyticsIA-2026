from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    code: str
    name: str
    monthly_price_mxn: int
    daily_predictions_per_sport: int | None
    benefits: tuple[str, ...]

    @property
    def yearly_price_mxn(self) -> int:
        return round(self.monthly_price_mxn * 12 * 0.90)


PLANS = {
    "free": Plan("free", "Free", 0, 1, ("1 predicción diaria por deporte", "Acceso a todos los deportes")),
    "basic": Plan("basic", "Basic", 300, 5, ("5 predicciones diarias por deporte", "Historial personal")),
    "pro": Plan("pro", "Pro", 500, 10, ("10 predicciones diarias por deporte", "Ideal para análisis frecuente")),
    "full": Plan("full", "Full", 1000, None, ("Predicciones ilimitadas", "Acceso completo sin límites")),
}
BILLING_CYCLES = {"monthly", "yearly"}


def get_plan(code: str) -> Plan:
    try:
        return PLANS[str(code).lower()]
    except KeyError as exc:
        raise ValueError("Plan no válido") from exc


def plan_amount(code: str, billing_cycle: str) -> int:
    plan = get_plan(code)
    if billing_cycle not in BILLING_CYCLES:
        raise ValueError("Ciclo de facturación no válido")
    return plan.yearly_price_mxn if billing_cycle == "yearly" else plan.monthly_price_mxn
