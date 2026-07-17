from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from billing.billing_manager import BillingManager
from core.plans import PLANS, Plan, get_plan, plan_amount
from promotions.promotion_service import PromotionService
from usage.usage_tracker import UsageTracker


DEFAULT_BANK = "Mercado Pago"
DEFAULT_HOLDER = "Jose de Jesus Bernal Muñoz"
DEFAULT_CLABE = "722969010772233909"


def render_account_screen(
    user: dict,
    billing: BillingManager,
    usage: UsageTracker,
    sports: list[str],
) -> None:
    st.title("Mi perfil")
    st.caption("Consulta tu acceso, uso diario y solicitudes de suscripción.")
    subscription = billing.active_subscription(user["id"])
    plan = get_plan("full" if user.get("is_admin") else (subscription["plan"] if subscription else "free"))
    usage_rows = [usage.can_user_predict(user["id"], sport) | {"sport": sport} for sport in sports]
    promotion = PromotionService().status(user["id"])

    _render_profile_dashboard(user, plan, subscription, usage_rows)
    _render_opening_promotion(user, promotion)
    _render_daily_usage(usage_rows)
    _render_payment_statuses(billing.list_requests(user_id=user["id"]))

    st.subheader("Planes y suscripciones")
    st.caption("Elige pago mensual o ahorra 10% con el plan anual.")
    plan_items = list(PLANS.items())
    for start in range(0, len(plan_items), 2):
        columns = st.columns(2)
        for column, (code, offered_plan) in zip(columns, plan_items[start:start + 2]):
            with column:
                _render_pricing_card(code, offered_plan, plan.code)

    selected_plan = st.session_state.get("checkout_plan")
    if selected_plan:
        _render_checkout(user, billing, selected_plan, st.session_state.get("checkout_cycle", "monthly"))


def _render_profile_dashboard(
    user: dict,
    plan: Plan,
    subscription: dict | None,
    usage_rows: list[dict],
) -> None:
    used_today = sum(int(item["used"]) for item in usage_rows)
    with st.container(border=True):
        st.markdown(":material/account_circle: **Usuario**")
        st.write(user["email"])
        st.caption(f"Plan {plan.name} · {'Superusuario' if user.get('is_admin') else 'Cuenta activa'}")

    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.markdown(":material/query_stats: **Uso del día**")
        st.metric("Predicciones realizadas", used_today)
        st.caption("El límite se calcula de forma independiente para cada deporte.")
    with right.container(border=True, height="stretch"):
        st.markdown(":material/diamond: **Plan actual**")
        st.metric(plan.name, _limit_label(plan))
        for benefit in plan.benefits:
            st.write(f":material/check_circle: {benefit}")
        if subscription and not user.get("is_admin"):
            st.caption(f'Activo hasta {subscription["end_date"].strftime("%d/%m/%Y")}.')


def _render_daily_usage(rows: list[dict]) -> None:
    st.subheader("Uso por deporte")
    for start in range(0, len(rows), 2):
        columns = st.columns(2)
        for column, item in zip(columns, rows[start:start + 2]):
            limit = item["limit"]
            available = "Ilimitado" if limit is None else f'{item["used"]} de {limit + item["extra"]}'
            with column.container(border=True):
                st.markdown(f"**{item['sport']}**")
                st.metric("Uso de hoy", available)
                if limit is not None:
                    st.progress(min(1.0, item["used"] / max(1, limit + item["extra"])))


def _render_opening_promotion(user: dict, promotion: dict) -> None:
    if user.get("is_admin"):
        return
    if promotion.get("active"):
        with st.container(border=True):
            st.markdown(":green-badge[Promoción activa] **Apertura SportsAnalyticsAI**")
            st.write("Tienes 5 predicciones diarias por deporte durante el periodo promocional.")
            st.caption(
                f"Quedan {promotion['days_remaining']} días · termina el "
                f"{promotion['ends_on'].strftime('%d/%m/%Y')}."
            )
        return
    if promotion.get("expired"):
        with st.container(border=True):
            st.markdown("### Continúa con Basic")
            st.write("Tu promoción terminó. Conserva 5 predicciones diarias por deporte con Basic.")
            if st.button(
                "Elegir plan Basic",
                type="primary",
                icon=":material/arrow_forward:",
                key="expired_promo_basic",
            ):
                _select_checkout("basic", "monthly")
        return
    if user.get("plan") != "free":
        return

    with st.container(border=True):
        st.markdown("### Promoción de apertura")
        st.write("Activa 5 predicciones al día por deporte durante 5 días.")
        with st.form("opening_promotion_form", border=False):
            code = st.text_input("Código promocional", max_chars=40)
            submitted = st.form_submit_button(
                "Activar promoción", type="primary", icon=":material/redeem:"
            )
        if submitted:
            try:
                PromotionService().redeem(user["id"], code)
                st.success("Promoción activada. Ya tienes 5 predicciones diarias por deporte.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))


def _render_pricing_card(code: str, plan: Plan, current_code: str) -> None:
    badge = " :green-badge[Recomendado]" if code == "pro" else " :blue-badge[Premium]" if code == "full" else ""
    with st.container(border=True, height="stretch"):
        st.markdown(f"### {plan.name}{badge}")
        st.markdown(f"**${plan.monthly_price_mxn:,} MXN** / mes")
        st.caption(f"Anual: ${plan.yearly_price_mxn:,} MXN · ahorro del 10%")
        st.write(f":material/speed: {_limit_label(plan)}")
        for benefit in plan.benefits:
            st.write(f":material/check: {benefit}")
        if code == "free":
            st.button(
                "Plan actual" if current_code == code else "Incluido",
                disabled=True,
                width="stretch",
                key="pricing_free",
            )
        else:
            with st.container(horizontal=True):
                if st.button("Seleccionar mensual", key=f"monthly_{code}", width="stretch"):
                    _select_checkout(code, "monthly")
                if st.button("Elegir anual", key=f"yearly_{code}", type="primary", width="stretch"):
                    _select_checkout(code, "yearly")


def _select_checkout(plan: str, cycle: str) -> None:
    st.session_state["checkout_plan"] = plan
    st.session_state["checkout_cycle"] = cycle
    st.rerun()


def _render_checkout(user: dict, billing: BillingManager, selected_plan: str, cycle: str) -> None:
    plan = get_plan(selected_plan)
    amount = plan_amount(selected_plan, cycle)
    bank = os.getenv("TRANSFER_BANK_NAME", DEFAULT_BANK)
    holder = os.getenv("TRANSFER_ACCOUNT_HOLDER", DEFAULT_HOLDER)
    clabe = os.getenv("TRANSFER_CLABE", DEFAULT_CLABE)

    st.subheader("Completa tu pago")
    with st.container(border=True):
        st.markdown(
            f"**{plan.name} · {'Anual' if cycle == 'yearly' else 'Mensual'}**"
        )
        st.metric("Total a transferir", f"${amount:,} MXN")
        st.write(f"Banco: **{bank}**")
        st.write(f"Beneficiario: **{holder}**")
        st.caption("CLABE · usa el icono de copiar")
        st.code(clabe, language=None)
        st.info(
            "Realiza la transferencia por el importe exacto y sube tu comprobante. "
            "Tu acceso se activará después de la revisión manual.",
            icon=":material/info:",
        )

        proof = st.file_uploader(
            "Sube tu comprobante",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key=f"proof_{selected_plan}_{cycle}",
            help="PDF, PNG o JPG. Tamaño máximo: 5 MB.",
        )
        with st.container(horizontal=True):
            if st.button("Cancelar", key="cancel_checkout"):
                st.session_state["checkout_plan"] = None
                st.rerun()
            submitted = st.button(
                "Enviar para revisión",
                type="primary",
                icon=":material/upload_file:",
                disabled=proof is None,
                key="submit_payment",
            )
        if submitted and proof:
            try:
                billing.create_payment_request(
                    user["id"], selected_plan, cycle,
                    proof.getvalue(), Path(proof.name).suffix,
                )
                st.session_state["checkout_plan"] = None
                st.success("Comprobante recibido. Tu pago está en revisión.")
            except ValueError as error:
                st.error(str(error))


def _render_payment_statuses(requests: list[dict]) -> None:
    if not requests:
        return
    st.subheader("Estado de pagos")
    labels = {
        "pending": ("Pago en revisión", "orange"),
        "approved": ("Pago aprobado", "green"),
        "rejected": ("Pago rechazado", "red"),
    }
    for request in requests[:5]:
        label, color = labels.get(request["status"], ("Estado desconocido", "gray"))
        with st.container(border=True):
            st.markdown(f":{color}-badge[{label}] **{request['plan'].upper()}**")
            st.caption(
                f"{'Anual' if request['billing_cycle'] == 'yearly' else 'Mensual'} · "
                f"${request['amount']:,.0f} MXN · {request['created_at'].strftime('%d/%m/%Y %H:%M')}"
            )


def _limit_label(plan: Plan) -> str:
    if plan.daily_predictions_per_sport is None:
        return "Predicciones ilimitadas"
    return f"{plan.daily_predictions_per_sport} por día y deporte"
