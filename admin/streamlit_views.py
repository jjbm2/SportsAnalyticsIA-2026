from __future__ import annotations

import pandas as pd
import streamlit as st

from admin.admin_service import AdminService
from billing.billing_manager import BillingManager
from core.plans import PLANS
from core.system_health import deployment_health


def render_admin_screen(user: dict, admin: AdminService, billing: BillingManager, sports: list[str]) -> None:
    if not user.get("is_admin"):
        st.error("Acceso administrativo requerido")
        return
    st.title("Administración")
    dashboard_tab, users_tab, payments_tab, loyalty_tab = st.tabs(
        ["Dashboard", "Usuarios", "Pagos", "Lealtad"]
    )
    users = admin.list_users(user["id"])
    user_options = {f"{item['email']} · {item['plan'].upper()}": item for item in users}

    with dashboard_tab:
        metrics = admin.dashboard(user["id"])
        with st.container(horizontal=True):
            st.metric("Usuarios", metrics["total_users"], border=True)
            st.metric(
                "Ingresos aprobados",
                f'${metrics["approved_revenue_mxn"]:,.0f} MXN',
                border=True,
            )
            st.metric("Predicciones", metrics["predictions"], border=True)
            st.metric("Pagos pendientes", metrics["pending_payments"], border=True)

        st.subheader("Estado del sistema")
        health = deployment_health()
        with st.container(border=True):
            st.badge(
                health["database_backend"],
                icon=":material/database:",
                color="green" if health["database_ok"] else "red",
            )
            st.caption("La información mostrada no incluye claves ni credenciales.")
            with st.container(horizontal=True):
                st.metric(
                    "API-Sports",
                    "Configurada" if health["api_sports"] else "Pendiente",
                    border=True,
                )
                st.metric(
                    "Sportmonks",
                    "Configurada" if health["sportmonks"] else "Pendiente",
                    border=True,
                )
                st.metric(
                    "BallDontLie",
                    "Configurada" if health["balldontlie"] else "Pendiente",
                    border=True,
                )

    with users_tab:
        st.caption(f"{len(users)} usuarios registrados en la base de datos actual.")
        st.dataframe(pd.DataFrame(users), hide_index=True, width="stretch")
        selected_label = st.selectbox("Usuario", list(user_options), key="admin_plan_user")
        selected_plan = st.selectbox("Nuevo plan", list(PLANS), key="admin_plan")
        if st.button("Cambiar plan manualmente", key="admin_change_plan"):
            admin.change_plan(user["id"], user_options[selected_label]["id"], selected_plan)
            st.success("Plan actualizado")
            st.rerun()

    with payments_tab:
        st.subheader("Pagos pendientes")
        st.caption("Revisa el comprobante antes de activar cualquier suscripción.")
        pending = billing.list_requests(status="pending")
        if not pending:
            st.caption("No hay solicitudes pendientes.")
        for request in pending:
            with st.container(border=True):
                requester = next((item for item in users if item["id"] == request["user_id"]), None)
                st.write(
                    f"**{requester['email'] if requester else 'Usuario'}** · "
                    f"{request['plan'].upper()} {request['billing_cycle']} · ${request['amount']:,.0f} MXN"
                )
                if request.get("proof_uploaded_at"):
                    st.caption(
                        f'Comprobante recibido: {request["proof_uploaded_at"].strftime("%d/%m/%Y %H:%M")}'
                    )
                receipt = billing.get_receipt(request["id"], user["id"])
                if receipt:
                    st.download_button(
                        "Ver comprobante",
                        data=receipt["content"],
                        file_name=receipt["name"],
                        mime=receipt["mime"],
                        icon=":material/attachment:",
                        key=f"receipt_{request['id']}",
                    )
                else:
                    st.caption("Sin comprobante adjunto")
                with st.container(horizontal=True):
                    if st.button(
                        "Aprobar",
                        key=f"approve_{request['id']}",
                        type="primary",
                        icon=":material/check_circle:",
                    ):
                        billing.approve(request["id"], user["id"])
                        st.success("Pago aprobado, suscripción activada y comprobante eliminado")
                        st.rerun()
                    if st.button(
                        "Rechazar",
                        key=f"reject_{request['id']}",
                        icon=":material/cancel:",
                    ):
                        billing.reject(request["id"], user["id"])
                        st.rerun()

    with loyalty_tab:
        selected_label = st.selectbox("Usuario", list(user_options), key="loyalty_user")
        selected = user_options[selected_label]
        sport = st.selectbox("Deporte", sports, key="loyalty_sport")
        extra = st.number_input("Predicciones extra para hoy", min_value=1, max_value=100, value=1)
        if st.button("Otorgar predicciones", key="grant_predictions"):
            admin.grant_extra_predictions(user["id"], selected["id"], sport, int(extra))
            st.success("Predicciones extra otorgadas")
        days = st.number_input("Días para extender suscripción", min_value=1, max_value=365, value=30)
        if st.button("Extender suscripción", key="extend_subscription"):
            try:
                admin.extend_subscription(user["id"], selected["id"], int(days))
                st.success("Suscripción extendida")
            except ValueError as error:
                st.error(str(error))
