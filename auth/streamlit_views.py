from __future__ import annotations

import streamlit as st

from auth.auth_manager import AuthManager


def render_auth_screen(auth: AuthManager) -> None:
    if st.button("Volver", icon=":material/arrow_back:", key="back_to_landing"):
        st.session_state["public_screen"] = "landing"
        st.rerun()
    st.title("SportsAnalyticsAI")
    st.caption("Inicia sesión para analizar eventos y administrar tu plan.")
    mode = st.segmented_control(
        "Acceso",
        ["Iniciar sesión", "Crear cuenta"],
        default=st.session_state.get("auth_mode", "Iniciar sesión"),
        width="stretch",
    )
    if mode == "Crear cuenta":
        with st.form("register_form"):
            email = st.text_input("Correo electrónico", autocomplete="email")
            password = st.text_input("Contraseña", type="password", autocomplete="new-password")
            confirmation = st.text_input("Confirmar contraseña", type="password", autocomplete="new-password")
            submitted = st.form_submit_button("Crear cuenta", width="stretch", type="primary")
        if submitted:
            if password != confirmation:
                st.error("Las contraseñas no coinciden")
                return
            try:
                st.session_state["current_user"] = auth.register(email, password)
                st.session_state["screen"] = (
                    "account" if st.session_state.get("checkout_plan") else "home"
                )
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        return

    with st.form("login_form"):
        email = st.text_input("Correo electrónico", autocomplete="email")
        password = st.text_input("Contraseña", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Entrar", width="stretch", type="primary")
    if submitted:
        try:
            user = auth.authenticate(email, password)
        except ValueError:
            user = None
        if user is None:
            st.error("Correo o contraseña incorrectos")
            return
        st.session_state["current_user"] = user
        st.session_state["screen"] = "home"
        st.rerun()


def render_account_navigation(user: dict) -> None:
    with st.sidebar:
        st.markdown(f"**{user['email']}**")
        st.caption(f"Plan {str(user['plan']).upper()}")
        if st.button("Inicio", icon=":material/home:", width="stretch", key="saas_home"):
            st.session_state["screen"] = "home"
            st.rerun()
        if st.button("Mi cuenta", icon=":material/account_circle:", width="stretch", key="saas_account"):
            st.session_state["screen"] = "account"
            st.rerun()
        if user.get("is_admin") and st.button(
            "Administración", icon=":material/admin_panel_settings:", width="stretch", key="saas_admin"
        ):
            st.session_state["screen"] = "admin"
            st.rerun()
        if st.button("Cerrar sesión", icon=":material/logout:", width="stretch", key="saas_logout"):
            st.session_state["current_user"] = None
            st.session_state["screen"] = "home"
            st.session_state["public_screen"] = "landing"
            st.rerun()
