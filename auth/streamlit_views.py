from __future__ import annotations

import time
from collections.abc import MutableMapping

import streamlit as st

from auth.auth_manager import AuthManager
from auth.session_security import clear_authenticated_session, establish_authenticated_session


MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300
_LOGIN_ATTEMPTS_KEY = "auth_failed_attempts"
_LOGIN_LOCKOUT_KEY = "auth_lockout_until"


def login_lock_remaining(state: MutableMapping, now: float | None = None) -> int:
    current_time = time.time() if now is None else now
    lockout_until = float(state.get(_LOGIN_LOCKOUT_KEY, 0) or 0)
    remaining = max(0, int(lockout_until - current_time + 0.999))
    if remaining == 0 and lockout_until:
        state.pop(_LOGIN_LOCKOUT_KEY, None)
        state.pop(_LOGIN_ATTEMPTS_KEY, None)
    return remaining


def record_login_failure(state: MutableMapping, now: float | None = None) -> int:
    current_time = time.time() if now is None else now
    attempts = int(state.get(_LOGIN_ATTEMPTS_KEY, 0) or 0) + 1
    if attempts >= MAX_LOGIN_ATTEMPTS:
        state[_LOGIN_LOCKOUT_KEY] = current_time + LOGIN_LOCKOUT_SECONDS
        state[_LOGIN_ATTEMPTS_KEY] = 0
        return LOGIN_LOCKOUT_SECONDS
    state[_LOGIN_ATTEMPTS_KEY] = attempts
    return 0


def clear_login_failures(state: MutableMapping) -> None:
    state.pop(_LOGIN_ATTEMPTS_KEY, None)
    state.pop(_LOGIN_LOCKOUT_KEY, None)


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
            password = st.text_input(
                "Contraseña", type="password", autocomplete="new-password", max_chars=72
            )
            st.caption("Usa al menos 8 caracteres, una letra y un número.")
            confirmation = st.text_input(
                "Confirmar contraseña", type="password", autocomplete="new-password", max_chars=72
            )
            submitted = st.form_submit_button("Crear cuenta", width="stretch", type="primary")
        if submitted:
            if password != confirmation:
                st.error("Las contraseñas no coinciden")
                return
            try:
                user = auth.register(email, password)
                establish_authenticated_session(st.session_state, user)
                st.session_state["screen"] = (
                    "account" if st.session_state.get("checkout_plan") else "home"
                )
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        return

    with st.form("login_form"):
        email = st.text_input("Correo electrónico", autocomplete="email")
        password = st.text_input(
            "Contraseña", type="password", autocomplete="current-password", max_chars=72
        )
        submitted = st.form_submit_button("Entrar", width="stretch", type="primary")
    if submitted:
        remaining = login_lock_remaining(st.session_state)
        if remaining:
            st.error(f"Demasiados intentos. Intenta de nuevo en {remaining} segundos.")
            return
        try:
            user = auth.authenticate(email, password)
        except ValueError:
            user = None
        if user is None:
            lockout = record_login_failure(st.session_state)
            if lockout:
                st.error("Demasiados intentos. Intenta de nuevo en 5 minutos.")
            else:
                st.error("Correo o contraseña incorrectos")
            return
        clear_login_failures(st.session_state)
        establish_authenticated_session(st.session_state, user)
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
            clear_authenticated_session(st.session_state)
            st.rerun()
