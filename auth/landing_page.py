from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.plans import PLANS, Plan


def render_landing_page(logo_path: Path, app_name: str) -> None:
    _render_hero(logo_path, app_name)
    _render_intro()
    _render_benefits()
    _render_plans()
    _render_final_cta()
    _render_cookie_banner()
    _render_footer()


def _render_hero(logo_path: Path, app_name: str) -> None:
    with st.container(horizontal_alignment="center"):
        if logo_path.exists():
            st.image(str(logo_path), width=120)
        st.title(app_name, text_alignment="center")
        st.markdown(
            "### Predicciones deportivas inteligentes, explicables y en mejora constante",
            text_alignment="center",
        )
        st.caption(
            "Analiza partidos con modelos especializados, simulaciones y datos históricos desde una sola plataforma.",
            text_alignment="center",
        )
        with st.container(horizontal=True, horizontal_alignment="center"):
            if st.button(
                "Iniciar sesión",
                icon=":material/login:",
                width="content",
                key="landing_login",
            ):
                _open_auth("Iniciar sesión")
            if st.button(
                "Crear cuenta",
                type="primary",
                icon=":material/person_add:",
                width="content",
                key="landing_register",
            ):
                _open_auth("Crear cuenta")


def _render_intro() -> None:
    st.space("medium")
    st.header("Decisiones respaldadas por análisis", text_alignment="center")
    st.markdown(
        "SportsAnalyticsAI combina predicciones deportivas con IA, análisis estadístico avanzado "
        "y aprendizaje continuo. Cada resultado incluye probabilidades, confianza y contexto para "
        "que puedas interpretar la señal con claridad.",
        text_alignment="center",
    )


def _render_benefits() -> None:
    st.header("Todo lo necesario para analizar mejor", text_alignment="center")
    benefits = [
        (":material/psychology:", "IA inteligente", "Modelos especializados y controles de calidad antes de mostrar una señal."),
        (":material/analytics:", "Análisis detallado", "Probabilidades, confianza, mercados y explicaciones basadas en datos."),
        (":material/sports_soccer:", "Múltiples deportes", "Fútbol, basketball, béisbol, NFL y más desde una experiencia unificada."),
        (":material/model_training:", "Mejora constante", "Evaluación post-partido y aprendizaje controlado sin degradar modelos activos."),
    ]
    for start in range(0, len(benefits), 2):
        columns = st.columns(2)
        for column, (icon, title, description) in zip(columns, benefits[start:start + 2]):
            with column.container(border=True, height="stretch"):
                st.markdown(f"### {icon} {title}")
                st.write(description)


def _render_plans() -> None:
    st.header("Elige el plan que se adapta a ti", text_alignment="center")
    st.caption("Comienza gratis y cambia de plan cuando necesites más análisis.", text_alignment="center")
    items = list(PLANS.items())
    for start in range(0, len(items), 2):
        columns = st.columns(2)
        for column, (code, plan) in zip(columns, items[start:start + 2]):
            with column:
                _render_plan_card(code, plan)


def _render_plan_card(code: str, plan: Plan) -> None:
    badge = " :green-badge[Recomendado]" if code == "pro" else " :blue-badge[Premium]" if code == "full" else ""
    with st.container(border=True, height="stretch"):
        st.markdown(f"### {plan.name}{badge}")
        st.markdown(f"**${plan.monthly_price_mxn:,} MXN** / mes")
        st.caption(f"${plan.yearly_price_mxn:,} MXN al año · 10% de descuento")
        for benefit in plan.benefits:
            st.write(f":material/check_circle: {benefit}")
        if st.button(
            "Crear cuenta y elegir plan",
            type="primary" if code in {"pro", "full"} else "secondary",
            width="stretch",
            key=f"landing_plan_{code}",
        ):
            _open_auth("Crear cuenta", code)


def _render_final_cta() -> None:
    st.space("medium")
    with st.container(border=True, horizontal_alignment="center"):
        st.header("Tu próximo análisis empieza aquí", text_alignment="center")
        st.caption("Crea tu cuenta Free y realiza tu primera predicción.", text_alignment="center")
        if st.button(
            "Empieza gratis ahora",
            type="primary",
            icon=":material/rocket_launch:",
            key="landing_final_cta",
        ):
            _open_auth("Crear cuenta", "free")


def _render_cookie_banner() -> None:
    if st.session_state.get("cookies_accepted"):
        return
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.write(
                ":material/cookie: Usamos cookies esenciales de sesión para mantener tu acceso y preferencias durante la visita."
            )
            if st.button("Aceptar cookies", type="primary", key="accept_cookies"):
                st.session_state["cookies_accepted"] = True
                st.rerun()


def _render_footer() -> None:
    st.space("small")
    with st.container(horizontal=True, horizontal_alignment="center"):
        with st.popover("Términos y condiciones"):
            st.write(
                "Las predicciones expresan probabilidades y no garantizan resultados. "
                "El uso de la plataforma requiere una cuenta válida y el respeto de sus límites."
            )
        with st.popover("Política de privacidad"):
            st.write(
                "Protegemos las contraseñas con bcrypt y no mostramos información privada de otros usuarios. "
                "Los comprobantes se eliminan al concluir su revisión."
            )
        with st.popover("Aviso de cookies"):
            st.write(
                "La aplicación utiliza estado de sesión para autenticación, navegación, preferencias y consentimiento."
            )
    st.caption("© SportsAnalyticsAI · Análisis deportivo responsable", text_alignment="center")


def _open_auth(mode: str, plan: str | None = None) -> None:
    st.session_state["public_screen"] = "auth"
    st.session_state["auth_mode"] = mode
    st.session_state["checkout_plan"] = None if plan in {None, "free"} else plan
    st.session_state["checkout_cycle"] = "monthly"
    st.rerun()
