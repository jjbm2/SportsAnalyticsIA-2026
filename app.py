from __future__ import annotations

import importlib
import os
import random
from html import escape
from datetime import date, datetime
from typing import Any, Callable

import streamlit as st
from dotenv import load_dotenv

from core.analysis_transparency import analysis_method_copy, classify_analysis
from core.constants import (
    DEFAULT_SIMULATIONS,
)
from core.cache_cleanup import cleanup_expired_cache
from core.game_status import (
    extract_final_score,
    is_available_for_pregame,
    is_finished_status,
)
from core.league_filters import filter_games_by_league_view, is_primary_league
from core.market_visibility import visible_markets
from core.prediction_confidence import enrich_football_markets
from core.match_quality import calculate_match_quality
from core.game_style import apply_game_style_to_markets, classify_game_style
from core.logger import logger
from core.paths import LOGO_PATH
from core.version import PROJECT, VERSION
from database.database import create_database
from database.model_metrics_repository import ModelMetricsRepository
from database.post_match_review_repository import PostMatchReviewRepository
from database.prediction_repository import PredictionRepository
from database.seed import seed_sports
from services.api_manager import APIManager
from services.post_match_service import PostMatchService
from services.player_availability_service import PlayerAvailabilityService
from auth.auth_manager import AuthManager
from auth.streamlit_views import render_account_navigation, render_auth_screen
from auth.landing_page import render_landing_page
from usage.usage_tracker import UsageTracker
from billing.billing_manager import BillingManager
from billing.streamlit_views import render_account_screen
from admin.admin_service import AdminService
from admin.streamlit_views import render_admin_screen


load_dotenv()


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title=PROJECT,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# ESTILOS
# =========================================================

st.markdown(
    """
    <style>
        :root {
            --background: #090D14;
            --surface: #111722;
            --surface-secondary: #0B1F3A;
            --primary: #1E88E5;
            --success: #00C853;
            --warning: #FFB300;
            --danger: #FF5252;
            --text: #F5F7FA;
            --muted: #AAB7C4;
            --border: rgba(30, 136, 229, 0.35);
        }

        .stApp {
            background:
                radial-gradient(circle at top, rgba(30, 136, 229, 0.10), transparent 35%),
                var(--background);
        }

        .block-container {
            width: min(1120px, 100%);
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .app-loading-screen {
            min-height: 68vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.85rem;
            text-align: center;
            opacity: 0;
            animation: app-loading-reveal 0.18s ease 0.7s forwards;
        }

        .app-loading-mark {
            width: 62px;
            height: 62px;
            border-radius: 50%;
            border: 4px solid rgba(30, 136, 229, 0.18);
            border-top-color: var(--primary);
            animation: app-loading-spin 0.85s linear infinite;
            box-shadow: 0 0 32px rgba(30, 136, 229, 0.22);
        }

        .app-loading-title {
            color: var(--text);
            font-size: clamp(1.35rem, 3vw, 1.8rem);
            font-weight: 780;
            margin-top: 0.35rem;
        }

        .app-loading-copy {
            color: var(--muted);
            font-size: 0.95rem;
            max-width: 420px;
        }

        @keyframes app-loading-spin {
            to { transform: rotate(360deg); }
        }

        @keyframes app-loading-reveal {
            to { opacity: 1; }
        }

        @media (prefers-reduced-motion: reduce) {
            .app-loading-mark { animation-duration: 1.8s; }
            .app-loading-screen { animation-duration: 0.01s; }
        }

        .small-text {
            color: var(--muted);
            text-align: center;
            font-size: 0.88rem;
            margin-top: 0.25rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            color: var(--text);
            font-size: clamp(1.25rem, 2vw, 1.65rem);
            font-weight: 750;
            margin-top: 1.8rem;
            margin-bottom: 0.8rem;
        }

        .home-card {
            background: linear-gradient(145deg, rgba(17, 23, 34, 0.98), rgba(11, 31, 58, 0.95));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: clamp(20px, 4vw, 42px);
            box-shadow: 0 20px 55px rgba(0, 0, 0, 0.28);
            margin-bottom: 1.2rem;
        }

        .home-heading {
            color: var(--text);
            font-size: clamp(1.65rem, 4vw, 2.65rem);
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.4rem;
        }

        .home-description {
            color: var(--muted);
            font-size: clamp(0.95rem, 2vw, 1.1rem);
            text-align: center;
            margin-bottom: 0;
        }

        .sport-card {
            background: linear-gradient(145deg, rgba(17, 23, 34, 0.98), rgba(11, 31, 58, 0.95));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: clamp(18px, 4vw, 32px);
            text-align: center;
            margin-bottom: 1.2rem;
            box-shadow: 0 15px 45px rgba(0, 0, 0, 0.24);
        }

        .sport-logo {
            font-size: clamp(2.8rem, 7vw, 4.5rem);
            line-height: 1;
            margin-bottom: 0.5rem;
        }

        .sport-name {
            color: var(--text);
            font-size: clamp(1.6rem, 4vw, 2.4rem);
            font-weight: 800;
            margin: 0;
        }

        .sport-date {
            color: var(--muted);
            margin-top: 0.4rem;
        }

        .game-card {
            background: rgba(17, 23, 34, 0.96);
            border: 1px solid rgba(170, 183, 196, 0.18);
            border-radius: 18px;
            padding: clamp(15px, 3vw, 22px);
            margin-bottom: 12px;
            width: 100%;
            box-sizing: border-box;
        }

        .game-teams {
            font-size: clamp(1rem, 3vw, 1.35rem);
            font-weight: 750;
            color: var(--text);
            overflow-wrap: anywhere;
        }

        .game-info {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 6px;
            overflow-wrap: anywhere;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(17, 23, 34, 0.98), rgba(11, 31, 58, 0.82));
            padding: clamp(16px, 3vw, 25px);
            border-radius: 16px;
            text-align: center;
            border: 1px solid rgba(170, 183, 196, 0.16);
            min-height: 135px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-bottom: 0.65rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035), 0 12px 28px rgba(0, 0, 0, 0.18);
        }

        .metric-title {
            color: var(--muted);
            font-size: 0.95rem;
            overflow-wrap: anywhere;
        }

        .metric-value {
            color: var(--success);
            font-size: clamp(1.8rem, 5vw, 2.35rem);
            font-weight: 800;
            margin-top: 0.35rem;
        }

        .prediction-insight {
            background: linear-gradient(135deg, rgba(18, 36, 58, 0.96), rgba(13, 24, 38, 0.96));
            border: 1px solid rgba(103, 181, 255, 0.28);
            border-radius: 18px;
            padding: clamp(17px, 3vw, 24px);
            margin: 1rem 0 0.35rem;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.2);
        }

        .prediction-kicker {
            color: #67B5FF;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .prediction-pick {
            color: var(--text);
            font-size: clamp(1.15rem, 3vw, 1.55rem);
            font-weight: 800;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }

        .prediction-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.45rem;
        }

        .market-card {
            background: rgba(17, 23, 34, 0.92);
            border: 1px solid rgba(170, 183, 196, 0.16);
            border-left: 4px solid var(--warning);
            border-radius: 15px;
            padding: 15px 17px;
            margin-bottom: 0.7rem;
            min-height: 104px;
        }

        .market-card.high { border-left-color: var(--success); }
        .market-card.medium { border-left-color: var(--warning); }

        .market-category {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .market-selection {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.3rem;
            overflow-wrap: anywhere;
        }

        .market-probability {
            color: var(--warning);
            font-size: 1.35rem;
            font-weight: 850;
            margin-top: 0.3rem;
        }

        .market-card.high .market-probability { color: var(--success); }

        div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }

        div[data-testid="stButton"] button {
            min-height: 50px;
            border-radius: 14px;
            border: 1px solid rgba(170, 183, 196, 0.22);
            font-weight: 750;
            letter-spacing: 0.01em;
            transition:
                transform 160ms ease,
                box-shadow 160ms ease,
                border-color 160ms ease,
                background 160ms ease;
        }

        div[data-testid="stButton"] button:hover:not(:disabled) {
            transform: translateY(-1px);
            border-color: rgba(103, 181, 255, 0.75);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
        }

        div[data-testid="stButton"] button:active:not(:disabled) {
            transform: translateY(0);
            box-shadow: 0 5px 12px rgba(0, 0, 0, 0.2);
        }

        div[data-testid="stButton"] button:focus-visible {
            outline: 3px solid rgba(103, 181, 255, 0.38);
            outline-offset: 2px;
        }

        div[data-testid="stButton"] button:disabled {
            opacity: 0.48;
            cursor: not-allowed;
            box-shadow: none;
        }

        .st-key-open_events button,
        .st-key-analyze_event button {
            min-height: 56px;
            color: #FFFFFF;
            border-color: rgba(103, 181, 255, 0.72);
            background: linear-gradient(135deg, #1976D2 0%, #1457B8 100%);
            box-shadow: 0 12px 30px rgba(30, 136, 229, 0.28);
        }

        .st-key-open_events button:hover:not(:disabled),
        .st-key-analyze_event button:hover:not(:disabled) {
            border-color: rgba(157, 211, 255, 0.95);
            background: linear-gradient(135deg, #238FEA 0%, #1765CB 100%);
            box-shadow: 0 14px 34px rgba(30, 136, 229, 0.38);
        }

        .st-key-back_home button,
        .st-key-refresh_events button {
            color: var(--text);
            background: rgba(17, 23, 34, 0.86);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 12px;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(170, 183, 196, 0.16);
            border-radius: 14px;
            background: rgba(17, 23, 34, 0.6);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
                padding-top: 0.7rem;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.65rem;
            }

            div[data-testid="column"] {
                min-width: 100% !important;
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            .home-card,
            .sport-card {
                border-radius: 17px;
            }

            .metric-card {
                min-height: 110px;
            }

            .game-card {
                text-align: center;
            }

            div[data-testid="stButton"] button,
            .st-key-open_events button,
            .st-key-analyze_event button {
                min-height: 52px;
                border-radius: 13px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONFIGURACIÓN DE DEPORTES Y MERCADOS
# =========================================================

SPORT_LOGOS = {
    "Fútbol": "⚽",
    "Béisbol": "⚾",
    "Basketball": "🏀",
    "NFL": "🏈",
    "Fórmula 1": "🏎️",
    "Hockey": "🏒",
    "MMA": "🥊",
}

SPORT_NAMES = list(SPORT_LOGOS.keys())

SPORT_MODEL_HINTS = {
    "Fútbol": "Poisson + Monte Carlo",
    "Béisbol": "Runs Model + Monte Carlo",
    "Basketball": "Points Pace Model + Monte Carlo",
    "NFL": "Drive/Points Model + Monte Carlo",
    "Fórmula 1": "Forma reciente + Monte Carlo",
    "Hockey": "Forma de temporada + Monte Carlo",
    "MMA": "Perfil competitivo + Monte Carlo",
}


# =========================================================
# ESTADO DE LA APLICACIÓN
# =========================================================

def init_state() -> None:
    defaults = {
        "screen": "home",
        "selected_sport": "Fútbol",
        "selected_date": date.today(),
        "search_text": "",
        "simulaciones": DEFAULT_SIMULATIONS,
        "force_refresh": False,
        "match_options": [],
        "selected_competition_label": "Todas las competencias",
        "league_view_mode": "Todas",
        "prefetched_games": {},
        "prefetched_sources": {},
        "prefetched_errors": {},
        "prefetched_date": None,
        "last_cache_cleanup": None,
        "football_match_quality": {},
        "current_user": None,
        "public_screen": "landing",
        "auth_mode": "Iniciar sesión",
        "cookies_accepted": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_game_results() -> None:
    st.session_state["match_options"] = []
    st.session_state["selected_competition_label"] = "Todas las competencias"


def match_quality_key(match: dict[str, Any]) -> str:
    return str(match.get("game_id") or f'{match.get("home", "")}::{match.get("away", "")}::{match.get("date", "")}')


def remember_match_quality(match: dict[str, Any], result: dict[str, Any]) -> None:
    score = result.get("match_quality_score")
    if score is None:
        return
    st.session_state.setdefault("football_match_quality", {})[match_quality_key(match)] = {
        "score": float(score),
        "label": result.get("match_quality_label", "Calidad pendiente"),
        "explanation": result.get("match_quality_explanation", ""),
    }


def render_match_quality(result: dict[str, Any]) -> None:
    score = result.get("match_quality_score")
    if score is None:
        return
    score = float(score)
    message = (
        f'{result.get("match_quality_label", "Calidad de predicción")} · {score:.0%}. '
        f'{result.get("match_quality_explanation", "")}'
    )
    if score >= 0.75:
        st.success(message, icon=":material/local_fire_department:")
    elif score >= 0.60:
        st.warning(message, icon=":material/warning:")
    else:
        st.error(
            f"No recomendado · {message}",
            icon=":material/visibility_off:",
        )


def render_game_style(result: dict[str, Any]) -> None:
    label = result.get("game_style_label")
    if not label:
        return
    icon = result.get("game_style_icon", ":material/sports_soccer:")
    explanation = result.get("game_style_explanation", "")
    st.markdown(f"{icon} **{label}** · {explanation}")


def format_event_schedule(match: dict[str, Any]) -> str:
    raw_date = str(match.get("date") or "").strip()
    raw_time = str(match.get("time") or "").strip()
    if not raw_date:
        return "Horario por confirmar"
    try:
        value = raw_date if "T" in raw_date else f"{raw_date}T{raw_time or '00:00:00'}"
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
            return parsed.strftime("%d/%m/%Y · %H:%M hora local")
        if raw_time:
            return parsed.strftime("%d/%m/%Y · %H:%M")
        return parsed.strftime("%d/%m/%Y")
    except ValueError:
        return raw_date


# =========================================================
# PERSISTENCIA
# =========================================================

def save_prediction_to_db(
    sport: str,
    home_team: str,
    away_team: str,
    model_name: str,
    simulations: int,
    markets: list[dict[str, Any]],
    selected_match: dict[str, Any] | None = None,
    context_json: dict[str, Any] | None = None,
) -> int | None:
    try:
        repository = PredictionRepository()

        stored_context = dict(context_json or {})
        availability = ((selected_match or {}).get("analysis_context") or {}).get("availability")
        if availability:
            stored_context["availability"] = availability
        if selected_match:
            stored_context["match_metadata"] = {
                key: selected_match.get(key)
                for key in ("league", "league_id", "date", "provider")
                if selected_match.get(key) is not None
            }
        stored_context["analysis_type"] = classify_analysis(model_name, stored_context)
        current_user = st.session_state.get("current_user")
        if current_user:
            stored_context["user_id"] = int(current_user["id"])

        run_id = repository.save_prediction_run(
            sport=sport,
            match_id=str(selected_match["game_id"]) if selected_match and selected_match.get("game_id") else None,
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
            simulations=simulations,
            markets=markets,
            context_json=stored_context,
        )

        return run_id

    except Exception as error:
        logger.exception("No se pudo guardar la predicción: %s", error)
        st.warning("El análisis terminó, pero no pudo añadirse al historial.")
        return None


def render_analysis_method(result: dict[str, Any]) -> None:
    model_name = str(result.get("model_name") or "Modelo")
    analysis_type = classify_analysis(model_name, result.get("context_json"))
    label, description = analysis_method_copy(analysis_type)
    color = "green" if analysis_type == "hybrid_ai" else "blue"
    st.markdown(f":{color}-badge[{label}] :small[{description}]")


def render_prediction_insight(result: dict[str, Any]) -> None:
    """Show the most useful prediction signal without overstating certainty."""
    markets = result.get("markets_to_save") or []
    valid_markets = []
    for market in markets:
        try:
            probability = float(market.get("probability"))
        except (TypeError, ValueError):
            continue
        if 0.0 <= probability <= 100.0:
            valid_markets.append((probability, str(market.get("selection") or "Mercado")))
    if not valid_markets:
        return

    context = result.get("context_json") or {}
    recommended_result = context.get("recommended_result") or {}
    recommended_total = context.get("recommended_total") or {}
    if recommended_result:
        probability = float(recommended_result.get("probability") or 0.0)
        selection = str(recommended_result.get("selection") or "Resultado")
        if recommended_total:
            selection += f' · {recommended_total.get("label")} {float(recommended_total.get("probability") or 0.0):.1f}%'
    else:
        probability, selection = max(valid_markets, key=lambda item: item[0])
    selection = escape(selection)
    quality_gate = context.get("quality_gate") or {}
    validated = sum(value is True for value in quality_gate.values())
    validation_copy = (
        f"{validated} mercado{'s' if validated != 1 else ''} con señal ML validada."
        if validated
        else "Estimación estadística con simulación; sin señal ML aprobada."
    )
    st.markdown(
        f"""
        <div class="prediction-insight">
            <div class="prediction-kicker">Lectura rápida</div>
            <div class="prediction-pick">{selection} · {probability:.1f}%</div>
            <div class="prediction-note">{validation_copy} La probabilidad expresa incertidumbre y no garantiza el resultado.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_strong_markets(markets: list[dict[str, Any]]) -> None:
    public_markets = visible_markets(markets)
    if not public_markets:
        return
    st.markdown("<div class='section-title'>Mercados principales</div>", unsafe_allow_html=True)
    st.caption("Mostramos señales de 50% o más. El resultado 1X2 permanece visible para comparar escenarios.")
    for start in range(0, len(public_markets), 2):
        columns = st.columns(2)
        for column, market in zip(columns, public_markets[start:start + 2]):
            probability = market["probability"]
            strength = "high" if probability > 65.0 else "medium"
            category = escape(str((market.get("extra_data_json") or {}).get("category") or "Otros"))
            selection = escape(str(market.get("selection") or "Mercado"))
            extra_data = market.get("extra_data_json") or {}
            confidence_score = market.get("confidence_score", extra_data.get("confidence_score"))
            explanation = escape(str(market.get("explanation") or extra_data.get("explanation") or ""))
            with column:
                st.markdown(
                    f"""
                    <div class="market-card {strength}">
                        <div class="market-category">{category}</div>
                        <div class="market-selection">{selection}</div>
                        <div class="market-probability">{probability:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if confidence_score is not None:
                    st.caption(f"Confianza del análisis: {float(confidence_score):.0f}/100")
                if explanation:
                    st.caption(explanation)
def render_history_section(
    title: str = "Historial de análisis",
    limit: int = 10,
    sport_filter: str | None = None,
) -> None:
    import pandas as pd

    st.markdown(
        f"<div class='section-title'>{title}</div>",
        unsafe_allow_html=True,
    )

    try:
        repository = PredictionRepository()
        current_user = st.session_state.get("current_user") or {}
        runs = repository.list_recent_runs(
            limit=limit * 3,
            user_id=None if current_user.get("is_admin") else current_user.get("id"),
        )

        if sport_filter:
            runs = [run for run in runs if run["sport"] == sport_filter]

        runs = runs[:limit]

        if not runs:
            st.info("Todavía no hay análisis en el historial.")
            return

        history_df = pd.DataFrame(
            {
                "Deporte": [run["sport"] for run in runs],
                "Partido": [f'{run["home_team"]} vs {run["away_team"]}' for run in runs],
                "Modelo": [run["model_name"] for run in runs],
                "Fecha": [
                    run["created_at"].strftime("%Y-%m-%d %H:%M")
                    if run["created_at"] else ""
                    for run in runs
                ],
            }
        )

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
        )

        selected_run_id = st.selectbox(
            "Consulta un análisis anterior",
            [run["id"] for run in runs],
            format_func=lambda run_id: (
                f'{next(run for run in runs if run["id"] == run_id)["home_team"]} vs '
                f'{next(run for run in runs if run["id"] == run_id)["away_team"]} · '
                f'{next(run for run in runs if run["id"] == run_id)["sport"]}'
            ),
            key=f"history_run_selector_{title}_{sport_filter}",
        )

        selected_run = next(run for run in runs if run["id"] == selected_run_id)

        st.markdown(
            f"""
            <div class="game-card">
                <div class="game-teams">
                    {selected_run["home_team"]} vs {selected_run["away_team"]}
                </div>
                <div class="game-info">Deporte: {selected_run["sport"]}</div>
                <div class="game-info">Modelo: {selected_run["model_name"]}</div>
                <div class="game-info">Simulaciones: {selected_run["simulations"]:,}</div>
                <div class="game-info">
                    Fecha: {selected_run["created_at"].strftime("%Y-%m-%d %H:%M") if selected_run["created_at"] else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hasattr(repository, "list_markets_by_run"):
            markets = repository.list_markets_by_run(selected_run_id)
        else:
            markets = []

        if markets:
            markets_df = pd.DataFrame(
                {
                    "Mercado": [market["market_type"] for market in markets],
                    "Selección": [market["selection"] for market in markets],
                    "Probabilidad": [f'{market["probability"]:.1f}%' for market in markets],
                    "Confianza": [market["confidence"] or "" for market in markets],
                    "Riesgo": [market["risk"] or "" for market in markets],
                }
            )

            st.dataframe(
                markets_df,
                width="stretch",
                hide_index=True,
            )

        context_json = selected_run.get("context_json")
        if context_json:
            with st.expander("Detalles avanzados del análisis"):
                st.caption("Información del modelo utilizada para auditoría y seguimiento.")
                st.json(context_json, expanded=False)

    except Exception as error:
        logger.exception("No se pudo cargar el historial: %s", error)
        st.warning("No pudimos cargar el historial en este momento.")


def render_recent_results(
    limit: int = 8,
    sport_filter: str | None = None,
) -> None:
    import pandas as pd

    try:
        reviews = PostMatchReviewRepository().list_recent_reviews(
            limit=limit,
            sport=sport_filter,
        )
        if not reviews:
            return

        st.markdown(
            "<div class='section-title'>Resultados recientes</div>",
            unsafe_allow_html=True,
        )

        results_df = pd.DataFrame(
            {
                "Deporte": [review["sport"] for review in reviews],
                "Partido": [
                    f'{review["home_team"]} vs {review["away_team"]}'
                    for review in reviews
                ],
                "Resultado": [
                    f'{_format_score(review["home_score"])} - '
                    f'{_format_score(review["away_score"])}'
                    for review in reviews
                ],
                "Aciertos": [
                    f'{review["correct_markets"]}/{review["evaluated_markets"]}'
                    for review in reviews
                ],
                "Precisión": [
                    f'{review["accuracy"] * 100:.0f}%'
                    if review["accuracy"] is not None else "—"
                    for review in reviews
                ],
            }
        )
        st.dataframe(results_df, width="stretch", hide_index=True)

        selected_review_id = st.selectbox(
            "Revisar resultado",
            [review["id"] for review in reviews],
            format_func=lambda review_id: _review_label(reviews, review_id),
            key=f"result_review_selector_{sport_filter}",
        )
        selected_review = next(
            review for review in reviews if review["id"] == selected_review_id
        )

        st.markdown(
            f"""
            <div class="game-card">
                <div class="game-teams">
                    {selected_review["home_team"]}
                    {_format_score(selected_review["home_score"])}
                    &nbsp;–&nbsp;
                    {_format_score(selected_review["away_score"])}
                    {selected_review["away_team"]}
                </div>
                <div class="game-info">
                    {selected_review["correct_markets"]} de
                    {selected_review["evaluated_markets"]} mercados acertados
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        details = selected_review.get("details_json") or []
        if details:
            with st.expander("Revisión por mercado"):
                detail_df = pd.DataFrame(
                    {
                        "Mercado": [
                            _market_label(item["market_type"]) for item in details
                        ],
                        "Pronóstico": [
                            f'{item["probability"]:.1f}%' for item in details
                        ],
                        "Resultado": [
                            "Ocurrió" if item["actual"] else "No ocurrió"
                            for item in details
                        ],
                        "Evaluación": [
                            "Acertado" if item["correct"] else "Fallado"
                            for item in details
                        ],
                    }
                )
                st.dataframe(detail_df, width="stretch", hide_index=True)
                st.caption(f'Modelo utilizado: {selected_review["model_name"]}')

    except Exception as error:
        logger.exception("No se pudieron mostrar resultados recientes: %s", error)


def _format_score(score: float) -> str:
    return str(int(score)) if float(score).is_integer() else f"{score:g}"


def _review_label(reviews: list[dict[str, Any]], review_id: int) -> str:
    review = next(item for item in reviews if item["id"] == review_id)
    return (
        f'{review["home_team"]} vs {review["away_team"]} · '
        f'{review["sport"]} · {review["model_name"]}'
    )


def _market_label(market_type: str) -> str:
    labels = {
        "home_win": "Victoria local",
        "draw": "Empate",
        "away_win": "Victoria visitante",
        "btts": "Ambos anotan",
        "over_2_5_goals": "Más de 2.5 goles",
        "under_3_5_goals": "Menos de 3.5 goles",
        "over_8_5_runs": "Más de 8.5 carreras",
        "under_10_5_runs": "Menos de 10.5 carreras",
        "home_over_3_5_runs": "Local: más de 3.5 carreras",
        "over_219_5_points": "Más de 219.5 puntos",
        "under_234_5_points": "Menos de 234.5 puntos",
        "home_over_109_5_points": "Local: más de 109.5 puntos",
        "over_41_5_points": "Más de 41.5 puntos",
        "under_52_5_points": "Menos de 52.5 puntos",
        "home_over_20_5_points": "Local: más de 20.5 puntos",
    }
    return labels.get(market_type, market_type.replace("_", " ").title())


def render_performance_summary(sport_filter: str | None = None) -> None:
    import pandas as pd

    try:
        summary = ModelMetricsRepository().get_performance_summary(sport_filter)
        if not summary:
            return

        st.markdown(
            "<div class='section-title'>Rendimiento histórico</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Ver desempeño de los análisis"):
            metric_columns = st.columns(3)
            with metric_columns[0]:
                st.metric("Análisis evaluados", summary["reviews"])
            with metric_columns[1]:
                st.metric("Mercados evaluados", summary["evaluated_markets"])
            with metric_columns[2]:
                st.metric("Precisión general", f'{summary["accuracy"] * 100:.0f}%')

            st.caption("Desempeño por modelo")
            model_df = pd.DataFrame(
                {
                    "Modelo": [item["model_name"] for item in summary["models"]],
                    "Análisis": [item["reviews"] for item in summary["models"]],
                    "Mercados": [item["evaluated"] for item in summary["models"]],
                    "Precisión": [
                        f'{item["accuracy"] * 100:.0f}%'
                        for item in summary["models"]
                    ],
                }
            )
            st.dataframe(model_df, width="stretch", hide_index=True)

            st.caption("Desempeño por mercado")
            market_df = pd.DataFrame(
                {
                    "Mercado": [
                        _market_label(item["market_type"])
                        for item in summary["markets"]
                    ],
                    "Evaluaciones": [
                        item["evaluated"] for item in summary["markets"]
                    ],
                    "Precisión": [
                        f'{item["accuracy"] * 100:.0f}%'
                        for item in summary["markets"]
                    ],
                }
            )
            st.dataframe(market_df, width="stretch", hide_index=True)
    except Exception as error:
        logger.exception("No se pudo mostrar el rendimiento histórico: %s", error)


# =========================================================
# NORMALIZACIÓN DE PARTIDOS Y COMPETENCIAS
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_competitions(
    sport: str,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    api = APIManager(sport)
    data = api.get_competitions(force_refresh=force_refresh)

    items = data.get("response", [])
    competitions: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        league = item.get("league") or item
        country = item.get("country") or {}
        seasons = item.get("seasons") or []

        league_id = league.get("id") or item.get("id")
        league_name = league.get("name") or item.get("name")

        if isinstance(country, dict):
            country_name = country.get("name") or "World"
        else:
            country_name = str(country or "World")

        if not league_id or not league_name:
            continue

        valid_seasons: list[int] = []

        for season in seasons:
            if not isinstance(season, dict):
                continue

            year = season.get("year", season.get("season"))
            if isinstance(year, int):
                valid_seasons.append(year)
            elif isinstance(year, str) and year.isdigit():
                valid_seasons.append(int(year))

        fallback_season = datetime.now().year
        latest_season = max(valid_seasons) if valid_seasons else fallback_season

        label = f"{league_name} | {country_name} | {latest_season}"

        competitions.append(
            {
                "label": label,
                "league_id": league_id,
                "league_name": league_name,
                "country": country_name,
                "season": latest_season,
            }
        )

    unique_competitions: dict[str, dict[str, Any]] = {}

    for competition in competitions:
        unique_key = (
            f"{competition['league_id']}-"
            f"{competition['season']}-"
            f"{competition['country']}"
        )
        unique_competitions[unique_key] = competition

    return sorted(
        unique_competitions.values(),
        key=lambda competition: competition["label"].lower(),
    )


def normalize_game(
    game: dict[str, Any],
    sport: str,
) -> dict[str, Any]:
    if sport == "Fórmula 1":
        race = game.get("race") or {}
        circuit = game.get("circuit") or {}
        raw_status = race.get("status") or "Scheduled"
        status = "Finalizado" if is_finished_status(raw_status) else "Programado"
        race_name = race.get("name") or "Grand Prix"
        circuit_name = circuit.get("name") or "Circuito"
        country = circuit.get("country") or ""
        city = circuit.get("city") or ""
        race_date = race.get("date") or ""
        race_time = race.get("time") or ""

        return {
            "label": f"Fórmula 1 | {race_name} | {status}",
            "home": race_name,
            "away": circuit_name,
            "home_id": race.get("id"),
            "away_id": circuit.get("id"),
            "home_logo": "",
            "away_logo": "",
            "game_id": race.get("id"),
            "provider": game.get("provider", "jolpica"),
            "league": "Formula 1",
            "country": country,
            "status": status,
            "is_finished": is_finished_status(raw_status),
            "is_available_for_pregame": is_available_for_pregame(raw_status),
            "home_score": None,
            "away_score": None,
            "date": race_date,
            "time": race_time,
            "season": race.get("season"),
            "round": race.get("round"),
            "circuit": circuit_name,
            "city": city,
            "event_type": "race",
        }

    league_data = game.get("league") or {}

    game_id = None
    status = "Programado"
    game_date = ""
    game_time = ""

    raw_status: Any

    if sport == "Fútbol":
        fixture = game.get("fixture") or {}
        game_id = fixture.get("id")
        game_date = fixture.get("date", "")

        status_data = fixture.get("status") or {}
        raw_status = status_data
        if isinstance(status_data, dict):
            status = status_data.get("long") or status_data.get("short") or "Programado"

    else:
        game_id = game.get("id")

        if game_id is None:
            nested_game = game.get("game") or {}
            game_id = nested_game.get("id")

        game_date = game.get("date", "")
        game_time = game.get("time", "")

        status_data = game.get("status") or {}
        raw_status = status_data
        if isinstance(status_data, dict):
            status = (
                status_data.get("long")
                or status_data.get("short")
                or status_data.get("type")
                or "Programado"
            )
        else:
            status = str(status_data or "Programado")

    teams = game.get("teams") or {}
    home_data = teams.get("home") or {}
    away_data = teams.get("away") or {}

    if isinstance(home_data, dict):
        home = home_data.get("name") or "Local"
        home_logo = home_data.get("logo") or ""
        home_id = home_data.get("id")
    else:
        home = str(home_data or "Local")
        home_logo = ""
        home_id = None

    if isinstance(away_data, dict):
        away = away_data.get("name") or "Visitante"
        away_logo = away_data.get("logo") or ""
        away_id = away_data.get("id")
    else:
        away = str(away_data or "Visitante")
        away_logo = ""
        away_id = None

    league = league_data.get("name") or "Competencia desconocida"
    country = league_data.get("country") or ""
    if str(country).lower() in {"league", "cup"}:
        country = ""

    status_translations = {
        "NOT STARTED": "Programado",
        "NS": "Programado",
        "SCHEDULED": "Programado",
        "GAME FINISHED": "Finalizado",
        "MATCH FINISHED": "Finalizado",
        "FINISHED": "Finalizado",
        "FT": "Finalizado",
    }
    status = status_translations.get(str(status).strip().upper(), status)

    label = f"{league} | {home} vs {away} | {status}"
    home_score, away_score = extract_final_score(game, sport)

    return {
        "label": label,
        "home": home,
        "away": away,
        "home_id": home_id,
        "away_id": away_id,
        "home_logo": home_logo,
        "away_logo": away_logo,
        "game_id": game_id,
        "provider": game.get("provider", "api_sports"),
        "league": league,
        "country": country,
        "status": status,
        "is_finished": is_finished_status(raw_status),
        "is_available_for_pregame": is_available_for_pregame(raw_status),
        "home_score": home_score,
        "away_score": away_score,
        "date": game_date,
        "time": game_time,
        "analysis_context": game.get("analysis_context") or {},
    }


def process_finished_games(
    sport: str,
    games: list[dict[str, Any]],
) -> None:
    PostMatchService().process_games(sport=sport, games=games)


def prefetch_all_sports(
    selected_date: date,
    force_refresh: bool = False,
    sports: list[str] | None = None,
) -> None:
    date_string = selected_date.strftime("%Y-%m-%d")
    requested_sports = sports or SPORT_NAMES
    same_date = st.session_state.get("prefetched_date") == date_string
    existing_games = st.session_state.get("prefetched_games", {}) if same_date else {}

    if (
        same_date
        and all(sport in existing_games for sport in requested_sports)
        and not force_refresh
    ):
        return

    all_games: dict[str, list[dict[str, Any]]] = dict(existing_games)
    sources: dict[str, str] = dict(st.session_state.get("prefetched_sources", {})) if same_date else {}
    errors: dict[str, str] = dict(st.session_state.get("prefetched_errors", {})) if same_date else {}

    progress_bar = st.progress(0)
    status_box = st.empty()

    total_sports = len(requested_sports)

    for index, sport in enumerate(requested_sports, start=1):
        errors.pop(sport, None)
        status_box.write(f"Cargando partidos de {sport} para {date_string}...")

        try:
            api = APIManager(sport)
            response = api.get_games_by_date(
                date=date_string,
                league_id=None,
                season=None,
                force_refresh=force_refresh,
            )

            raw_games = response.get("response", [])
            source = response.get("_source", "api")

            normalized_games: list[dict[str, Any]] = []

            for game in raw_games:
                if not isinstance(game, dict):
                    continue

                try:
                    normalized_games.append(normalize_game(game=game, sport=sport))
                except (KeyError, TypeError, AttributeError) as error:
                    logger.warning("No se pudo normalizar partido de %s: %s", sport, error)

            all_games[sport] = normalized_games
            sources[sport] = source
            process_finished_games(sport=sport, games=normalized_games)

        except Exception as error:
            all_games[sport] = []
            sources[sport] = "error"
            errors[sport] = str(error)
            logger.exception("Error precargando %s: %s", sport, error)

        percentage = int((index / total_sports) * 100)
        progress_bar.progress(percentage)

    st.session_state["prefetched_games"] = all_games
    st.session_state["prefetched_sources"] = sources
    st.session_state["prefetched_errors"] = errors
    st.session_state["prefetched_date"] = date_string

    progress_bar.empty()
    status_box.empty()


# =========================================================
# MOTOR DE PREDICCIÓN
# =========================================================

def get_optional_engine_factory(sport: str) -> Callable | None:
    """
    Deja app.py casi fijo a futuro.
    Cuando crees nuevos engines reales, solo deben existir:
      machine_learning/predictors/*_predictor.py -> predictor con fallback
    y este app.py los detectará automáticamente.
    """
    mapping = {
        "Béisbol": ("machine_learning.predictors.baseball_predictor", "BaseballPredictor"),
        "Basketball": ("machine_learning.predictors.basketball_predictor", "BasketballPredictor"),
        "NFL": ("machine_learning.predictors.nfl_predictor", "NFLPredictor"),
        "Fórmula 1": ("machine_learning.predictors.formula1_predictor", "Formula1Predictor"),
        "Hockey": ("engines.specialty_prediction_engines", "HockeyPredictionEngine"),
        "MMA": ("engines.specialty_prediction_engines", "MMAPredictionEngine"),
    }

    if sport not in mapping:
        return None

    module_name, class_name = mapping[sport]

    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception:
        return None


def run_shadow_validation(
    sport: str,
    selected_match: dict[str, Any] | None,
    simulations: int,
) -> None:
    """Load candidate validation only after a real analysis requests it."""
    from machine_learning.shadow_validation import ShadowValidationService

    ShadowValidationService().run(sport, selected_match, simulations)


def build_demo_markets_by_sport(
    sport: str,
    home_team: str,
    away_team: str,
    home_win_probability: float,
    draw_probability: float,
    away_win_probability: float,
    over_probability: float,
    under_probability: float,
    extra_probability: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    import pandas as pd

    if sport == "Fútbol":
        markets_df = pd.DataFrame(
            {
                "Mercado": [
                    f"{home_team} gana",
                    "Empate",
                    f"{away_team} gana",
                    "Over 2.5 goles",
                    "Under 3.5 goles",
                    f"{home_team} over 4.5 córners",
                ],
                "Probabilidad": [
                    f"{home_win_probability:.1f}%",
                    f"{draw_probability:.1f}%",
                    f"{away_win_probability:.1f}%",
                    f"{over_probability:.1f}%",
                    f"{under_probability:.1f}%",
                    f"{extra_probability:.1f}%",
                ],
                "Confianza": ["Demo"] * 6,
                "Riesgo": ["Demo"] * 6,
            }
        )

        markets_to_save = [
            {"market_type": "home_win", "selection": home_team, "probability": home_win_probability, "confidence": "Demo", "risk": "Demo"},
            {"market_type": "draw", "selection": "Empate", "probability": draw_probability, "confidence": "Demo", "risk": "Demo"},
            {"market_type": "away_win", "selection": away_team, "probability": away_win_probability, "confidence": "Demo", "risk": "Demo"},
            {"market_type": "over_2_5_goals", "selection": "Over 2.5 goles", "probability": over_probability, "confidence": "Demo", "risk": "Demo"},
            {"market_type": "under_3_5_goals", "selection": "Under 3.5 goles", "probability": under_probability, "confidence": "Demo", "risk": "Demo"},
            {"market_type": "home_over_4_5_corners", "selection": f"{home_team} over 4.5 córners", "probability": extra_probability, "confidence": "Demo", "risk": "Demo"},
        ]
        return markets_df, markets_to_save

    if sport == "Béisbol":
        markets_df = pd.DataFrame(
            {
                "Mercado": [
                    f"{home_team} gana",
                    f"{away_team} gana",
                    "Over 8.5 carreras",
                    "Under 10.5 carreras",
                    f"{home_team} over 3.5 carreras",
                ],
                "Probabilidad": [
                    f"{home_win_probability:.1f}%",
                    f"{away_win_probability:.1f}%",
                    f"{over_probability:.1f}%",
                    f"{under_probability:.1f}%",
                    f"{extra_probability:.1f}%",
                ],
                "Confianza": ["Inicial"] * 5,
                "Riesgo": ["Inicial"] * 5,
            }
        )

        markets_to_save = [
            {"market_type": "home_win", "selection": home_team, "probability": home_win_probability, "confidence": "Inicial", "risk": "Inicial"},
            {"market_type": "away_win", "selection": away_team, "probability": away_win_probability, "confidence": "Inicial", "risk": "Inicial"},
            {"market_type": "over_8_5_runs", "selection": "Over 8.5 carreras", "probability": over_probability, "confidence": "Inicial", "risk": "Inicial"},
            {"market_type": "under_10_5_runs", "selection": "Under 10.5 carreras", "probability": under_probability, "confidence": "Inicial", "risk": "Inicial"},
            {"market_type": "home_over_3_5_runs", "selection": f"{home_team} over 3.5 carreras", "probability": extra_probability, "confidence": "Inicial", "risk": "Inicial"},
        ]
        return markets_df, markets_to_save

    if sport == "Basketball":
        markets_df = pd.DataFrame(
            {
                "Mercado": [
                    f"{home_team} gana",
                    f"{away_team} gana",
                    "Over 219.5 puntos",
                    "Under 234.5 puntos",
                    f"{home_team} over 109.5 puntos",
                ],
                "Probabilidad": [
                    f"{home_win_probability:.1f}%",
                    f"{away_win_probability:.1f}%",
                    f"{over_probability:.1f}%",
                    f"{under_probability:.1f}%",
                    f"{extra_probability:.1f}%",
                ],
                "Confianza": ["Inicial"] * 5,
                "Riesgo": ["Inicial"] * 5,
            }
        )

        markets_to_save = [
            {"market_type": "home_win", "selection": home_team, "probability": home_win_probability, "confidence": "Inicial", "risk": "Inicial"},
            {"market_type": "away_win", "selection": away_team, "probability": away_win_probability, "confidence": "Inicial", "risk": "Inicial"},
            {"market_type": "over_219_5_points", "selection": "Over 219.5 puntos", "probability": over_probability, "confidence": "Inicial", "risk": "Inicial"},
            {"market_type": "under_234_5_points", "selection": "Under 234.5 puntos", "probability": under_probability, "confidence": "Inicial", "risk": "Inicial"},
            {"market_type": "home_over_109_5_points", "selection": f"{home_team} over 109.5 puntos", "probability": extra_probability, "confidence": "Inicial", "risk": "Inicial"},
        ]
        return markets_df, markets_to_save

    markets_df = pd.DataFrame(
        {
            "Mercado": [
                f"{home_team} gana",
                f"{away_team} gana",
                "Over 41.5 puntos",
                "Under 52.5 puntos",
                f"{home_team} over 20.5 puntos",
            ],
            "Probabilidad": [
                f"{home_win_probability:.1f}%",
                f"{away_win_probability:.1f}%",
                f"{over_probability:.1f}%",
                f"{under_probability:.1f}%",
                f"{extra_probability:.1f}%",
            ],
            "Confianza": ["Inicial"] * 5,
            "Riesgo": ["Inicial"] * 5,
        }
    )

    markets_to_save = [
        {"market_type": "home_win", "selection": home_team, "probability": home_win_probability, "confidence": "Inicial", "risk": "Inicial"},
        {"market_type": "away_win", "selection": away_team, "probability": away_win_probability, "confidence": "Inicial", "risk": "Inicial"},
        {"market_type": "over_41_5_points", "selection": "Over 41.5 puntos", "probability": over_probability, "confidence": "Inicial", "risk": "Inicial"},
        {"market_type": "under_52_5_points", "selection": "Under 52.5 puntos", "probability": under_probability, "confidence": "Inicial", "risk": "Inicial"},
        {"market_type": "home_over_20_5_points", "selection": f"{home_team} over 20.5 puntos", "probability": extra_probability, "confidence": "Inicial", "risk": "Inicial"},
    ]
    return markets_df, markets_to_save


# =========================================================
# LOGO
# =========================================================

def show_logo() -> None:
    columns = st.columns([1, 2.4, 1])

    with columns[1]:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width="stretch")
        else:
            st.markdown(
                f"<h1 style='text-align:center'>{PROJECT}</h1>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div class='small-text'>Análisis deportivo inteligente · Versión {VERSION}</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# PANTALLA DE INICIO
# =========================================================

def home_screen() -> None:
    show_logo()

    st.markdown(
        """
        <div class="home-card">
            <div class="home-heading">
                Analiza el partido que buscas
            </div>
            <div class="home-description">
                Explora los partidos disponibles, compara probabilidades
                y guarda cada análisis en tu historial.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    form_columns = st.columns(2)

    with form_columns[0]:
        sport = st.selectbox(
            "¿Qué deporte quieres abrir primero?",
            SPORT_NAMES,
            index=SPORT_NAMES.index(st.session_state.get("selected_sport", "Fútbol")),
        )

    with form_columns[1]:
        selected_date = st.date_input(
            "Fecha de los partidos",
            value=st.session_state.get("selected_date", date.today()),
            format="YYYY/MM/DD",
        )

    search_text = st.text_input(
        "Filtrar por liga, competencia, equipo o partido",
        value=st.session_state.get("search_text", ""),
        placeholder="Ejemplo: Liga MX, Lakers, UFC, NHL o Fórmula 1...",
    )

    simulations = DEFAULT_SIMULATIONS
    force_refresh = False

    prefetched_date = st.session_state.get("prefetched_date")
    selected_date_string = selected_date.strftime("%Y-%m-%d")

    if prefetched_date == selected_date_string:
        st.caption("Los partidos de esta fecha están listos para continuar.")

    if st.button(
        "Ver eventos disponibles",
        width="stretch",
        type="primary",
        icon=":material/arrow_forward:",
        key="open_events",
    ):
        st.session_state["selected_sport"] = sport
        st.session_state["selected_date"] = selected_date
        st.session_state["search_text"] = search_text
        st.session_state["simulaciones"] = int(simulations)
        st.session_state["force_refresh"] = force_refresh

        with st.spinner("Cargando los eventos deportivos disponibles..."):
            prefetch_all_sports(
                selected_date=selected_date,
                force_refresh=force_refresh,
                sports=[sport],
            )

        st.session_state["screen"] = "sport"
        reset_game_results()
        st.rerun()

# =========================================================
# PANTALLA DEL DEPORTE
# =========================================================

def sport_screen() -> None:
    sport = st.session_state["selected_sport"]
    selected_date = st.session_state["selected_date"]
    search_text = st.session_state.get("search_text", "").strip().lower()
    simulations = int(st.session_state["simulaciones"])

    header_columns = st.columns([1, 5])

    with header_columns[0]:
        if st.button(
            "Volver",
            width="stretch",
            icon=":material/arrow_back:",
            key="back_home",
        ):
            st.session_state["screen"] = "home"
            reset_game_results()
            st.rerun()

    with header_columns[1]:
        st.markdown(
            f"""
            <div class="sport-card">
                <div class="sport-logo">{SPORT_LOGOS.get(sport, "🏟️")}</div>
                <div class="sport-name">{sport}</div>
                <div class="sport-date">Partidos del {selected_date.strftime("%d/%m/%Y")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    selected_sport = st.selectbox(
        "Cambiar de deporte",
        SPORT_NAMES,
        index=SPORT_NAMES.index(sport),
    )

    if selected_sport != sport:
        st.session_state["selected_sport"] = selected_sport
        reset_game_results()
        st.rerun()

    prefetched_games = st.session_state.get("prefetched_games", {})
    if sport not in prefetched_games:
        with st.spinner(f"Cargando partidos de {sport}..."):
            prefetch_all_sports(
                selected_date=selected_date,
                force_refresh=False,
                sports=[sport],
            )
        prefetched_games = st.session_state.get("prefetched_games", {})

    all_sport_games = prefetched_games.get(sport, [])
    sport_error = st.session_state.get("prefetched_errors", {}).get(sport)

    st.markdown("<div class='section-title'>Competencia</div>", unsafe_allow_html=True)

    league_view = st.segmented_control(
        "Vista de ligas",
        ["Principales", "Todas"],
        default=st.session_state.get("league_view_mode", "Todas"),
        key=f"league_view_selector_{sport}",
        width="stretch",
    ) or "Principales"
    st.session_state["league_view_mode"] = league_view

    visible_sport_games = filter_games_by_league_view(
        all_sport_games,
        sport=sport,
        view=league_view,
    )

    competitions_map: dict[str, dict[str, str]] = {}
    for game in visible_sport_games:
        league_name = game.get("league", "Competencia desconocida")
        country_name = game.get("country", "")
        label = f"{league_name} | {country_name}" if country_name else league_name

        competitions_map[label] = {
            "label": label,
            "league_name": league_name,
            "country": country_name,
        }

    competitions = sorted(
        competitions_map.values(),
        key=lambda competition: (
            0 if is_primary_league(
                sport,
                competition["league_name"],
                competition["country"],
            ) else 1,
            competition["label"].lower(),
        ),
    )

    competition_labels = ["Todas las competencias"] + [competition["label"] for competition in competitions]

    current_competition = st.session_state.get("selected_competition_label", "Todas las competencias")
    if current_competition not in competition_labels:
        current_competition = "Todas las competencias"

    selected_competition_label = st.selectbox(
        "Selecciona la competencia",
        competition_labels,
        index=competition_labels.index(current_competition),
    )

    st.session_state["selected_competition_label"] = selected_competition_label

    selected_league = None
    if selected_competition_label != "Todas las competencias":
        selected_league = competitions_map[selected_competition_label]

    games = [
        game
        for game in visible_sport_games
        if game.get("is_available_for_pregame", True)
    ]

    if selected_league:
        games = [
            game for game in games
            if game.get("league") == selected_league["league_name"]
            and game.get("country", "") == selected_league["country"]
        ]

    if search_text:
        games = [game for game in games if search_text in game["label"].lower()]

    if sport == "Fútbol":
        show_all_quality = st.toggle(
            "Ver todos los partidos",
            value=False,
            key="show_all_football_quality",
            help="Incluye partidos ya evaluados con calidad inferior a 60%.",
        )
        known_quality = st.session_state.get("football_match_quality", {})
        if not show_all_quality:
            games = [
                game for game in games
                if known_quality.get(match_quality_key(game), {}).get("score", 1.0) >= 0.60
            ]

    st.session_state["match_options"] = games

    st.markdown("<div class='section-title'>Partidos del día</div>", unsafe_allow_html=True)

    if sport_error:
        st.error(f"No pudimos cargar los partidos de {sport}. Intenta actualizar los datos.")
    elif games:
        st.caption(f"{len(games)} partidos disponibles")
    elif all_sport_games and all(game.get("is_finished") for game in all_sport_games):
        st.info(
            f"Los {len(all_sport_games)} eventos registrados para esta fecha ya finalizaron. "
            "Puedes consultar su evaluación en resultados recientes."
        )
    else:
        st.info("No hay partidos que coincidan con los filtros seleccionados.")

    with st.expander("Opciones de actualización"):
        st.caption(f"Consulta nuevamente la información de {sport}.")
        if st.button(
            "Actualizar eventos",
            width="stretch",
            icon=":material/refresh:",
            key="refresh_events",
        ):
            with st.spinner("Actualizando partidos..."):
                prefetch_all_sports(selected_date=selected_date, force_refresh=True, sports=[sport])

            reset_game_results()
            st.rerun()

        if st.button(
            "Actualizar todos los deportes",
            width="stretch",
            icon=":material/sync:",
            key="refresh_all_events",
        ):
            with st.spinner("Actualizando todos los deportes..."):
                prefetch_all_sports(selected_date=selected_date, force_refresh=True)

            reset_game_results()
            st.rerun()

    selected_match = None

    if games:
        game_labels = [game["label"] for game in games]

        selected_label = st.selectbox(
            "Selecciona el partido que quieres analizar",
            game_labels,
        )

        selected_match = next(game for game in games if game["label"] == selected_label)
        schedule_text = format_event_schedule(selected_match)
        selected_quality = st.session_state.get("football_match_quality", {}).get(
            match_quality_key(selected_match)
        ) if sport == "Fútbol" else None

        if selected_quality:
            if selected_quality["score"] >= 0.75:
                st.success(
                    f'Partido recomendado · {selected_quality["score"]:.0%}. '
                    f'{selected_quality["explanation"]}',
                    icon=":material/local_fire_department:",
                )
            elif selected_quality["score"] >= 0.60:
                st.warning(
                    f'Calidad media · {selected_quality["score"]:.0%}. '
                    f'{selected_quality["explanation"]}',
                    icon=":material/warning:",
                )

        if sport == "Fórmula 1":
            location = " · ".join(
                value for value in (selected_match.get("city"), selected_match.get("country"))
                if value
            )
            st.markdown(
                f"""
                <div class="game-card">
                    <div class="game-teams">{selected_match["home"]}</div>
                    <div class="game-info">{selected_match["circuit"]}</div>
                    <div class="game-info">{location}</div>
                    <div class="game-info">Estado: {selected_match["status"]}</div>
                    <div class="game-info">{schedule_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            team_columns = st.columns([1, 2, 1])

            with team_columns[0]:
                if selected_match["home_logo"]:
                    st.image(selected_match["home_logo"], width=95)

            with team_columns[1]:
                st.markdown(
                    f"""
                    <div class="game-card">
                        <div class="game-teams">
                            {selected_match["home"]} &nbsp; vs &nbsp; {selected_match["away"]}
                        </div>
                        <div class="game-info">
                            {selected_match["league"]} · {selected_match["country"]}
                        </div>
                        <div class="game-info">
                            Estado: {selected_match["status"]}
                        </div>
                        <div class="game-info">{schedule_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with team_columns[2]:
                if selected_match["away_logo"]:
                    st.image(selected_match["away_logo"], width=95)

        home_team = selected_match["home"]
        away_team = selected_match["away"]

    elif sport not in {"Fórmula 1", "Hockey", "MMA"}:
        st.markdown("<div class='section-title'>Selección manual</div>", unsafe_allow_html=True)
        manual_columns = st.columns(2)

        with manual_columns[0]:
            home_team = st.text_input("Equipo local", "Equipo local")

        with manual_columns[1]:
            away_team = st.text_input("Equipo visitante", "Equipo visitante")

    else:
        home_team = ""
        away_team = ""
        if not sport_error:
            st.info("Elige otra fecha para encontrar un evento disponible para análisis.")

    current_user = st.session_state.get("current_user")
    usage_status = usage_tracker.can_user_predict(current_user["id"], sport)
    can_analyze = (
        (selected_match is not None or sport not in {"Fórmula 1", "Hockey", "MMA"})
        and usage_status["allowed"]
    )
    if usage_status["limit"] is None:
        st.caption("Tu plan permite análisis ilimitados por deporte.")
    else:
        st.caption(
            f'Uso de hoy en {sport}: {usage_status["used"]} de '
            f'{usage_status["limit"] + usage_status["extra"]}.'
        )
    if not usage_status["allowed"]:
        st.warning("Has alcanzado tu límite diario para este deporte")
    if st.button(
        "Analizar evento",
        width="stretch",
        type="primary",
        icon=":material/analytics:",
        disabled=not can_analyze,
        key="analyze_event",
    ):
        try:
            usage_tracker.record_prediction(current_user["id"], sport)
            run_analysis(
                home_team=home_team,
                away_team=away_team,
                simulations=simulations,
                sport=sport,
                selected_match=selected_match,
            )
        except PermissionError:
            st.warning("Has alcanzado tu límite diario para este deporte")

# =========================================================
# ANÁLISIS
# =========================================================

def run_analysis(
    home_team: str,
    away_team: str,
    simulations: int,
    sport: str,
    selected_match: dict[str, Any] | None = None,
) -> None:
    logger.info("Análisis solicitado: %s vs %s | %s", home_team, away_team, sport)

    st.markdown("<div class='section-title'>Preparando tu análisis</div>", unsafe_allow_html=True)
    st.caption(
        "Estamos revisando datos, forma reciente y miles de escenarios posibles. "
        "Mantén esta pestaña abierta mientras termina el proceso."
    )

    progress_bar = st.progress(0)
    status_text = st.empty()
    simulation_counter = st.empty()

    force_refresh = bool(st.session_state.get("force_refresh", False))

    if selected_match:
        selected_match = dict(selected_match)
        analysis_context = dict(selected_match.get("analysis_context") or {})
        analysis_context["availability"] = PlayerAvailabilityService().get_match_availability(
            sport=sport,
            match={**selected_match, "sport": sport},
            force_refresh=force_refresh,
        )
        selected_match["analysis_context"] = analysis_context

    # ---------------- FÚTBOL REAL ----------------
    if sport == "Fútbol" and selected_match and selected_match.get("home_id") and selected_match.get("away_id"):
        try:
            from machine_learning.predictors.football_predictor import FootballPredictor

            status_text.write("15% - Preparando el modelo de análisis")
            progress_bar.progress(15)

            predictor = FootballPredictor()

            status_text.write("40% - Comparando forma reciente y rendimiento")
            progress_bar.progress(40)

            status_text.write("75% - Evaluando escenarios y probabilidades")
            progress_bar.progress(75)

            result = predictor.predict_match(
                home_team_id=selected_match["home_id"],
                away_team_id=selected_match["away_id"],
                home_team_name=home_team,
                away_team_name=away_team,
                simulations=simulations,
                force_refresh=force_refresh,
                provider=selected_match.get("provider", "api_sports"),
            )
            remember_match_quality(selected_match, result)

            progress_bar.progress(100)
            status_text.write("100% - Análisis completado")
            simulation_counter.write("La evaluación terminó correctamente.")

            st.success("Análisis completado")
            render_analysis_method(result)
            render_match_quality(result)
            render_game_style(result)
            render_prediction_insight(result)

            summary_cards = result.get("summary_cards", [])
            if summary_cards:
                st.markdown("<div class='section-title'>Resumen principal</div>", unsafe_allow_html=True)
                cols = st.columns(len(summary_cards))
                for col, card in zip(cols, summary_cards):
                    with col:
                        st.markdown(
                            f'''
                            <div class="metric-card">
                                <div class="metric-title">{card["label"]}</div>
                                <div class="metric-value">{card["value"]}</div>
                            </div>
                            ''',
                            unsafe_allow_html=True,
                        )

            extra_metrics = result.get("extra_metrics", {})
            if extra_metrics:
                with st.expander("Cómo se realizó este análisis"):
                    metric_cols = st.columns(len(extra_metrics))
                    for col, (label, value) in zip(metric_cols, extra_metrics.items()):
                        with col:
                            st.metric(label, value)

            render_strong_markets(result.get("markets_to_save", []))

            run_id = save_prediction_to_db(
                sport=sport,
                home_team=home_team,
                away_team=away_team,
                model_name=result.get("model_name", "Football ML + Poisson + Monte Carlo"),
                simulations=simulations,
                markets=result.get("markets_to_save", []),
                selected_match=selected_match,
                context_json=result.get("context_json"),
            )

            if run_id:
                st.caption("Análisis guardado en tu historial.")

            run_shadow_validation(sport, selected_match, simulations)

            return

        except FileNotFoundError:
            st.warning(
                "No se encontró el modelo entrenado de fútbol. "
                "Se usará el motor base Poisson + Monte Carlo."
            )

            try:
                from engines.football_prediction_engine import FootballPredictionEngine

                engine = FootballPredictionEngine()

                status_text.write("35% - Revisando el historial de ambos equipos")
                progress_bar.progress(35)

                home_profile, away_profile = engine.get_team_profiles(
                    home_team_id=selected_match["home_id"],
                    away_team_id=selected_match["away_id"],
                    force_refresh=force_refresh,
                    provider=selected_match.get("provider", "api_sports"),
                )

                status_text.write("60% - Comparando ataque, defensa y forma reciente")
                progress_bar.progress(60)

                home_lambda, away_lambda = engine.calculate_expected_goals(
                    home_profile=home_profile,
                    away_profile=away_profile,
                )

                status_text.write("80% - Evaluando escenarios y probabilidades")
                progress_bar.progress(80)

                result = engine.run_monte_carlo(
                    home_lambda=home_lambda,
                    away_lambda=away_lambda,
                    simulations=simulations,
                )

                progress_bar.progress(100)
                status_text.write("100% - Análisis completado")
                simulation_counter.write("La evaluación terminó correctamente.")

                st.success("Análisis completado")
                st.markdown(
                    ":blue-badge[Modelo estadístico] "
                    ":small[Utiliza Poisson y simulación; no se presenta como IA entrenada.]"
                )

                st.markdown("<div class='section-title'>Resumen principal</div>", unsafe_allow_html=True)
                result_choices = [
                    (home_team, result["home_win_probability"]),
                    ("Empate", result["draw_probability"]),
                    (away_team, result["away_win_probability"]),
                ]
                likely_result, likely_probability = max(result_choices, key=lambda item: item[1])
                recommended_total = result["recommended_total"]
                result_columns = st.columns(2)
                result_cards = [
                    (f"Resultado más probable: {likely_result}", likely_probability),
                    (recommended_total["label"], recommended_total["probability"]),
                ]

                for column, result_card in zip(result_columns, result_cards):
                    label, percentage = result_card
                    with column:
                        st.markdown(
                            f'''
                            <div class="metric-card">
                                <div class="metric-title">{label}</div>
                                <div class="metric-value">{percentage:.1f}%</div>
                            </div>
                            ''',
                            unsafe_allow_html=True,
                        )

                markets_to_save = engine.build_market_options(
                    home_team, away_team, result["home_win_probability"],
                    result["draw_probability"], result["away_win_probability"],
                    result["goal_lines"], result["btts_probability"],
                    result["home_score_probability"], result["away_score_probability"],
                )
                fallback_features = {
                    "home_matches_played": home_profile.get("played", 0),
                    "away_matches_played": away_profile.get("played", 0),
                    "home_avg_scored_last5": home_profile.get("avg_scored", 0),
                    "away_avg_scored_last5": away_profile.get("avg_scored", 0),
                    "home_avg_conceded_last5": home_profile.get("avg_conceded", 0),
                    "away_avg_conceded_last5": away_profile.get("avg_conceded", 0),
                }
                enrich_football_markets(
                    markets_to_save,
                    fallback_features,
                    {},
                    result,
                    {"result": False, "over_2_5": False, "btts": False},
                )
                match_quality = calculate_match_quality(
                    markets_to_save,
                    fallback_features,
                    quality_gate={"result": False, "over_2_5": False, "btts": False},
                )
                game_style = classify_game_style(fallback_features, match_quality)
                apply_game_style_to_markets(markets_to_save, game_style)
                remember_match_quality(selected_match, match_quality)
                render_match_quality(match_quality)
                render_game_style(game_style)
                render_strong_markets(markets_to_save)

                run_id = save_prediction_to_db(
                    sport=sport,
                    home_team=home_team,
                    away_team=away_team,
                    model_name="Poisson + Monte Carlo",
                    simulations=simulations,
                    markets=markets_to_save,
                    selected_match=selected_match,
                    context_json={
                        "home_lambda": home_lambda,
                        "away_lambda": away_lambda,
                        "home_profile": home_profile,
                        "away_profile": away_profile,
                        "top_scores": result.get("top_scores", []),
                        "goal_lines": result.get("goal_lines", {}),
                        "recommended_total": recommended_total,
                        "recommended_result": {"selection": likely_result, "probability": likely_probability},
                        "match_quality": match_quality,
                        "game_style": game_style,
                    },
                )

                if run_id:
                    st.caption("Análisis guardado en tu historial.")

                return

            except Exception as error:
                logger.exception("Falló el análisis base de fútbol: %s", error)
                progress_bar.empty()
                status_text.empty()
                simulation_counter.empty()
                st.error("No pudimos completar el análisis en este momento. Intenta actualizar los datos.")

        except Exception as error:
            logger.exception("Falló el análisis real de fútbol: %s", error)
            progress_bar.empty()
            status_text.empty()
            simulation_counter.empty()
            st.error("No pudimos completar el análisis en este momento. Intenta actualizar los datos.")


    # ---------------- OTROS ENGINES REALES FUTUROS ----------------
    engine_factory = get_optional_engine_factory(sport)

    if engine_factory and selected_match:
        try:
            status_text.write(f"20% - Preparando el análisis de {sport}")
            progress_bar.progress(20)

            engine = engine_factory()

            status_text.write("60% - Comparando datos y evaluando escenarios")
            progress_bar.progress(60)

            result = engine.analyze_match(
                selected_match=selected_match,
                simulations=simulations,
                force_refresh=force_refresh,
            )

            progress_bar.progress(100)
            status_text.write("100% - Análisis completado")
            simulation_counter.write("La evaluación terminó correctamente.")

            st.success("Análisis completado")
            render_analysis_method(result)
            render_prediction_insight(result)

            summary_cards = result.get("summary_cards", [])
            if summary_cards:
                st.markdown("<div class='section-title'>Resumen principal</div>", unsafe_allow_html=True)
                cols = st.columns(len(summary_cards))
                for col, card in zip(cols, summary_cards):
                    with col:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">{card["label"]}</div>
                                <div class="metric-value">{card["value"]}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            extra_metrics = result.get("extra_metrics", {})
            if extra_metrics:
                with st.expander("Cómo se realizó este análisis"):
                    metric_cols = st.columns(len(extra_metrics))
                    for col, (label, value) in zip(metric_cols, extra_metrics.items()):
                        with col:
                            st.metric(label, value)

            render_strong_markets(result.get("markets_to_save", []))

            run_id = save_prediction_to_db(
                sport=sport,
                home_team=home_team,
                away_team=away_team,
                model_name=result.get("model_name", SPORT_MODEL_HINTS.get(sport, "Modelo")),
                simulations=simulations,
                markets=result.get("markets_to_save", []),
                selected_match=selected_match,
                context_json=result.get("context_json"),
            )

            if run_id:
                st.caption("Análisis guardado en tu historial.")

            run_shadow_validation(sport, selected_match, simulations)

            return

        except Exception as error:
            logger.exception("Falló el análisis real de %s: %s", sport, error)
            progress_bar.empty()
            status_text.empty()
            simulation_counter.empty()
            st.error("No pudimos completar el análisis en este momento. Intenta actualizar los datos.")
            return

    # ---------------- MODO INICIAL POR DEPORTE ----------------
    home_wins = 0
    draws = 0
    away_wins = 0
    over_probability_hits = 0
    under_probability_hits = 0
    extra_market_hits = 0

    for simulation in range(1, simulations + 1):
        home_score = random.choices([0, 1, 2, 3, 4, 5], weights=[12, 24, 28, 20, 11, 5])[0]
        away_score = random.choices([0, 1, 2, 3, 4, 5], weights=[22, 30, 24, 14, 7, 3])[0]
        total_score = home_score + away_score

        if home_score > away_score:
            home_wins += 1
        elif home_score == away_score:
            draws += 1
        else:
            away_wins += 1

        if total_score > 2.5:
            over_probability_hits += 1

        if total_score < 3.5:
            under_probability_hits += 1

        if home_score > 2:
            extra_market_hits += 1

        percent = int((simulation / simulations) * 100)
        update_interval = max(1, simulations // 100)

        if simulation == 1 or simulation % update_interval == 0 or simulation == simulations:
            status_text.write(f"{percent}% - Ejecutando simulación inicial")
            simulation_counter.write("Procesando escenarios posibles. Puedes seguir esperando.")
            progress_bar.progress(percent)

    st.success("Análisis completado")
    st.markdown(
        ":blue-badge[Modelo estadístico] "
        ":small[Utiliza simulación inicial; no se presenta como IA entrenada.]"
    )

    home_win_percentage = (home_wins / simulations) * 100
    draw_percentage = (draws / simulations) * 100
    away_win_percentage = (away_wins / simulations) * 100
    over_percentage = (over_probability_hits / simulations) * 100
    under_percentage = (under_probability_hits / simulations) * 100
    extra_percentage = (extra_market_hits / simulations) * 100

    st.markdown("<div class='section-title'>Resumen principal</div>", unsafe_allow_html=True)

    result_columns = st.columns(3)
    result_cards = [
        (f"Victoria {home_team}", home_win_percentage),
        ("Empate", draw_percentage),
        (f"Victoria {away_team}", away_win_percentage),
    ]

    for column, result_card in zip(result_columns, result_cards):
        label, percentage = result_card
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value">{percentage:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Cómo se realizó este análisis"):
        model_columns = st.columns(3)
        with model_columns[0]:
            st.metric("Estado", "Inicial")
        with model_columns[1]:
            st.metric("Modelo", SPORT_MODEL_HINTS.get(sport, "Modelo"))
        with model_columns[2]:
            st.metric("Simulaciones", f"{simulations:,}")

    markets_df, markets_to_save = build_demo_markets_by_sport(
        sport=sport,
        home_team=home_team,
        away_team=away_team,
        home_win_probability=home_win_percentage,
        draw_probability=draw_percentage,
        away_win_probability=away_win_percentage,
        over_probability=over_percentage,
        under_probability=under_percentage,
        extra_probability=extra_percentage,
    )

    render_strong_markets(markets_to_save)

    run_id = save_prediction_to_db(
        sport=sport,
        home_team=home_team,
        away_team=away_team,
        model_name=f"{sport} Initial Model",
        simulations=simulations,
        markets=markets_to_save,
        selected_match=selected_match,
        context_json={"mode": "initial", "sport": sport},
    )

    if run_id:
        st.caption("Análisis guardado en tu historial.")

    st.info(
        f"Este flujo ya está listo para {sport}. "
        "Cuando agregues el engine especializado, app.py no tendrá que cambiar."
    )


# =========================================================
# INICIO
# =========================================================

logger.info("Aplicación iniciada")

initial_loading = st.empty()
with initial_loading.container():
    st.markdown(
        """
        <div class="app-loading-screen" role="status" aria-live="polite">
            <div class="app-loading-mark" aria-hidden="true"></div>
            <div class="app-loading-title">Preparando tu experiencia</div>
            <div class="app-loading-copy">
                Estamos cargando los modelos y datos necesarios para comenzar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

initialization_failed = False
try:
    create_database()
    seed_sports()
    init_state()

    auth_manager = AuthManager()
    usage_tracker = UsageTracker()
    billing_manager = BillingManager()
    admin_service = AdminService()
    try:
        auth_manager.ensure_admin_from_environment(
            os.getenv("ADMIN_EMAIL"),
            os.getenv("ADMIN_PASSWORD"),
        )
    except ValueError as error:
        logger.error("No se pudo inicializar el administrador configurado: %s", error)
except Exception as error:
    initialization_failed = True
    logger.exception("No se pudo inicializar la aplicación: %s", error)
finally:
    initial_loading.empty()

if initialization_failed:
    with st.container(border=True, horizontal_alignment="center"):
        st.title("Volvemos en un momento", text_alignment="center")
        st.write(
            "Estamos restableciendo la conexión con el servicio. "
            "Tus datos permanecen protegidos.",
            text_alignment="center",
        )
        if st.button(
            "Intentar nuevamente",
            icon=":material/refresh:",
            type="primary",
            key="retry_initialization",
        ):
            st.rerun()
    st.stop()

current_user = st.session_state.get("current_user")
if current_user:
    current_user = auth_manager.get_user(current_user["id"])
    st.session_state["current_user"] = current_user
if current_user is None:
    if st.session_state.get("public_screen") == "auth":
        render_auth_screen(auth_manager)
    else:
        render_landing_page(LOGO_PATH, PROJECT)
    st.stop()

render_account_navigation(current_user)

today_key = date.today().isoformat()
if st.session_state.get("last_cache_cleanup") != today_key:
    reviewed_runs = PostMatchService().process_cached_results()
    deleted_cache_files = cleanup_expired_cache(max_age_days=7)
    learning_enabled = (os.getenv("ENABLE_CONTINUOUS_LEARNING") or "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    learning_started = False
    if learning_enabled:
        from machine_learning.continuous_learning import start_continuous_learning

        learning_started = start_continuous_learning()
    st.session_state["last_cache_cleanup"] = today_key
    logger.info("Evaluaciones post-partido procesadas: %s", reviewed_runs)
    logger.info("Limpieza segura de caché completada: %s archivos", deleted_cache_files)
    logger.info(
        "Aprendizaje continuo habilitado: %s | iniciado: %s",
        learning_enabled,
        learning_started,
    )

if st.session_state["screen"] == "account":
    render_account_screen(current_user, billing_manager, usage_tracker, SPORT_NAMES)
elif st.session_state["screen"] == "admin":
    render_admin_screen(current_user, admin_service, billing_manager, SPORT_NAMES)
elif st.session_state["screen"] == "home":
    home_screen()
else:
    sport_screen()
