import random
import time

import pandas as pd
import streamlit as st

from core.version import PROJECT, VERSION
from core.constants import (
    MIN_SIMULATIONS,
    DEFAULT_SIMULATIONS,
    MAX_SIMULATIONS,
    SIMULATION_STEP,
)
from core.paths import LOGO_PATH
from core.logger import logger


st.set_page_config(
    page_title=PROJECT,
    layout="wide"
)


st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
    }

    .metric-card {
        background-color: #121212;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #263238;
    }

    .metric-title {
        color: #B0BEC5;
        font-size: 16px;
    }

    .metric-value {
        color: #00C853;
        font-size: 36px;
        font-weight: 800;
    }

    .section-title {
        color: #F5F5F5;
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
    }

    .small-text {
        color: #B0BEC5;
        text-align: center;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


logger.info("Aplicación iniciada")


col_logo_1, col_logo_2, col_logo_3 = st.columns([1, 2, 1])

with col_logo_2:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.markdown(f"<h1 style='text-align:center'>{PROJECT}</h1>", unsafe_allow_html=True)

st.markdown(
    f"<div class='small-text'>Versión {VERSION} | Motor inicial de análisis deportivo</div>",
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("Configuración")

    deporte = st.selectbox(
        "Deporte",
        ["Fútbol", "Béisbol", "Basketball", "NFL"]
    )

    competencia = st.selectbox(
        "Competencia",
        ["Mundial", "Champions League", "Liga MX", "MLB", "NBA", "NFL"]
    )

    fecha = st.date_input("Fecha del partido")

    simulaciones = st.number_input(
        "Simulaciones",
        min_value=MIN_SIMULATIONS,
        max_value=MAX_SIMULATIONS,
        value=DEFAULT_SIMULATIONS,
        step=SIMULATION_STEP
    )


st.markdown("<div class='section-title'>Selección del partido</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    equipo_local = st.text_input("Equipo local", "Francia")

with col2:
    equipo_visitante = st.text_input("Equipo visitante", "Marruecos")


analizar = st.button("Analizar partido", use_container_width=True)


if analizar:
    logger.info(f"Análisis solicitado: {equipo_local} vs {equipo_visitante}")

    st.markdown("<div class='section-title'>Proceso de análisis</div>", unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    sim_counter = st.empty()

    local_wins = 0
    draws = 0
    visitor_wins = 0
    total_goals_over_25 = 0
    under_35_goals = 0
    over_85_corners = 0
    local_over_45_corners = 0

    start_time = time.time()

    for sim in range(1, simulaciones + 1):
        local_goals = random.choices(
            [0, 1, 2, 3, 4, 5],
            weights=[12, 24, 28, 20, 11, 5]
        )[0]

        visitor_goals = random.choices(
            [0, 1, 2, 3, 4, 5],
            weights=[22, 30, 24, 14, 7, 3]
        )[0]

        local_corners = random.randint(2, 9)
        visitor_corners = random.randint(1, 8)
        total_corners = local_corners + visitor_corners

        if local_goals > visitor_goals:
            local_wins += 1
        elif local_goals == visitor_goals:
            draws += 1
        else:
            visitor_wins += 1

        if local_goals + visitor_goals > 2.5:
            total_goals_over_25 += 1

        if local_goals + visitor_goals < 3.5:
            under_35_goals += 1

        if total_corners > 8.5:
            over_85_corners += 1

        if local_corners > 4.5:
            local_over_45_corners += 1

        percent = int((sim / simulaciones) * 100)

        if sim == 1 or sim % max(1, simulaciones // 100) == 0 or sim == simulaciones:
            elapsed = time.time() - start_time
            speed = sim / elapsed if elapsed > 0 else 0

            status_text.write(f"{percent}% - Ejecutando simulación Monte Carlo")
            sim_counter.write(
                f"Simulación {sim:,} de {simulaciones:,} | "
                f"{speed:,.0f} simulaciones/segundo"
            )
            progress_bar.progress(percent)

    st.success("Análisis completado")

    local_win_pct = (local_wins / simulaciones) * 100
    draw_pct = (draws / simulaciones) * 100
    visitor_win_pct = (visitor_wins / simulaciones) * 100

    over_25_pct = (total_goals_over_25 / simulaciones) * 100
    under_35_pct = (under_35_goals / simulaciones) * 100
    over_85_corners_pct = (over_85_corners / simulaciones) * 100
    local_over_45_corners_pct = (local_over_45_corners / simulaciones) * 100

    st.markdown("<div class='section-title'>Resumen principal</div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Victoria {equipo_local}</div>
            <div class='metric-value'>{local_win_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Empate</div>
            <div class='metric-value'>{draw_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Victoria {equipo_visitante}</div>
            <div class='metric-value'>{visitor_win_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Sports Confidence Index</div>", unsafe_allow_html=True)

    col_sci1, col_sci2, col_sci3 = st.columns(3)

    with col_sci1:
        st.metric("SCI", "Demo")

    with col_sci2:
        st.metric("Modelo usado", "Monte Carlo")

    with col_sci3:
        st.metric("Simulaciones reales", f"{simulaciones:,}")

    st.markdown("<div class='section-title'>Mercados principales</div>", unsafe_allow_html=True)

    results = pd.DataFrame({
        "Mercado": [
            f"{equipo_local} gana",
            "Empate",
            f"{equipo_visitante} gana",
            "Over 2.5 goles",
            "Under 3.5 goles",
            "Over 8.5 corners",
            f"{equipo_local} over 4.5 corners"
        ],
        "Probabilidad": [
            f"{local_win_pct:.1f}%",
            f"{draw_pct:.1f}%",
            f"{visitor_win_pct:.1f}%",
            f"{over_25_pct:.1f}%",
            f"{under_35_pct:.1f}%",
            f"{over_85_corners_pct:.1f}%",
            f"{local_over_45_corners_pct:.1f}%"
        ],
        "Confianza": ["Demo"] * 7,
        "Riesgo": ["Demo"] * 7
    })

    st.dataframe(results, use_container_width=True)

    st.info(
        "Este modelo ya ejecuta el número real de simulaciones seleccionado. "
        "Todavía usa datos simulados; el siguiente paso será conectar el Database Engine y después APIs deportivas reales."
    )