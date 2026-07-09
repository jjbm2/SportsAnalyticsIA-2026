import streamlit as st

from core.settings import APP_NAME, APP_VERSION, DEFAULT_SIMULATIONS
from core.logger import logger

st.set_page_config(
    page_title=APP_NAME,
    layout="wide"
)

logger.info("Aplicación iniciada")

st.title(f"{APP_NAME} v{APP_VERSION}")
st.write("Motor inicial de análisis deportivo basado en datos.")

st.divider()

deporte = st.selectbox(
    "Selecciona el deporte",
    ["Fútbol", "Béisbol", "Basketball", "NFL"]
)

col1, col2 = st.columns(2)

with col1:
    equipo_local = st.text_input("Equipo local", "Francia")

with col2:
    equipo_visitante = st.text_input("Equipo visitante", "Marruecos")

simulaciones = st.number_input(
    "Número de simulaciones",
    min_value=10_000,
    max_value=1_000_000,
    value=DEFAULT_SIMULATIONS,
    step=10_000
)

if st.button("Analizar partido"):
    logger.info(f"Análisis solicitado: {equipo_local} vs {equipo_visitante}")

    st.success("Análisis recibido correctamente")

    st.write("Deporte:", deporte)
    st.write("Partido:", f"{equipo_local} vs {equipo_visitante}")
    st.write("Simulaciones:", simulaciones)

    st.info("En la siguiente fase conectaremos este botón con el motor estadístico.")