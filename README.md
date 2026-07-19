# SportsAnalyticsAI

Plataforma Streamlit de análisis deportivo con modelos validados, simulación estadística, historial persistente y evaluación post-partido.

## Funciones principales

- Fútbol, béisbol, baloncesto, NFL, Fórmula 1, hockey y MMA.
- Predictores especializados con fallback estadístico cuando el modelo ML no supera el control de calidad.
- Probabilidades, confianza, riesgo y explicación por mercado.
- Historial y evaluaciones post-partido en SQLite o PostgreSQL.
- Registro, autenticación, planes, límites de uso y administración manual de pagos.
- Caché compartido para evitar solicitudes duplicadas a los proveedores.
- Backtesting temporal y aprendizaje continuo sujeto a reglas de promoción.

## Configuración local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

La aplicación queda disponible normalmente en `http://localhost:8501`.

## Variables de entorno

Crea un archivo `.env` local. Nunca lo subas al repositorio.

```env
API_SPORTS_KEY=
SPORTMONKS_API_TOKEN=
BALLDONTLIE_API_KEY=
SPORTSDATA_API_KEY=
SPORTSDATA_SOCCER_COMPETITIONS=3
DATABASE_URL=
ADMIN_EMAIL=
ADMIN_PASSWORD=
ENABLE_CONTINUOUS_LEARNING=false
```

`SPORTSDATA_SOCCER_COMPETITIONS` acepta identificadores o claves separados por
coma (por ejemplo `3,EPL,MLS`). La prueba gratuita de SportsDataIO ofrece la
Champions League (`3`); configura únicamente competiciones incluidas en tu
suscripción para no desperdiciar solicitudes.

`DATABASE_URL` es opcional en local. Sin ella se utiliza SQLite; en producción se recomienda PostgreSQL persistente.

## Validación

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q app.py auth billing admin core database engines machine_learning promotions services usage
.\.venv\Scripts\python.exe -m pip check
```

## Backtesting

```powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py --help
```

El backtesting usa orden temporal y evita incorporar información posterior al partido analizado.

## Seguridad y datos

- Las contraseñas se almacenan con bcrypt.
- Las claves de proveedores se leen únicamente del entorno.
- Los comprobantes de pago se eliminan al aprobarse o rechazarse.
- Los modelos activos no se sustituyen si el candidato no supera las métricas de promoción.
- Los errores técnicos se registran internamente y no se muestran al usuario final.
