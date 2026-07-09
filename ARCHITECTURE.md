# SportsAnalyticsAI - Arquitectura del Sistema

## Objetivo

SportsAnalyticsAI es una plataforma de análisis deportivo basada en estadística, inteligencia artificial y simulación.

El sistema analizará partidos de diferentes deportes y generará probabilidades objetivas basadas en datos verificables.

## Deportes soportados

- Fútbol
- Béisbol
- Basketball
- NFL

## Motores principales

### Interface Engine

Gestiona la interfaz del usuario.

Inicialmente usaremos Streamlit como laboratorio de pruebas.

Más adelante se añadirá PySide6 para una aplicación de escritorio profesional.

### API Engine

Se encargará de conectarse a APIs deportivas externas.

### Data Engine

Descarga, limpia, valida y normaliza los datos deportivos.

### Database Engine

Guarda equipos, jugadores, partidos, estadísticas, predicciones y resultados históricos.

### Feature Engine

Crea variables predictivas como forma reciente, goles esperados, tiros, corners, descanso, localía y rendimiento individual.

### Statistical Engine

Ejecuta modelos estadísticos como Poisson, Binomial, Bayes y Monte Carlo.

### AI Engine

Ejecuta modelos de machine learning como Random Forest, XGBoost, LightGBM y redes neuronales.

### Prediction Engine

Combina todos los modelos y genera probabilidades finales.

### Visualization Engine

Genera gráficas, tablas, dashboards y visualizaciones.

### Report Engine

Exporta análisis en PDF, Excel, CSV y HTML.

## Flujo general

Usuario selecciona deporte, competición, fecha y partido.

El sistema descarga datos desde APIs deportivas.

Los datos se limpian y guardan en la base de datos.

El Feature Engine crea variables predictivas.

El Statistical Engine y AI Engine generan predicciones.

El Prediction Engine fusiona resultados.

El Visualization Engine muestra gráficas y tablas.

El Report Engine permite exportar resultados.

## Base de datos

Se usará SQLite en la primera versión.

Tablas principales:

- competitions
- teams
- players
- matches
- match_statistics
- player_statistics
- predictions
- model_results
- logs

## Versión actual

v0.1.0 Foundation