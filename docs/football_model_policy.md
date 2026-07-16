# Política del modelo de fútbol

La aplicación intenta crear `FootballPredictor` antes de ejecutar un análisis de
fútbol. Si faltan sus artefactos, `app.py` captura `FileNotFoundError` y conserva
el servicio mediante `FootballPredictionEngine` con Poisson y Monte Carlo.

Tener archivos de modelo no basta para usar sus probabilidades. Cada mercado ML
debe demostrar calidad fuera de muestra:

- resultado: exactitud superior al baseline y mejora mínima de 0.02;
- over 2.5: AUC de al menos 0.50;
- ambos anotan: AUC de al menos 0.50.

Los mercados que no superan estas reglas usan únicamente la estimación Poisson y
Monte Carlo. El contexto de cada corrida guarda `quality_gate` y
`ml_probabilities`, de modo que se puede auditar qué parte del análisis utilizó
ML. Un candidato nunca se promueve automáticamente; requiere resultados reales,
comparación emparejada y confirmación explícita.

La persistencia de una corrida y todos sus mercados comparte una transacción
SQLite. Cualquier error ejecuta rollback y evita que queden corridas o mercados
parciales.

## Consenso de modelos

La probabilidad pública es un consenso, no la salida ciega de un solo algoritmo.
Los modelos supervisados calificados se combinan con el motor estadístico y Monte
Carlo. Los candidatos nuevos permanecen en sombra hasta demostrar una mejora de
AUC de al menos 0.02 sin degradar Brier ni el error de calibración. Basketball
puede entrenar además un ensamble interno de Random Forest y regresión logística
calibrados; su salida combinada pasa por las mismas reglas antes de publicarse.
