# AGENTS.md

## Proyecto
SportsAnalyticsAI

---

# 🎯 OBJETIVO FINAL

Convertir el sistema en un motor de predicción deportiva:

- confiable
- estable
- explicable
- auto-mejorable
- sin intervención manual
- sin degradación de modelos
- sin lógica de apuestas (NO cuotas, NO EV)

---

# 🧠 MODO DE EJECUCIÓN

Codex debe:

- analizar estado actual del repo
- NO reiniciar proyecto
- NO rehacer código existente
- trabajar solo sobre lo pendiente
- ejecutar por fases automáticamente
- no detenerse entre pasos pequeños

Solo detenerse si:
- hay bloqueo externo
- hay riesgo de corrupción de datos

---

# 🚫 REGLAS CRÍTICAS

- no romper app.py
- no eliminar SQLite
- no eliminar modelos
- no mostrar debug al usuario
- no usar apuestas ni cuotas
- no entrenar en caliente sin validación
- cambios mínimos pero correctos

---

# 🔥 PRIORIDAD ACTUAL

## FASE 1 — VALIDACIÓN TOTAL FÚTBOL

Validar flujo completo:

- predictor → fallback → SQLite → mercados → contexto → historial

Requisitos:

- atomicidad total
- cero registros parciales
- test end-to-end obligatorio

---

## FASE 2 — POST-MATCH INTELIGENTE

Flujo automático:

1. detectar partido finalizado
2. buscar predicción previa
3. comparar con resultado real
4. guardar evaluación

Persistencia:
- post_match_reviews
- métricas por mercado

---

## FASE 3 — AUTOAPRENDIZAJE

Flujo:

1. acumular resultados
2. generar dataset incremental
3. limpiar datos inválidos
4. entrenar candidato
5. comparar contra modelo activo
6. promover solo si mejora

---

## REGLAS DE PROMOCIÓN

- AUC mejora ≥ +0.02
- no empeora Brier
- no empeora calibración

---

# 🧠 MEJORA DE MODELO (PRIORIDAD)

## BACKTESTING

Crear módulo:

machine_learning/backtesting/

Función:

- usar temporadas pasadas
- simular predicciones históricas
- generar dataset masivo

Reglas:
- NO usar datos futuros
- respetar orden temporal

---

## FEATURES AVANZADAS

Agregar:

- forma últimos 5 partidos
- goles anotados/recibidos
- rachas
- clean sheets
- BTTS rate
- home advantage
- consistencia de goles

---

## CALIBRACIÓN

Implementar:

- Platt Scaling o Isotonic

Aplicar en:
FootballPredictor

---

## ENSEMBLE INTELIGENTE

- weighting dinámico
- priorizar modelo más confiable
- ignorar modelos débiles

---

## ERROR ANALYSIS

Guardar:

- predicción vs resultado
- errores por liga
- errores por mercado

---

# 🧠 SISTEMA DE CONFIANZA

Cada predicción debe incluir:

- probabilidad
- confidence_score

Basado en:

- datos históricos
- consistencia
- acuerdo entre modelos
- calidad del modelo

---

# 🧠 EXPLICACIONES

Generar texto automático:

Ejemplo:

"Alta probabilidad de goles debido a alto promedio ofensivo y baja defensa."

Reglas:
- usar datos reales
- no texto genérico

---

# 🎨 UI PROFESIONAL

## MOSTRAR

1. resultado principal
2. probabilidad
3. confianza
4. explicación

---

## FILTRO DE MERCADOS

- mostrar solo ≥50%
- excepción: 1X2

---

## DISEÑO

- tarjetas limpias
- colores:
  - verde (>65%)
  - amarillo (55–65%)
- sin tablas largas

---

## RESPONSIVE

- 2 columnas desktop
- 1 columna móvil

---

## OCULTAR

- JSON
- debug
- logs

---

# 🚫 PARTIDOS FINALIZADOS

- no mostrar en análisis previo
- mover a historial

---

# 🧹 LIMPIEZA

Eliminar:
- cache viejo

No eliminar:
- SQLite
- modelos
- historial

---

# 🧪 TESTING

- test end-to-end
- test SQLite
- test fallback
- test post-match

---

# 🎯 DEFINICIÓN DE TERMINADO

Sistema completo cuando:

- IA predice correctamente
- IA mejora automáticamente
- probabilidades calibradas
- explicaciones claras
- UI limpia
- sin intervención manual