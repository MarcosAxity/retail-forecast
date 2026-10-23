# PROCESS.md · Proceso de trabajo y uso de herramientas

> Borrador para que el autor lo revise y ajuste (tiempos, detalles personales) antes de entregar.

## Enfoque

Se trabajó como un proyecto de ciencia de datos con fases cerradas por decisión documentada, no como un solo notebook:

| Fase | Pregunta | Resultado |
|---|---|---|
| 0 · Scaffold | ¿Cómo hacerlo reproducible? | Proyecto con `uv`, `uv.lock`, `src/`, `tests/` |
| 1 · EDA y calidad | ¿Qué dicen y qué no dicen los datos? | Nulos MCAR, panel con banderas de calidad, promoción asignada al azar, `total_transactions` inflado |
| 2 · Target, features y baselines | ¿Qué se pronostica, con qué y contra qué? | Target `amount_total`; `replenishment_signal` excluida por fuga; features anti-fuga con pruebas; baselines y folds walk-forward |
| 3 · Modelado y competencia | ¿Qué modelo y cómo se justifica? | LightGBM + capa de régimen; competencia de 8 modelos con significancia; argumentación de la métrica |
| 4 · Uplift promocional | ¿Las promociones venden más? | Sin venta incremental medible (−0.25 %, acotada a ±0.5 %) |
| Cierre | ¿Se puede reproducir y auditar? | Pipeline de punta a punta (`run_pipeline.py`), scripts en lugar de notebooks y carpeta `artifacts/` |

Reglas que se fijaron **antes** de mirar resultados: folds de desarrollo, de confirmación y de evento; umbral de adopción de un cambio (≥ 0.2 pt de WAPE en desarrollo);
criterios K1/K2 de la propuesta final. Los resultados que no cumplen un criterio se reportan como tales (K2 no se cumple).

## Herramientas

- **Claude (Anthropic)**, en conversaciones de un proyecto de claude.ai y en una sesión de agente con acceso al repositorio. Se usó como asistente para:
  planear fases, redactar y refactorizar código, proponer pruebas estadísticas, revisar resultados, migrar los notebooks a scripts y redactar documentación.
- **Python 3.12 · `uv`** (entorno bloqueado en `uv.lock`), pandas, NumPy, SciPy, statsmodels, scikit-learn, LightGBM, XGBoost, SHAP, matplotlib, pytest, ruff.
- **Git** para el historial de fases.

## Qué decidió y verificó la persona, y qué aportó la AI

- La persona definió el problema de negocio, el alcance de cada fase, los criterios de aceptación y qué se conserva o se descarta; revisó cada resultado antes de cerrar la fase.
- La AI generó borradores de código y análisis. Nada se aceptó sin ejecutarse: toda cifra citada en `README.md` sale de `outputs/` y se regenera con el pipeline.
- Controles: pruebas automáticas (`uv run pytest`, incluidas pruebas anti-fuga temporal que perturban el futuro y comprueban que ninguna feature cambia),
  reproducción exacta de las predicciones del backtest oficial en los pasos 09 y 10 (diferencia 0.0 MXN), y un manifiesto con sha256 de los artifacts.

## Errores encontrados al consolidar el proyecto (y cómo se corrigieron)

- `xgboost` se usaba en `competitors.py` pero no estaba declarado en `pyproject.toml`: el repositorio no era reproducible desde cero. Se declaró y se regeneró `uv.lock`.
- El notebook `06b` pedía el nivel `'total x semana'` mientras `src/models/backtest.py` define `'total × semana'`: el notebook no corría contra el código actual. El script 09 usa el nombre correcto.
- Con las versiones ahora fijadas, XGBoost y Random Forest difieren ≈ 0.01 pt de lo reportado antes en regulares, y en post-Navidad XGBoost mejora de 13.1 % a 12.0 %. La conclusión (K1 sí, K2 no) no cambia.
- En el primer borrador del análisis de uplift, las conclusiones estaban escritas a mano y afirmaban «no se distingue de cero»; al ejecutar con efectos fijos el efecto resultó ligeramente negativo (−0.25 %) con IC95 que excluye el cero.
  Las conclusiones ahora se generan a partir de los resultados y se reportan la sensibilidad a la especificación (en niveles p = 0.24) y la cota más amplia.

## Cómo reproducir

Ver `README.md` (sección «Cómo reproducirlo»). Orden de ejecución y entradas/salidas de cada paso: `uv run python run_pipeline.py --list`.
