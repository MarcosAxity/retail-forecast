# Retail Forecast · Prueba técnica de Data Scientist

**Pronóstico de ventas por tienda × categoría a 14 días y medición del efecto real de las promociones, para una cadena retail en México.**
Este README sigue el orden en que conviene revisar el proyecto: primero qué se pidió y qué se propone, después el dashboard para verlo funcionando
y, al final, toda la solución técnica.

| | |
|---|---|
| [1 · La evaluación y la propuesta](#1--la-evaluación-y-la-propuesta) | qué se pide y qué se responde |
| [2 · El dashboard](#2--el-dashboard-véalo-primero) | cómo ejecutarlo en 3 comandos y qué ver en cada pestaña |
| [3 · La solución completa](#3--la-solución-completa) | resultados, hallazgos, pipeline, decisiones y limitaciones |

---

## 1 · La evaluación y la propuesta

### Qué se pide

Una prueba técnica abierta ([enunciado](docs/ENUNCIADO.md); diccionario de datos en [`data/data_dictionary.md`](data/data_dictionary.md)): se entrega un histórico de operaciones de una cadena retail (`transactions`, `stores` y `calendar`) y
**el candidato define el problema de negocio, el enfoque, cómo validarlo y qué métricas demuestran su valor.** Se entrega un repositorio público, **funcional y reproducible**, con un
[`PROCESS.md`](PROCESS.md) que documente el proceso y el uso de herramientas de AI.

### Qué se propone

| Decisión | Respuesta |
|---|---|
| **Problema de negocio** | Cuánto reponer en cada tienda y categoría con **14 días de anticipación** (Capa 1: pronóstico) y si las promociones **realmente venden más** (Capa 2: uplift) |
| **Datos** | 80 tiendas × 6 categorías × 425 días (2023-01-01 → 2024-02-29): 204 000 filas |
| **Enfoque** | Un LightGBM global (30 variables, sin fugas de información) más una capa que corrige los días posteriores a Buen Fin y Navidad. Se compara contra 7 modelos y baselines |
| **Cómo se valida** | Backtest *walk-forward* de 48 semanas y 14 folds; reglas de aceptación fijadas **antes** de ver resultados; pruebas de significancia |
| **Métrica** | WAPE en tienda × categoría × semana (la unidad de la decisión de reposición), con el sesgo al lado |
| **Resultado** | WAPE **7.3 %** en periodos regulares, a 0.8 pts del piso de ruido (6.5 %); supera a los baselines pero **empata** con Ridge, Random Forest y XGBoost. Las promociones **no generan venta incremental** (−0.25 %, IC95 −0.46 a −0.04 %, acotado a ±0.5 %) |
| **Lo que no funciona (y se dice)** | En Buen Fin y Navidad los baselines simples le ganan al ML (K2 no se cumple); hay una sola ocurrencia de cada evento |

> La propuesta completa, con cifras, hallazgos de datos y limitaciones, está en la [sección 3](#3--la-solución-completa).

---

## 2 · El dashboard (véalo primero)

Un tablero en [Streamlit](https://streamlit.io) que muestra el modelo entrenado, simula demanda, explica el uplift y permite preguntar a los datos.
**No reentrena nada ni exige una llave de API**: lee el modelo y la evidencia que ya están en el repositorio.

### Así se ve

**1 · Modelo** — métricas globales, competencia de modelos y lectura honesta de resultados.

![Pestaña Modelo](docs/img/dashboard_1_modelo.png)

**2 · Simulador** — pronóstico de los próximos 14 días por categoría y, opcionalmente, por tienda.

![Pestaña Simulador](docs/img/dashboard_2_simulador.png)

**3 · Uplift promocional** — qué es, cómo se midió, qué resultó y qué acciones detona.

![Pestaña Uplift promocional](docs/img/dashboard_3_uplift.png)

**4 · Chat (opcional)** — preguntas sobre el histórico con gráficas, con o sin API.

![Pestaña Chat](docs/img/dashboard_4_chat.png)

### Ejecutarlo

**Requisitos previos**

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (gestor de entornos; instala solo el Python 3.12 que fija `.python-version`) y `git`.
- Nada más: sin GPU, sin llaves de API, sin datos externos, sin correr el pipeline. Después de instalar las dependencias funciona sin conexión.

```bash
git clone <este repositorio> && cd retail-forecast
uv sync                              # entorno exacto desde uv.lock
uv run streamlit run app.py          # abre http://localhost:8501
```

La primera carga tarda unos segundos mientras se arma el panel de datos; luego todo responde al instante.

<details>
<summary><b>Opciones de ejecución</b> (puerto, chat, solución de problemas)</summary>

```bash
uv run streamlit run app.py --server.port 8502             # si el 8501 está ocupado
uv run streamlit run app.py -- --chat offline              # chat guiado sin LLM (valor por defecto sin llave)
uv run streamlit run app.py -- --chat off                  # oculta la pestaña Chat
uv run streamlit run app.py -- --chat claude               # chat con Claude (requiere llave, ver abajo)
```

| `--chat` | Comportamiento |
|---|---|
| `auto` (por defecto) | Claude si hay una `ANTHROPIC_API_KEY` válida; si no, modo guiado sin LLM |
| `offline` | Modo guiado: reglas + las mismas herramientas de análisis. Sin red, sin llave |
| `claude` | Lenguaje libre con Claude. Sin llave válida avisa y cae al modo guiado |
| `off` | Quita la pestaña Chat |

También se puede fijar con la variable `CHAT_MODE`. Si algo falla: `uv sync` de nuevo (el entorno debe ser Python 3.12) y `uv run pytest tests/test_dashboard.py tests/test_chat.py` para comprobar el entorno.

</details>

### Qué hace cada pestaña

**1 · Modelo** — qué pronostica, cómo, y qué tan bien: KPI de WAPE, sesgo, piso de ruido; la competencia de 8 modelos en periodos regulares y en eventos
con una «lectura honesta» (el modelo empata con Ridge/RF/XGBoost y pierde contra baselines en eventos); backtest por periodo; importancia de variables (ganancia o SHAP) y error por segmento.

**2 · Simulador** — se elige una **categoría** (y opcionalmente una **tienda**) y devuelve el pronóstico de los 14 días siguientes al último dato (1–14 de marzo de 2024): venta, unidades estimadas,
comparación contra las 2 semanas previas, error histórico del modelo para esa selección, el desglose por formato, NSE, región y categoría, y la tabla por tienda con **descarga en CSV**.


**3 · Uplift promocional** — qué es el uplift, cómo se midió (efectos fijos, aleatorización, equivalencia), la respuesta corta (**−0.25 %**, equivalente a cero), el aviso sobre el artefacto de `total_transactions`,
la búsqueda de segmentos (0 de 35 con efecto real tras corrección), si la promoción adelanta o roba ventas de otros días y una calculadora de punto de equilibrio con márgenes ajustables. Cierra con **qué acciones detona** el resultado.


**4 · Chat (opcional)** — preguntas sobre las transacciones históricas con gráficas: rankings, evolución, efecto de Buen Fin/Navidad/quincena, crecimiento interanual, anomalías, comparaciones y promoción.
Dos modos con las mismas herramientas: **guiado** (sin LLM, funciona al clonar el repo) y **Claude** (lenguaje libre). En ambos las cifras salen de pandas, nunca del modelo de lenguaje; el interruptor «Detalle técnico» muestra qué consulta se ejecutó.


<details>
<summary><b>Activar el chat con Claude</b> (opcional)</summary>

```bash
uv sync --extra chat                 # instala el SDK de Anthropic
cp .env.example .env                 # y escribe tu ANTHROPIC_API_KEY en .env (no se versiona)
uv run streamlit run app.py          # modo auto: detecta la llave y activa Claude
```

El modelo se cambia con `CLAUDE_MODEL` en `.env`. Claude llama a las herramientas del tablero (máx. 6 pasos por pregunta); si la API falla, la pestaña responde en modo guiado.

</details>

### De dónde salen los números del dashboard

Todo se lee de `artifacts/` y `outputs/` (versionados) y de `data/raw/`; las cifras son las mismas del pipeline. El pronóstico del simulador viene de `outputs/13_pronostico_proximos_14_dias/`
y, si faltara, se recalcula con el modelo guardado. Las unidades son una **estimación** (monto ÷ precio unitario reciente): el modelo pronostica monto. El histórico no trae SKU, por eso el simulador trabaja por categoría y tienda.

---

## 3 · La solución completa

### Resultados en una página

| Tema | Resultado |
|---|---|
| Modelo final | LightGBM global (30 features, objetivo L2 sobre log del monto) + capa de reinicio de régimen tras Buen Fin y Navidad |
| Periodos regulares (9 folds) | WAPE **7.3 %** (serie × semana) contra 7.8 % de `sma4`, 7.8 % de `ma28`, 9.1 % de `snaive` y 9.5 % de ETS |
| Competidores de ML | Ridge 7.26 %, Random Forest 7.32 %, XGBoost 7.34 %: empate técnico con la propuesta; no se afirma que sea «el mejor» |
| Eventos (Buen Fin, Navidad) | Los baselines `ma28` y `sma4` superan a los modelos de ML; es la limitación principal y está documentada |
| Piso de ruido | 15.6 % (serie × día) · 6.5 % (serie × semana) · 3.4 % (tienda × semana) · 2.1 % (total): el modelo está cerca del mínimo alcanzable |
| Uplift promocional | Efecto sobre el monto de **−0.25 %** (IC95 −0.46 a −0.04 %): sin venta incremental; acotado dentro de ±0.5 % |

Las cifras salen de `outputs/` y se regeneran con el pipeline. Al fijar las versiones de las librerías (`xgboost` no estaba declarado
en `pyproject.toml`) Random Forest y XGBoost difieren ≈ 0.01 pt de lo documentado antes en periodos regulares; en post-Navidad XGBoost
con capa queda en 12.0 % (antes 13.1 %), así que la propuesta pasa del 3.º al 4.º puesto de 8. Las conclusiones no cambian: K1 se cumple y K2 no.

### Hallazgos de datos que condicionan el diseño

El análisis completo, con evidencia y decisiones, está en [`docs/EDA.md`](docs/EDA.md) (documento único del EDA). Resumen:

1. **`replenishment_signal` es fuga de información.** Es el promedio de `total_transactions` en la ventana [t−5, t+1] (coincide en 99.76 % de las filas) y la demanda del día objetivo se reconstruye con error < 0.5 transacciones en el 99.7 % de los casos. Se excluye (script 04).
2. **`total_transactions` está inflado ≈ 22 % solo en días con promoción** y no coincide con efectivo + tarjeta; no mide demanda. El target es `amount_total` (script 05).
3. **La promoción se asignó al azar** (≈ 20 % por serie-día, sin memoria, sin relación con segmento ni con la venta reciente): es un experimento natural y permite estimar uplift sin sesgo de selección (script 03).
4. **Los nulos son fallas puntuales independientes (MCAR)**: 5.9 % de efectivo, 3.0 % de unidades, 2.4 % de ticket (script 02).
5. **El nivel lo explica el formato de tienda** (R² = 0.78; 0.89 con todos los atributos); la **dinámica** la dominan Buen Fin (×2.7–3.4 el nivel base) y Navidad (script 03).
6. **El ruido irreducible fija el techo de precisión** (σ residual ≈ 0.20 en log, sin autocorrelación): ver tabla de pisos arriba.

### Cómo reproducirlo

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.12 (se fija en `.python-version`; `uv` lo instala solo).

```bash
git clone <este repositorio> && cd retail-forecast
uv sync                                   # entorno exacto desde uv.lock
uv run python run_pipeline.py             # pipeline completo (≈ 6–10 min en 4 núcleos)
uv run pytest                             # 70 pruebas (incluye anti-fuga temporal, pronóstico, dashboard y chat)
uv run streamlit run app.py               # dashboard sobre los resultados regenerados
```

Los datos originales están en `data/raw/`. Todo lo aleatorio usa semilla fija. No hace falta correr el pipeline para ver el dashboard: los resultados ya están en `artifacts/` y `outputs/`.

```bash
uv run python run_pipeline.py --list              # pasos, entradas y salidas, sin ejecutar
uv run python run_pipeline.py --only 11           # un paso
uv run python run_pipeline.py --from 07 --to 10   # un tramo
uv run python run_pipeline.py --skip 10           # todo menos la competencia de modelos (el paso más lento)
uv run python scripts/11_uplift_promocional.py    # o ejecutar un script directamente
```

### Orden de ejecución

Cada paso es un script de `scripts/` (sin notebooks). Cada uno escribe `log.txt`, `tabla_XX.csv` y `fig_XX.png`
en `outputs/<paso>/`, en el mismo orden en que aparecen en el script. `run_pipeline.py` registra el estado y
la duración de cada paso en `outputs/pipeline_run.json`.

| Paso | Script | Qué hace | Lee | Escribe |
|---|---|---|---|---|
| 00 | `00_preparar_datos` | Panel maestro, features h=14 y baselines | `data/raw/` | `data/processed/`, `outputs/reports/baselines_h14.csv` |
| 01 | `01_eda_calidad_y_dinamica` | EDA 1/3: estructura, calidad, segmentos, dinámica temporal, promociones | `data/raw/` | `outputs/01_*/` |
| 02 | `02_eda_nulos_consistencias_calendario` | EDA 2/3: nulos (¿MCAR?), consistencias, calendario, extremos | `data/raw/` | `outputs/02_*/` |
| 03 | `03_eda_tiendas_promocion_ruido` | EDA 3/3: atributos de tienda, asignación aleatoria de promociones, piso de ruido | `data/raw/` | `outputs/03_*/` |
| 04 | `04_auditoria_replenishment` | Auditoría de fuga de información de `replenishment_signal` | `data/raw/` | `outputs/04_*/` |
| 05 | `05_definicion_target` | Monto vs. unidades vs. transacciones como target | `data/raw/` | `outputs/05_*/` |
| 06 | `06_features_y_baselines` | Features anti-fuga, esquema de validación, baselines | `data/raw/` | `outputs/06_*/` |
| 07 | `07_modelado_lightgbm` | Backtest, ablaciones, hiperparámetros, SHAP | `data/processed/` | `outputs/07_*/`, `configs/model_params.json` |
| 08 | `08_backtest_oficial_y_modelo_final` | Backtest oficial (8 folds) y modelo final | `configs/` | `outputs/reports/`, `outputs/models/` |
| 09 | `09_modelado_eventos` | Reinicio de régimen y target multiplicativo (backtest continuo de 48 semanas) | `outputs/reports/` | `outputs/09_*/` |
| 10 | `10_comparacion_modelos` | Competencia de 8 modelos, significancia y argumentación de la métrica | `outputs/reports/` | `outputs/10_*/` |
| 11 | `11_uplift_promocional` | Uplift promocional: efectos fijos, aleatorización, heterogeneidad, dinámica | `data/processed/` | `outputs/11_*/` |
| 12 | `12_exportar_artifacts` | Reúne el modelo y su evidencia con manifiesto de integridad | `outputs/`, `configs/` | `artifacts/` |
| 13 | `13_pronostico_proximos_14_dias` | Pronóstico de los 14 días siguientes al último dato (alimenta el simulador del dashboard) | `artifacts/model/`, `data/raw/` | `outputs/13_*/` |

Los pasos 01–06 dependen solo de los datos; 07 → 08 → (09, 10) deben ir en ese orden porque 09 y 10 reutilizan las
predicciones del backtest oficial y verifican que se reproducen (diferencia 0.0 MXN).

### Carpeta de artifacts

`artifacts/` contiene lo necesario para auditar y reutilizar el modelo (`manifest.json` lista el sha256 de cada archivo):

```
artifacts/
  model/        lgbm_h14.joblib · model_spec.json (features, hiperparámetros, capa de régimen, protocolo) · feature_importance.csv
  metrics/      backtest_metrics_h14.csv · baselines_h14.csv · comparacion_regulares.csv · comparacion_eventos.csv
                significancia_vs_competidores.csv · uplift_resumen.json
  environment/  versions.txt · data_fingerprint.json   (el entorno exacto está en uv.lock)
  manifest.json
```

### Estructura del repositorio

```
app.py           entrada del dashboard       src/dashboard/    pestañas, gráficos y chat del dashboard
data/raw/        datos originales            data/processed/   generado (no versionado)
scripts/         pasos del pipeline          src/              lógica reutilizable (data, features, models, dashboard)
outputs/         salidas por paso            artifacts/        modelo y evidencia exportados
configs/         hiperparámetros elegidos    tests/            pruebas (anti-fuga, backtest, regímenes, dashboard, chat)
docs/            EDA.md e imágenes           notebooks/        versión exploratoria original (los scripts son la fuente de verdad)
.streamlit/      tema del dashboard          .env.example      variables del chat opcional con Claude
```

Correspondencia notebook → script: `01_eda` → 01 · `02_eda_complementario` → 02 y 03 · `03_auditoria_replenishment_signal` → 04 ·
`04_definicion_del_target` → 05 · `05_features_y_baselines` → 06 · `06_modelado` → 07 · `06b_modelado_eventos` → 09 · `07_comparacion_modelos` → 10.

### Decisiones metodológicas

- **Por qué WAPE** (script 10): la decisión es la reposición semanal por tienda y categoría, así que el error se mide como
  % de venta mal planeada en ese nivel. En estos datos MAE y RMSE dan el mismo ranking; WAPE se elige por
  interpretación, comparabilidad entre niveles y ausencia de escala. MAPE se descarta porque flatera el subpronóstico
  (comprobado con modelos sesgados a propósito); el sesgo se reporta aparte porque, solo, engaña.
- **Validación:** walk-forward con corte = inicio del fold − horizonte (14 días). Los folds de desarrollo, de
  confirmación y de evento se fijaron antes de mirar resultados; un cambio se adopta solo si mejora ≥ 0.2 pt en desarrollo.
- **Criterios de aceptación fijados antes de ejecutar:** K1 (regulares) se cumple; K2 (top 2 en post-evento) **no** se cumple.
- **Uplift** (script 11): contraste con efectos fijos de serie y de fecha (el diseño es aleatorio), errores agrupados por
  tienda, prueba de aleatorización por permutaciones, equivalencia (TOST), efecto mínimo detectable, heterogeneidad con
  corrección de Benjamini-Hochberg, event study de ±7 días y días consecutivos de promoción.

### Limitaciones

- Una sola ocurrencia de Buen Fin y Navidad: las ventanas de la capa de régimen son hipótesis y no pueden validarse hasta ver un segundo año.
- En eventos, los baselines simples superan al ML; la política híbrida (baseline limpio con alerta en eventos y ML fuera de ellos) queda como trabajo futuro.
- El dataset no trae costos, descuentos ni márgenes de las promociones: el análisis de uplift mide venta, no rentabilidad (la tabla de punto de equilibrio usa márgenes ilustrativos).
- Pendientes: cuantiles p10/p90 y ensamble Ridge + LightGBM.

Proceso de trabajo y uso de herramientas de AI: ver [`PROCESS.md`](PROCESS.md).
