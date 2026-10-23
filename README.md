# retail-forecast

Pronóstico de ventas por tienda × categoría para planear la reposición de una cadena retail en México
(**Capa 1**) y medición del efecto incremental de sus promociones (**Capa 2 · uplift**).

- **Datos:** 80 tiendas × 6 categorías × 425 días (2023-01-01 → 2024-02-29) = 204 000 filas (42 ausentes).
- **Decisión de negocio:** cuánto reponer por tienda y categoría con 14 días de anticipación.
- **Métrica principal:** WAPE en tienda × categoría × semana, con el sesgo al lado (ver «Por qué WAPE»).

## Resultados en una página

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

## Cómo reproducirlo

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.12 (se fija en `.python-version`; `uv` lo instala solo).

```bash
git clone <este repositorio> && cd retail-forecast
uv sync                                   # entorno exacto desde uv.lock
uv run python run_pipeline.py             # pipeline completo (≈ 6–10 min en 4 núcleos)
uv run pytest                             # 23 pruebas (incluye pruebas anti-fuga temporal)
```

Los datos originales están en `data/raw/`. Todo lo aleatorio usa semilla fija.

```bash
uv run python run_pipeline.py --list              # pasos, entradas y salidas, sin ejecutar
uv run python run_pipeline.py --only 11           # un paso
uv run python run_pipeline.py --from 07 --to 10   # un tramo
uv run python run_pipeline.py --skip 10           # todo menos la competencia de modelos (el paso más lento)
uv run python scripts/11_uplift_promocional.py    # o ejecutar un script directamente
```

## Orden de ejecución

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

Los pasos 01–06 dependen solo de los datos; 07 → 08 → (09, 10) deben ir en ese orden porque 09 y 10 reutilizan las
predicciones del backtest oficial y verifican que se reproducen (diferencia 0.0 MXN).

## Carpeta de artifacts

`artifacts/` contiene lo necesario para auditar y reutilizar el modelo (`manifest.json` lista el sha256 de cada archivo):

```
artifacts/
  model/        lgbm_h14.joblib · model_spec.json (features, hiperparámetros, capa de régimen, protocolo) · feature_importance.csv
  metrics/      backtest_metrics_h14.csv · baselines_h14.csv · comparacion_regulares.csv · comparacion_eventos.csv
                significancia_vs_competidores.csv · uplift_resumen.json
  environment/  versions.txt · data_fingerprint.json   (el entorno exacto está en uv.lock)
  manifest.json
```

## Estructura del repositorio

```
data/raw/        datos originales            data/processed/   generado (no versionado)
scripts/         pasos del pipeline          src/              lógica reutilizable (data, features, models)
outputs/         salidas por paso            artifacts/        modelo y evidencia exportados
configs/         hiperparámetros elegidos    tests/            pruebas (anti-fuga, backtest, regímenes, targets)
notebooks/       versión exploratoria original (referencia; los scripts son la fuente de verdad)
```

Correspondencia notebook → script: `01_eda` → 01 · `02_eda_complementario` → 02 y 03 · `03_auditoria_replenishment_signal` → 04 ·
`04_definicion_del_target` → 05 · `05_features_y_baselines` → 06 · `06_modelado` → 07 · `06b_modelado_eventos` → 09 · `07_comparacion_modelos` → 10.

## Decisiones metodológicas

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

## Limitaciones

- Una sola ocurrencia de Buen Fin y Navidad: las ventanas de la capa de régimen son hipótesis y no pueden validarse hasta ver un segundo año.
- En eventos, los baselines simples superan al ML; la política híbrida (baseline limpio con alerta en eventos y ML fuera de ellos) queda como trabajo futuro.
- El dataset no trae costos, descuentos ni márgenes de las promociones: el análisis de uplift mide venta, no rentabilidad (la tabla de punto de equilibrio usa márgenes ilustrativos).
- Pendientes: cuantiles p10/p90 y ensamble Ridge + LightGBM.

Proceso de trabajo y uso de herramientas de AI: ver [`PROCESS.md`](PROCESS.md).
