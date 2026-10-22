# retail-forecast

Pronóstico de ventas por tienda y categoría y medición del efecto incremental de las promociones
para una cadena retail en México.

> Estado: scaffold inicial (Fase 0). El README se completa al cierre del proyecto.

## Problema
1. **Capa 1 – Forecast:** predecir `amount_total` por tienda × categoría × día (evaluado por semana) para planear reposición.
2. **Capa 2 – Uplift promocional:** medir si las promociones generan venta incremental real y dónde.

## Entorno
```bash
uv sync
uv run python -c "import lightgbm"
```

## Pipeline (Fases 0–2)
```bash
uv run python -m src.data.preprocess                    # panel maestro → data/processed/panel.parquet
uv run python -m src.features.engineering --horizon 14  # features → data/processed/features_h14.parquet
uv run python -m src.models.baselines --horizon 14      # baselines → outputs/reports/baselines_h14.csv
uv run pytest                                           # tests (incluye pruebas anti-fuga temporal)
```
Decisiones clave: el target es `amount_total` (las unidades se derivan), `replenishment_signal` se excluye
por fuga de información y la meta se mide en tienda × categoría × semana con WAPE contra baselines.

## Estructura
```
data/raw/        datos originales (transactions, stores, calendar)
data/processed/  dataset maestro (generado)
notebooks/       01 EDA · 02 EDA complementario · 03 auditoría de replenishment_signal
                 04 definición del target · 05 features y baselines
src/data/        carga y limpieza
src/features/    ingeniería de variables
src/models/      entrenamiento, evaluación y uplift
src/dashboard/   app Streamlit
outputs/         modelos y reportes
```
