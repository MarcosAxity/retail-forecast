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

## Estructura
```
data/raw/        datos originales (transactions, stores, calendar)
data/processed/  dataset maestro (generado)
notebooks/       EDA
src/data/        carga y limpieza
src/features/    ingeniería de variables
src/models/      entrenamiento, evaluación y uplift
src/dashboard/   app Streamlit
outputs/         modelos y reportes
```
