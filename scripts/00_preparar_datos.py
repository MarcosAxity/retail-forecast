"""00 · Preparación de datos: panel maestro, features (h = 14) y baselines.

Ejecuta, en orden, los módulos de `src/`:
  1. src.data.preprocess        → data/processed/panel.parquet
  2. src.features.engineering   → data/processed/features_h14.parquet
  3. src.models.baselines       → outputs/reports/baselines_h14.csv

Ejecutar desde la raíz del repo:  uv run python scripts/00_preparar_datos.py
Entradas : data/raw/{transactions,stores,calendar}.csv
Salidas  : data/processed/ (no versionado) · outputs/00_preparar_datos/log.txt · outputs/reports/baselines_h14.csv
"""
import runpy
import sys

from _common import stage

stage("00_preparar_datos")
for mod, argv in [("src.data.preprocess", []), ("src.features.engineering", ["--horizon", "14"]),
                  ("src.models.baselines", ["--horizon", "14"])]:
    print(f"\n=== python -m {mod} {' '.join(argv)}")
    sys.argv = [mod, *argv]
    runpy.run_module(mod, run_name="__main__")
