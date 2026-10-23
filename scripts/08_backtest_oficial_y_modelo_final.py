"""08 · Backtest oficial (8 folds) y entrenamiento del modelo final LightGBM (h = 14).

Ejecuta `src.models.train` con los hiperparámetros de configs/model_params.json (paso 07).
Los scripts 09 y 10 leen estas predicciones como referencia y verifican que se reproducen.

Ejecutar desde la raíz del repo:  uv run python scripts/08_backtest_oficial_y_modelo_final.py
Entradas : data/processed/features_h14.parquet · configs/model_params.json
Salidas  : outputs/reports/backtest_{predictions,metrics}_h14.parquet|csv · outputs/models/lgbm_h14.joblib
           outputs/08_backtest_oficial_y_modelo_final/log.txt
"""
import runpy
import sys

from _common import stage

stage("08_backtest_oficial_y_modelo_final")
sys.argv = ["src.models.train", "--horizon", "14"]
runpy.run_module("src.models.train", run_name="__main__")
