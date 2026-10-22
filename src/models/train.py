"""Entrenamiento y backtest del modelo de pronóstico (LightGBM sobre log del monto).

Protocolo de validación (ver `backtest.py`): cada fold entrena con fechas <= inicio − horizonte y
pronostica el fold completo. Los folds se dividen *antes* de mirar resultados:
- desarrollo (se usan para elegir hiperparámetros y variantes): semana_santa_2023, may_jun_2023, sep_2023
- confirmación (solo verifican la elección):                    oct_2023, feb_2024
- evento y post_evento: se reportan, nunca se usan para seleccionar.
Regla de decisión: un cambio se adopta solo si mejora el WAPE medio de desarrollo >= 0.2 puntos.

Uso:
    uv run python -m src.models.train --horizon 14            # backtest + modelo final
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data.load import ROOT
from src.features.engineering import FeatureConfig, load_features
from src.models.backtest import FOLDS, Fold, evaluate

DEV_FOLDS = ["semana_santa_2023", "may_jun_2023", "sep_2023"]
CONFIRM_FOLDS = ["oct_2023", "feb_2024"]
MIN_GAIN = 0.2          # puntos de WAPE
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS = ROOT / "outputs" / "reports"
PARAMS_FILE = ROOT / "configs" / "model_params.json"

DEFAULT_PARAMS = dict(objective="regression", n_estimators=400, learning_rate=0.05, num_leaves=31,
                      min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=0.0, random_state=0, verbose=-1, n_jobs=-1)
KEEP = ["store_id", "category", "date", "store_format", "region", "y", "base_snaive", "base_sma4", "base_ma28"]


def load_params() -> dict:
    """Hiperparámetros elegidos (configs/model_params.json) o los de referencia."""
    p = dict(DEFAULT_PARAMS)
    if PARAMS_FILE.exists():
        p.update(json.loads(PARAMS_FILE.read_text()))
    return p


def fit(train: pd.DataFrame, cols: list[str], params: dict) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**params).fit(train[cols], np.log(train["y"]))


def predict(model: lgb.LGBMRegressor, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.exp(model.predict(df[cols]))


def run_backtest(F: pd.DataFrame, cfg: FeatureConfig, params: dict | None = None, cols: list[str] | None = None,
                 folds: list[Fold] | None = None) -> pd.DataFrame:
    """Predicciones diarias fuera de muestra de todos los folds (una fila por serie-día elegible)."""
    params = params or load_params(); cols = cols or cfg.feature_columns(); out = []
    for f in folds or FOLDS:
        tr = F[(F["date"] <= f.train_cutoff(cfg.horizon)) & F["eligible"]]
        te = F[f.test_mask(F["date"]) & F["eligible"]][KEEP].copy()
        te["pred"] = predict(fit(tr, cols, params), F.loc[te.index], cols)
        te["fold"], te["kind"] = f.name, f.kind
        out.append(te)
    return pd.concat(out, ignore_index=True)


def fold_metrics(preds: pd.DataFrame, models: dict[str, str] | None = None, level: str = "serie × semana") -> pd.DataFrame:
    """Métricas por fold y modelo. `models` = {nombre: columna de pronóstico}."""
    models = models or {"modelo": "pred", "sma4": "base_sma4", "snaive": "base_snaive", "ma28": "base_ma28"}
    rows = []
    for (fold, kind), g in preds.groupby(["fold", "kind"], sort=False):
        for name, col in models.items():
            rows.append({"fold": fold, "tipo": kind, "modelo": name, **evaluate(g, col, level)})
    return pd.DataFrame(rows)


def dev_score(F: pd.DataFrame, cfg: FeatureConfig, params: dict | None = None, cols: list[str] | None = None) -> dict:
    """WAPE semanal medio en folds de desarrollo y de confirmación (para ablaciones y tuning)."""
    folds = [f for f in FOLDS if f.name in DEV_FOLDS + CONFIRM_FOLDS]
    m = fold_metrics(run_backtest(F, cfg, params, cols, folds), {"m": "pred"})
    dev, conf = m[m.fold.isin(DEV_FOLDS)], m[m.fold.isin(CONFIRM_FOLDS)]
    return {"dev": dev.wape.mean() * 100, "confirmación": conf.wape.mean() * 100, "sesgo_dev": dev.bias.mean() * 100}


def train_final(F: pd.DataFrame, cfg: FeatureConfig, params: dict | None = None, cols: list[str] | None = None):
    """Modelo final: todas las filas elegibles hasta el último día con target conocido."""
    params = params or load_params(); cols = cols or cfg.feature_columns()
    tr = F[F["eligible"]]
    return fit(tr, cols, params), cols, tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--horizon", type=int, default=14)
    cfg = FeatureConfig(horizon=ap.parse_args().horizon); F = load_features(cfg); params = load_params()
    preds = run_backtest(F, cfg, params)
    REPORTS.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(REPORTS / f"backtest_predictions_h{cfg.horizon}.parquet", index=False)
    m = fold_metrics(preds); m.to_csv(REPORTS / f"backtest_metrics_h{cfg.horizon}.csv", index=False)
    print((m.pivot_table(index=["tipo", "fold"], columns="modelo", values="wape") * 100).round(1).to_string())
    model, cols, tr = train_final(F, cfg, params)
    joblib.dump({"model": model, "columns": cols, "horizon": cfg.horizon, "params": params,
                 "train_end": str(tr["date"].max().date())}, MODELS_DIR / f"lgbm_h{cfg.horizon}.joblib")
    print(f"Modelo final entrenado con {len(tr):,} filas hasta {tr['date'].max().date()} → outputs/models/lgbm_h{cfg.horizon}.joblib")
