"""Competidores del modelo final (notebook 07). Misma información y mismos folds que LightGBM:
los modelos de ML usan las mismas 30 features, objetivo log(y) y se reentrenan por fold con fechas
<= inicio − horizonte. ETS por serie usa solo la historia de la propia serie hasta el origen.

Selección de hiperparámetros: una rejilla pequeña evaluada solo en los folds de desarrollo, igual
que en el 06; confirmación, eventos y folds extra solo se reportan."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.backtest import Fold
from src.models.evaluate import wape

KEY = ["store_id", "category", "date"]
GRIDS = {"ridge": [{"alpha": a} for a in (1.0, 10.0, 100.0)],
         "rf": [{"min_samples_leaf": m} for m in (10, 50)],
         "xgb": [{"max_depth": d} for d in (4, 6, 8)]}


def make_model(name: str, df: pd.DataFrame, cols: list[str], **hp):
    cat = [c for c in cols if str(df[c].dtype) == "category"]
    if name == "xgb":   # maneja NaN y categorías de forma nativa; hiperparámetros alineados con el LightGBM del 06
        return xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=hp.get("max_depth", 6), min_child_weight=100,
                                subsample=0.8, colsample_bytree=0.8, tree_method="hist", enable_categorical=True, n_jobs=4, random_state=0)
    if name == "ridge":  # lineal: el día de la semana entra como dummies, no como número
        cat = cat + ["day_of_week"]; num = [c for c in cols if c not in cat]
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                                 ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)])
        return make_pipeline(pre, Ridge(alpha=hp.get("alpha", 10.0)))
    if name == "rf":
        num = [c for c in cols if c not in cat]
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat), ("n", SimpleImputer(strategy="median"), num)])
        return make_pipeline(pre, RandomForestRegressor(n_estimators=100, min_samples_leaf=hp.get("min_samples_leaf", 30), max_features=0.5,
                                                         max_samples=0.3, n_jobs=-1, random_state=0))
    raise ValueError(name)


def run_backtest_model(F: pd.DataFrame, horizon: int, cols: list[str], folds: list[Fold], name: str, **hp) -> pd.DataFrame:
    out = []
    for f in folds:
        tr = F[(F["date"] <= f.train_cutoff(horizon)) & F["eligible"]]
        te = F[f.test_mask(F["date"]) & F["eligible"]][KEY].copy()
        m = make_model(name, F, cols, **hp).fit(tr[cols], np.log(tr["y"]))
        te[f"pred_{name}"] = np.exp(m.predict(F.loc[te.index, cols])); te["fold"] = f.name
        out.append(te)
    return pd.concat(out, ignore_index=True)


def tune_on_dev(F: pd.DataFrame, horizon: int, cols: list[str], name: str, dev_folds: list[Fold]) -> tuple[dict, pd.DataFrame]:
    """Elige el hiperparámetro con menor WAPE medio (diario agregado a serie × semana vía WAPE de filas) en desarrollo."""
    from src.models.backtest import evaluate
    rows = []
    for hp in GRIDS[name]:
        p = run_backtest_model(F, horizon, cols, dev_folds, name, **hp).merge(F[KEY + ["y"]], on=KEY)
        w = np.mean([evaluate(g, f"pred_{name}")["wape"] for _, g in p.groupby("fold")])
        rows.append({**hp, "WAPE desarrollo %": w * 100})
    t = pd.DataFrame(rows); best = t.sort_values("WAPE desarrollo %").iloc[0]
    return {k: (int(best[k]) if k in ("max_depth", "min_samples_leaf") else float(best[k])) for k in t.columns if k != "WAPE desarrollo %"}, t


def _ets_one(y: np.ndarray, n_steps: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    warnings.filterwarnings("ignore")
    ly = pd.Series(np.log(np.clip(y, 1.0, None))).interpolate(limit_direction="both").values
    try:
        m = ExponentialSmoothing(ly, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7, initialization_method="estimated").fit()
        return np.exp(m.forecast(n_steps))
    except Exception:   # respaldo: promedio de las últimas 4 semanas del mismo día
        last = np.exp(ly[-28:]).reshape(4, 7).mean(0)
        return np.tile(last, n_steps // 7 + 2)[:n_steps]


def run_ets(F: pd.DataFrame, horizon: int, folds: list[Fold], n_jobs: int = 4) -> pd.DataFrame:
    """ETS aditivo en log (tendencia amortiguada, estacionalidad semanal) por serie y por fold."""
    piv = F.pivot(index="date", columns=["store_id", "category"], values="y")
    out = []
    for f in folds:
        origin = f.train_cutoff(horizon); n = (pd.Timestamp(f.end) - origin).days
        hist = piv.loc[:origin]
        fc = Parallel(n_jobs=n_jobs)(delayed(_ets_one)(hist[c].values, n) for c in piv.columns)
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=n)
        d = pd.DataFrame(np.column_stack(fc), index=dates, columns=piv.columns).loc[f.start:f.end]
        long = d.stack(["store_id", "category"]).rename("pred_ets").reset_index(); long["fold"] = f.name
        out.append(long.rename(columns={"level_0": "date"}) if "date" not in long else long)
    return pd.concat(out, ignore_index=True)
