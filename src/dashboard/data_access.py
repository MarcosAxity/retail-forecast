"""Acceso a datos del dashboard: lee SOLO artefactos y salidas del pipeline (sin recalcular).

Funciones puras con ``lru_cache`` (testeables sin Streamlit). Si falta un archivo se lanza
``ArtifactMissing`` con el paso del pipeline que lo genera, para que la UI lo explique.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config as C


class ArtifactMissing(FileNotFoundError):
    """Falta un archivo que produce el pipeline (se indica qué paso lo regenera)."""


def _path(rel: str, step: str) -> Path:
    p = C.ROOT / rel
    if not p.exists():
        raise ArtifactMissing(f"No se encontró `{rel}`. Se genera con: uv run python run_pipeline.py --only {step}")
    return p


@lru_cache(maxsize=None)
def model_spec() -> dict:
    return json.loads(_path("artifacts/model/model_spec.json", "12").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def comparacion_regulares() -> pd.DataFrame:
    """Competencia de 8 modelos en periodos regulares (backtest continuo, paso 10)."""
    return pd.read_csv(_path("artifacts/metrics/comparacion_regulares.csv", "10"))


@lru_cache(maxsize=None)
def comparacion_eventos() -> pd.DataFrame:
    """WAPE medio en eventos (4 folds) sin y con la capa de régimen."""
    df = pd.read_csv(_path("artifacts/metrics/comparacion_eventos.csv", "10"), index_col=0)
    df.index.name = "modelo"
    return df.reset_index()


@lru_cache(maxsize=None)
def significancia() -> pd.DataFrame:
    df = pd.read_csv(_path("artifacts/metrics/significancia_vs_competidores.csv", "10"), index_col=0)
    df.index.name = "competidor"
    return df.reset_index()


@lru_cache(maxsize=None)
def backtest_oficial() -> pd.DataFrame:
    """WAPE/sesgo/MAE por fold y modelo del backtest oficial (paso 08), en %."""
    df = pd.read_csv(_path("artifacts/metrics/backtest_metrics_h14.csv", "08"))
    for col in ("wape", "mape", "bias"):
        df[col] = df[col] * 100
    return df


@lru_cache(maxsize=None)
def importancia_ganancia() -> pd.DataFrame:
    return pd.read_csv(_path("artifacts/model/feature_importance.csv", "12"))


@lru_cache(maxsize=None)
def importancia_shap() -> pd.DataFrame:
    """% de importancia SHAP (media del |valor|) de las variables principales (paso 07)."""
    df = pd.read_csv(_path("outputs/07_modelado_lightgbm/tabla_09.csv", "07"), index_col=0)
    df.index.name = "feature"
    return df.rename(columns={"% de la importancia total": "% SHAP"}).reset_index()


@lru_cache(maxsize=None)
def metricas_por_segmento() -> pd.DataFrame:
    return pd.read_csv(_path("outputs/07_modelado_lightgbm/tabla_06.csv", "07"))


@lru_cache(maxsize=None)
def piso_de_ruido() -> pd.DataFrame:
    """Error mínimo alcanzable por nivel de agregación (paso 03)."""
    df = pd.read_csv(_path("outputs/03_eda_tiendas_promocion_ruido/tabla_04.csv", "03"), index_col=0)
    df.index.name = "nivel"
    return df.reset_index()


@lru_cache(maxsize=None)
def uplift_resumen() -> dict:
    return json.loads(_path("artifacts/metrics/uplift_resumen.json", "11").read_text(encoding="utf-8"))


def headline_metrics() -> dict:
    """Cifras de portada de la Pestaña 1, todas derivadas de los artefactos."""
    reg = comparacion_regulares().set_index("modelo")
    ev = comparacion_eventos().set_index("modelo")
    piso = piso_de_ruido().set_index("nivel")["WAPE piso %"]
    lgbm = reg.loc["LightGBM (paso 07)"]
    return {
        "wape_reg": float(lgbm["WAPE %"]),
        "sesgo_reg": float(lgbm["sesgo %"]),
        "mae_reg": float(lgbm["MAE (MXN)"]),
        "vs_sma4_pts": float(lgbm["vs sma4 (pts)"]),
        "wape_sma4": float(reg.loc["sma4", "WAPE %"]),
        "piso_serie_semana": float(piso["serie × semana"]),
        "wape_evento_capa": float(ev.loc["LightGBM (paso 07)", "con capa"]),
        "wape_evento_sma4_capa": float(ev.loc["sma4", "con capa"]),
        "n_series": 480,
    }


# ───────────────────────── Simulador (Pestaña 2) ─────────────────────────
PRONOSTICO_CSV = "outputs/13_pronostico_proximos_14_dias/pronostico_14d.csv"
HISTORIA_DIAS = 56


@lru_cache(maxsize=None)
def pronostico() -> pd.DataFrame:
    """Pronóstico diario de los 14 días siguientes al último dato, por tienda × categoría.

    Lee la salida del paso 13; si no existe, lo calcula con el modelo (≈ 2 s) para que el tablero arranque igual.
    """
    path = C.ROOT / PRONOSTICO_CSV
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    from src.data.preprocess import load_panel
    from src.models import forecast as FC
    if not FC.MODEL_PATH.exists():
        raise ArtifactMissing("No se encontró `artifacts/model/lgbm_h14.joblib`. Se genera con: uv run python run_pipeline.py --only 08 12")
    return FC.forecast_next(load_panel())


@lru_cache(maxsize=None)
def historia() -> pd.DataFrame:
    """Venta observada diaria (MXN) por tienda × categoría de las últimas 8 semanas con datos."""
    from src.data.preprocess import load_panel
    p = load_panel()
    p = p[p["date"] > p["date"].max() - pd.Timedelta(days=HISTORIA_DIAS)]
    return p.loc[p["amount_total"].notna(), ["date", "store_id", "category", "amount_total"]].reset_index(drop=True)


@lru_cache(maxsize=None)
def tiendas() -> pd.DataFrame:
    df = pd.read_csv(_path("data/raw/stores.csv", "00"))
    return df.rename(columns={"socioeconomic_level": "nse"}).sort_values("store_id").reset_index(drop=True)


@lru_cache(maxsize=None)
def _backtest_predicciones() -> pd.DataFrame | None:
    path = C.ROOT / "outputs" / "reports" / "backtest_predictions_h14.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["store_id", "category", "date", "y", "pred", "kind"])
    return df[df["kind"] == "regular"]


def wape_historico(categoria: str | None, tienda: str | None) -> float | None:
    """WAPE semanal del backtest (periodos regulares) para la selección, al nivel de agregación elegido.

    Una tienda × categoría → serie × semana; una categoría → categoría × semana; toda la cadena → total × semana.
    ``None`` si no hay predicciones de backtest (se regeneran con el paso 08).
    """
    bt = _backtest_predicciones()
    if bt is None:
        return None
    if categoria:
        bt = bt[bt["category"] == categoria]
    if tienda:
        bt = bt[bt["store_id"] == tienda]
    if bt.empty:
        return None
    wk = bt.assign(semana=bt["date"].dt.to_period("W-SUN")).groupby("semana")[["y", "pred"]].sum()
    return float((wk["pred"] - wk["y"]).abs().sum() / wk["y"].sum() * 100)


# ───────────────────────── Uplift (Pestaña 3) ─────────────────────────
def _uplift(n: str) -> Path:
    return _path(f"outputs/11_uplift_promocional/tabla_{n}.csv", "11")


@lru_cache(maxsize=None)
def uplift_resultados() -> pd.DataFrame:
    """Efecto de la promoción por variable de resultado (% vs. día sin promoción) con IC95."""
    df = pd.read_csv(_uplift("03"), index_col=0)
    df.index.name = "resultado"
    return df.reset_index()


@lru_cache(maxsize=None)
def uplift_subgrupos() -> pd.DataFrame:
    return pd.read_csv(_uplift("04"))


@lru_cache(maxsize=None)
def uplift_event_study() -> pd.DataFrame:
    return pd.read_csv(_uplift("05"))


@lru_cache(maxsize=None)
def uplift_consecutivos() -> pd.DataFrame:
    df = pd.read_csv(_uplift("06"), index_col=0)
    df.index.name = "dia"
    return df.reset_index()


@lru_cache(maxsize=None)
def uplift_negocio() -> pd.DataFrame:
    return pd.read_csv(_uplift("07"))


def importancia_promocion() -> dict:
    """% de ganancia que el modelo asigna a las variables de promoción (evidencia independiente del uplift)."""
    g = importancia_ganancia().set_index("feature")["% ganancia"]
    return {k: float(g.get(k, 0.0)) for k in ("has_promotion", "promo_share_28")}
