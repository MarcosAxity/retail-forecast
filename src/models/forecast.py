"""Pronóstico a futuro (h = 14 días) con el modelo final: de los datos al pronóstico por serie-día.

El modelo es *directo* a 14 días: el feature de la fecha t usa solo datos con fecha <= t − 14.
Por eso, desde el último día con datos (T) se pueden pronosticar T+1 … T+14 sin recursión: todos
los rezagos que necesitan ya existen.

Pasos:
1. ``extend_panel``: agrega a la malla las 14 fechas futuras (target vacío, calendario por regla).
2. ``build_features``: el mismo código de entrenamiento (sin duplicar lógica de features).
3. Modelo LightGBM (log del monto) × factor de la capa de reinicio de régimen (1.0 fuera de eventos).
4. Unidades estimadas = monto pronosticado ÷ precio medio por unidad reciente de cada serie.

Límites declarados:
- El calendario futuro se genera por regla (fines de semana, quincenas, festivos fijos, temporada
  navideña). Buen Fin y Semana Santa no se proyectan: es válido mientras el horizonte no los toque
  (con el corte de 2024-02-29 el horizonte llega al 14-mar-2024).
- ``has_promotion`` futuro se fija en 0 (no hay promoción planeada conocida).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import ROOT
from src.features.engineering import FeatureConfig, build_features
from src.models.regime import final_factor

MODEL_PATH = ROOT / "artifacts" / "model" / "lgbm_h14.joblib"
HORIZON = 14
PRICE_WINDOW_DAYS = 56                         # ventana para el precio medio por unidad
# Festivos fijos que aparecen en calendar.csv (mes, día); más el tercer lunes de marzo (Benito Juárez).
FIXED_HOLIDAYS = {(1, 1), (5, 1), (9, 16), (11, 2), (12, 12), (12, 24), (12, 25), (12, 31)}
STATIC_COLS = ["region", "store_format", "socioeconomic_level", "size_sqm", "num_checkouts",
               "opening_year", "has_pharmacy", "has_fuel_station"]


def _nth_weekday(ts: pd.Timestamp, month: int, weekday: int, n: int) -> bool:
    if ts.month != month or ts.weekday() != weekday:
        return False
    return (ts.day - 1) // 7 + 1 == n


def future_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Banderas de calendario por regla para fechas sin calendario publicado."""
    d = pd.DatetimeIndex(dates)
    hol = [(x.month, x.day) in FIXED_HOLIDAYS or _nth_weekday(x, 3, 0, 3) or _nth_weekday(x, 2, 0, 1) for x in d]
    month_end = d == (d + pd.offsets.MonthEnd(0))
    return pd.DataFrame({
        "date": d,
        "is_official_holiday": np.array(hol, dtype=bool),
        "is_buen_fin": False,
        "is_semana_santa": False,
        "is_payday": (d.day == 15) | month_end,
        "is_weekend": d.day_of_week >= 5,
        "is_navidad_season": ((d.month == 12) & (d.day >= 15)) | ((d.month == 1) & (d.day <= 6)),
    })


def extend_panel(panel: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Panel + ``horizon`` días futuros por serie, con target NaN (malla diaria completa)."""
    last = panel["date"].max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    stores = panel.drop_duplicates("store_id")[["store_id"] + [c for c in STATIC_COLS if c in panel.columns]]
    series = panel[["store_id", "category"]].drop_duplicates()
    grid = series.merge(pd.DataFrame({"date": dates}), how="cross").merge(stores, on="store_id", how="left")
    grid = grid.merge(future_calendar(dates), on="date", how="left")
    grid["amount_total"] = np.nan
    grid["has_promotion"] = 0
    grid["is_missing_row"] = False
    grid["category"] = pd.Categorical(grid["category"], categories=panel["category"].astype("category").cat.categories) \
        if str(panel["category"].dtype) == "category" else grid["category"]
    for c in STATIC_COLS:                                # conserva dtypes (p. ej. nivel socioeconómico ordenado)
        if c in panel.columns and str(panel[c].dtype) == "category":
            grid[c] = pd.Categorical(grid[c], categories=panel[c].cat.categories, ordered=panel[c].cat.ordered)
    cols = [c for c in panel.columns if c in grid.columns]
    return pd.concat([panel, grid[cols]], ignore_index=True).sort_values(["store_id", "category", "date"]).reset_index(drop=True)


def unit_prices(panel: pd.DataFrame, days: int = PRICE_WINDOW_DAYS) -> pd.Series:
    """Precio medio por unidad (MXN/unidad) por tienda × categoría en los últimos ``days`` días.

    Solo cuenta filas con ``units_sold`` observado. Si una serie no tiene datos, usa el promedio de su categoría.
    """
    r = panel[(panel["date"] > panel["date"].max() - pd.Timedelta(days=days)) & panel["units_sold"].notna()
              & panel["amount_total"].notna()]
    g = r.groupby(["store_id", "category"], observed=True)[["amount_total", "units_sold"]].sum()
    price = g["amount_total"] / g["units_sold"]
    cat = r.groupby("category", observed=True)[["amount_total", "units_sold"]].sum()
    cat_price = cat["amount_total"] / cat["units_sold"]
    return price.rename("price_unit"), cat_price.rename("price_unit_cat")


def load_bundle(path=MODEL_PATH) -> dict:
    import joblib
    return joblib.load(path)


def forecast_next(panel: pd.DataFrame, bundle: dict | None = None, horizon: int = HORIZON) -> pd.DataFrame:
    """Pronóstico diario de las ``horizon`` fechas siguientes al último dato, por tienda × categoría."""
    bundle = bundle or load_bundle()
    cfg = FeatureConfig(horizon=horizon)
    last = panel["date"].max()
    ext = extend_panel(panel, horizon)
    F = build_features(ext, cfg)
    F = F.assign(reset_factor=final_factor(F))
    fut = F[F["date"] > last].copy()
    cols = bundle["columns"]
    fut["pred_model"] = np.exp(bundle["model"].predict(fut[cols]))
    fut["pred_amount"] = fut["pred_model"] * fut["reset_factor"]

    price, cat_price = unit_prices(panel)
    fut = fut.merge(price.reset_index(), on=["store_id", "category"], how="left")
    fut["price_unit"] = fut["price_unit"].fillna(fut["category"].map(cat_price))
    fut["pred_units"] = fut["pred_amount"] / fut["price_unit"]

    stores = panel.drop_duplicates("store_id")[["store_id", "socioeconomic_level", "size_sqm", "num_checkouts",
                                                "has_pharmacy", "has_fuel_station"]]
    stores = stores.rename(columns={"socioeconomic_level": "nse"}).assign(nse=lambda d: d["nse"].astype(str))
    out = fut[["store_id", "category", "date", "store_format", "region", "pred_model", "reset_factor",
               "pred_amount", "price_unit", "pred_units"]].merge(stores, on="store_id", how="left")
    out["category"] = out["category"].astype(str)
    out["store_format"] = out["store_format"].astype(str)
    out["region"] = out["region"].astype(str)
    return out.sort_values(["store_id", "category", "date"]).reset_index(drop=True)
