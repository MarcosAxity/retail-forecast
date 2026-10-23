"""Definición del target.

El modelo se entrena sobre `amount_total` (MXN). Las unidades se **derivan**:

    unidades_hat = monto_hat / precio_unitario(serie)

con `precio_unitario = Σ monto / Σ unidades` por tienda × categoría, calculado solo con datos
hasta el corte del pronóstico y solo con filas que tienen unidades observadas (las filas con
unidades nulas no deben entrar al numerador). `total_transactions` no se pronostica.
Evidencia: scripts/05_definicion_target.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "amount_total"
SERIES = ["store_id", "category"]


def unit_price_by_series(panel: pd.DataFrame, cutoff: str | pd.Timestamp | None = None,
                         last_days: int | None = None) -> pd.Series:
    """Precio unitario por serie con información hasta `cutoff` (inclusive).

    `last_days` limita la ventana a los últimos N días antes del corte (None = toda la historia).
    """
    d = panel[(~panel["is_missing_row"]) & panel["units_sold"].notna() & panel[TARGET].notna()]
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)
        d = d[d["date"] <= cutoff]
        if last_days is not None:
            d = d[d["date"] > cutoff - pd.Timedelta(days=last_days)]
    g = d.groupby(SERIES)
    return (g[TARGET].sum() / g["units_sold"].sum()).rename("unit_price")


def derive_units(df: pd.DataFrame, amount_hat: str, price: pd.Series) -> pd.Series:
    """Unidades pronosticadas = monto pronosticado / precio de la serie. `df` trae store_id y category."""
    p = df[SERIES].merge(price.reset_index(), on=SERIES, how="left")["unit_price"].values
    return pd.Series(df[amount_hat].values / p, index=df.index, name="units_hat")


def price_index_by_category(panel: pd.DataFrame) -> pd.DataFrame:
    """Índice mensual del precio unitario por categoría (1 = precio medio de la serie).
    Sirve para monitorear que el supuesto de precio estable se mantiene."""
    d = panel[panel["units_sold"].notna() & ~panel["is_missing_row"]].copy()
    d["price"] = d[TARGET] / d["units_sold"]
    d["p_idx"] = d["price"] / d.groupby(SERIES)["price"].transform("mean")
    d["mes"] = d["date"].dt.to_period("M").astype(str)
    return d.pivot_table(index="mes", columns="category", values="p_idx", aggfunc="mean")
