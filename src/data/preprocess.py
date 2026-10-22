"""Construcción del panel maestro: malla completa, banderas de calidad y joins.

Estrategia de calidad (no se imputa el target ni se descartan filas):
- `amount_total` no tiene nulos: es el target.
- Filas ausentes (fallas de POS): se reincorporan a la malla con `is_missing_row=True`;
  el target queda NaN y se excluyen de entrenamiento y métricas.
- `cash_*`, `units_sold`, `avg_ticket`: se conservan como NaN con bandera. Son variables del
  mismo día que se pronostica (no están disponibles al predecir), así que imputarlas no
  aporta al modelo.
- `total_transactions` está inflado en días con promoción (no cuadra con efectivo+tarjeta),
  por eso se marca con `tx_inflated` y no se usa como target ni como medida de uplift.
- `replenishment_signal` se descarta: es un promedio móvil de `total_transactions` que incluye
  el día a pronosticar y el siguiente (fuga de información).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import PROCESSED_DIR, load_calendar, load_stores, load_transactions

KEY = ["store_id", "category", "date"]


def build_panel(
    transactions: pd.DataFrame, stores: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Panel completo (tienda x categoría x día) con banderas de calidad."""
    t = transactions.drop(columns=["replenishment_signal"], errors="ignore")

    grid = pd.MultiIndex.from_product(
        [sorted(t["store_id"].unique()), sorted(t["category"].unique()),
         pd.date_range(calendar["date"].min(), calendar["date"].max(), freq="D")],
        names=KEY,
    )
    p = t.set_index(KEY).reindex(grid).reset_index()
    p["is_missing_row"] = p["amount_total"].isna()
    p["has_promotion"] = p["has_promotion"].fillna(0).astype(int)

    p["cash_missing"] = p["cash_transactions"].isna() & ~p["is_missing_row"]
    p["units_missing"] = p["units_sold"].isna() & ~p["is_missing_row"]
    p["ticket_missing"] = p["avg_ticket"].isna() & ~p["is_missing_row"]

    paid = p["cash_transactions"] + p["card_transactions"]
    p["tx_inflated"] = np.where(paid.notna(), p["total_transactions"] != paid, np.nan)

    cal = calendar.copy()
    # `is_holiday` también marca el Buen Fin: se separa el festivo oficial.
    cal["is_official_holiday"] = cal["is_holiday"] & ~cal["is_buen_fin"]
    p = p.merge(cal.drop(columns=["is_holiday"]), on="date", how="left")
    p = p.merge(stores, on="store_id", how="left")
    return p.sort_values(KEY).reset_index(drop=True)


def load_panel(rebuild: bool = False) -> pd.DataFrame:
    """Devuelve el panel maestro; lo cachea en data/processed/panel.parquet."""
    path = PROCESSED_DIR / "panel.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    panel = build_panel(load_transactions(), load_stores(), load_calendar())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return panel


if __name__ == "__main__":
    panel = load_panel(rebuild=True)
    print(f"Panel: {panel.shape}, filas ausentes: {int(panel.is_missing_row.sum())}")
