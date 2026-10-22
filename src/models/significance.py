"""Significancia estadística de la diferencia de error entre dos pronósticos.

El error semanal de las series comparte choques comunes (todas las tiendas viven la misma semana),
así que las series **no son independientes**. Se agrega el error por semana y se remuestrea por
bloques de semanas (bootstrap por semanas) para el intervalo de confianza de ΔWAPE, además de una
prueba de rangos con signo de Wilcoxon sobre las diferencias semanales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.models.backtest import complete_weeks


def weekly_losses(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Por semana (agregando todas las series): error absoluto de a y de b y ventas reales.
    Solo series-semana completas y con ambos pronósticos."""
    d = complete_weeks(df[df[a].notna() & df[b].notna()], a)
    s = d.groupby(["store_id", "category", "semana"]).agg(y=("y", "sum"), a=(a, "sum"), b=(b, "sum")).reset_index()
    s["ea"], s["eb"] = (s["y"] - s["a"]).abs(), (s["y"] - s["b"]).abs()
    return s.groupby("semana")[["y", "ea", "eb"]].sum()


def compare(df: pd.DataFrame, a: str = "pred", b: str = "base_sma4", n_boot: int = 5000, seed: int = 0) -> dict:
    """ΔWAPE = WAPE(a) − WAPE(b) agrupado (negativo = a es mejor)."""
    w = weekly_losses(df, a, b); rng = np.random.default_rng(seed)
    delta = w.ea.sum() / w.y.sum() - w.eb.sum() / w.y.sum()
    idx = rng.integers(0, len(w), size=(n_boot, len(w)))
    y, ea, eb = w.y.values[idx].sum(1), w.ea.values[idx].sum(1), w.eb.values[idx].sum(1)
    boot = ea / y - eb / y
    diff = (w.ea - w.eb) / w.y
    wil = stats.wilcoxon(diff).pvalue if len(w) > 5 and (diff != 0).any() else np.nan
    return {"wape_a": w.ea.sum() / w.y.sum() * 100, "wape_b": w.eb.sum() / w.y.sum() * 100, "delta_pts": delta * 100,
            "ic95_inf": np.percentile(boot, 2.5) * 100, "ic95_sup": np.percentile(boot, 97.5) * 100,
            "p_boot": float(2 * min((boot >= 0).mean(), (boot <= 0).mean())), "semanas": len(w),
            "semanas_a_mejor": int((w.ea < w.eb).sum()), "p_wilcoxon": wil}
