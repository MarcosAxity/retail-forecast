"""Capa de reinicio de régimen (propuesta final, notebook 06b).

Problema: al terminar una temporada alta (Buen Fin, Navidad) los rezagos de 14–35 días y las
ventanas de 28 días siguen conteniendo el pico, y el modelo lo proyecta hacia adelante
(histéresis). La venta real vuelve de golpe al nivel base.

Regla: para una fecha *destino* fuera de la temporada alta cuyos 4 rezagos base (14, 21, 28 y 35 días)
tocan una ventana de evento, se sustituye el baseline `sma4` por el promedio de los 4 rezagos del
mismo día de la semana más recientes que caen fuera de toda ventana (hasta 16 semanas atrás,
mínimo 2). El pronóstico del modelo se multiplica por `limpio / sma4`. En cualquier otro caso el
factor es exactamente 1, así que los periodos regulares no cambian.

Dos ventanas de Navidad (decisión del 06b, sección 6):
- **referencia** (qué días contaminan los rezagos): 17 días antes a 14 después del 25-dic;
- **destino** (qué fechas se corrigen): hasta 6 días después (31-dic), porque la venta real se
  reinicia el 1-ene.

Anti-fuga: el factor usa solo `y` con fecha <= t − 14 (rezagos desde `base_lag`) y banderas de
calendario conocidas de antemano.

Límites: las ventanas son hipótesis tomadas de una sola ocurrencia de cada evento (2023). No deben
dejar huecos entre la de Buen Fin y la de Navidad (ver 06b, sección 6.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = ["store_id", "category"]
BF_WINDOW = (-21, 20)            # días respecto al primer día del Buen Fin
XMAS_REF_WINDOW = (-17, 14)      # respecto al 25-dic: ventana que contamina los rezagos
XMAS_TARGET_WINDOW = (-17, 6)    # respecto al 25-dic: fechas que se corrigen (fin de régimen 31-dic)
N_REFS, K_CLEAN, MIN_OK = 16, 4, 2


def buen_fin_anchors(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Primer día de cada tramo `is_buen_fin`, repetido ±364 días (mismo día de la semana),
    igual que las features de calendario."""
    d = df.drop_duplicates("date").sort_values("date")
    f = d["is_buen_fin"].astype(bool).values
    first = d["date"][f & ~np.r_[False, f[:-1]]]
    return sorted({pd.Timestamp(s) + pd.Timedelta(days=364 * k) for s in first for k in (-1, 0, 1)})


def make_window(dates, bf_anchors, bf=BF_WINDOW, xm=XMAS_REF_WINDOW) -> pd.Series:
    """Serie booleana por fecha: True dentro de una ventana de evento."""
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))
    w = pd.Series(False, index=days)
    anchors = [(pd.Timestamp(a), bf) for a in bf_anchors]
    anchors += [(pd.Timestamp(f"{y}-12-25"), xm) for y in range(days.min().year - 1, days.max().year + 1)]
    for a, (lo, hi) in anchors:
        w |= (days >= a + pd.Timedelta(days=lo)) & (days <= a + pd.Timedelta(days=hi))
    return w


def reset_factor(df: pd.DataFrame, win_ref: pd.Series, win_tgt: pd.Series, base_lag: int = 14,
                 n_refs: int = N_REFS, k: int = K_CLEAN, min_ok: int = MIN_OK) -> np.ndarray:
    """Factor `sma4 limpio / sma4` por fila (1.0 donde no aplica).

    `df`: malla diaria completa ordenada por serie y fecha, con `store_id`, `category`, `date`,
    `y` y `base_sma4` (promedio de los rezagos base_lag … base_lag+21 días).
    """
    lags = [base_lag + 7 * j for j in range(n_refs)]
    g = df.groupby(SERIES, sort=False)["y"]
    vals = np.column_stack([g.shift(L).values for L in lags])
    flag = np.column_stack([win_ref.reindex(pd.DatetimeIndex(df["date"] - pd.Timedelta(days=L)), fill_value=False).values
                            for L in lags]).astype(bool)
    ok = ~np.isnan(vals) & ~flag
    sel = ok & (np.cumsum(ok, axis=1) <= k)
    n = sel.sum(1)
    clean = np.where(n >= min_ok, np.where(sel, vals, 0.0).sum(1) / np.maximum(n, 1), np.nan)
    in_target = win_tgt.reindex(pd.DatetimeIndex(df["date"]), fill_value=False).values.astype(bool)
    contaminated = flag[:, :4].any(1) & ~in_target      # fuera de temporada alta, con rezagos base dentro
    base = df["base_sma4"].values
    return np.where(contaminated & ~np.isnan(clean) & ~np.isnan(base), clean / base, 1.0)


def final_factor(df: pd.DataFrame) -> np.ndarray:
    """Factor de la propuesta final con las ventanas por defecto."""
    anchors = buen_fin_anchors(df)
    return reset_factor(df, make_window(df["date"], anchors, xm=XMAS_REF_WINDOW),
                        make_window(df["date"], anchors, xm=XMAS_TARGET_WINDOW))
