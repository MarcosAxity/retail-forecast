"""Backtest continuo: los 8 folds oficiales de `backtest.FOLDS` más 6 folds extra que cubren los
huecos (abr–may, jun–sep y las semanas pegadas a Buen Fin y Navidad). Ver notebook 06b, sección 1.
Cobertura contigua del 2023-03-27 al 2024-02-25 (48 semanas). Los folds oficiales no cambian."""
from __future__ import annotations

import pandas as pd

from src.models.backtest import FOLDS, Fold

EXTRA_FOLDS: list[Fold] = [
    Fold("abr_may_2023", "regular", "2023-04-24", "2023-05-28"),
    Fold("jun_jul_2023", "regular", "2023-06-26", "2023-07-23"),
    Fold("jul_ago_2023", "regular", "2023-07-24", "2023-08-20"),
    Fold("ago_sep_2023", "regular", "2023-08-21", "2023-09-03"),
    Fold("pre_buen_fin_2023", "evento", "2023-10-30", "2023-11-05"),
    Fold("pre_navidad_2023", "evento", "2023-12-04", "2023-12-10"),
]
ALL_FOLDS: list[Fold] = sorted(FOLDS + EXTRA_FOLDS, key=lambda f: f.start)
OFFICIAL = [f.name for f in FOLDS]

assert all(pd.Timestamp(b.start) == pd.Timestamp(a.end) + pd.Timedelta(days=1) for a, b in zip(ALL_FOLDS, ALL_FOLDS[1:]))
