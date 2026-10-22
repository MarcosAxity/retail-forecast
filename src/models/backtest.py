"""Esquema de validación walk-forward y evaluación de pronósticos.

Pronóstico directo a horizonte h: el pronóstico de la fecha t usa información hasta t − base_lag.
Cada fold es una ventana de prueba de semanas completas (lunes a domingo). El modelo de un fold
se entrena solo con fechas <= inicio_del_fold − h, de modo que ninguna etiqueta usada en el
entrenamiento es posterior al origen del primer día de prueba (criterio conservador: no se
reentrena dentro del fold).

La meta se mide en **tienda × categoría × semana** (grano de la decisión de reposición). Una
semana solo cuenta si sus 7 días tienen dato real y pronóstico. Los periodos regulares y los
eventos (Buen Fin, Navidad) se reportan por separado: cada evento aparece una sola vez en los
datos, así que ningún modelo entrenado antes de él puede aprender su magnitud.

Tipos de fold (regla definida antes de ver resultados):
- `regular`:     ni la ventana de prueba ni los rezagos de 14–35 días tocan Buen Fin o Navidad.
- `evento`:      la ventana de prueba contiene Buen Fin o Navidad.
- `post_evento`: la ventana de prueba es normal pero sus rezagos (14–35 días) caen sobre un evento,
                 así que los baselines y los rezagos del modelo arrastran el pico.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.evaluate import summarize

SERIES = ["store_id", "category"]


@dataclass(frozen=True)
class Fold:
    name: str
    kind: str          # "regular" | "evento" | "post_evento"
    start: str         # lunes
    end: str           # domingo

    def train_cutoff(self, horizon: int) -> pd.Timestamp:
        return pd.Timestamp(self.start) - pd.Timedelta(days=horizon)

    def test_mask(self, dates: pd.Series) -> pd.Series:
        return dates.between(pd.Timestamp(self.start), pd.Timestamp(self.end))


FOLDS: list[Fold] = [
    Fold("semana_santa_2023", "regular", "2023-03-27", "2023-04-23"),
    Fold("may_jun_2023", "regular", "2023-05-29", "2023-06-25"),
    Fold("sep_2023", "regular", "2023-09-04", "2023-10-01"),
    Fold("oct_2023", "regular", "2023-10-02", "2023-10-29"),
    Fold("buen_fin_2023", "evento", "2023-11-06", "2023-12-03"),
    Fold("navidad_2023", "evento", "2023-12-11", "2024-01-07"),
    Fold("post_navidad_2024", "post_evento", "2024-01-08", "2024-02-11"),
    Fold("feb_2024", "regular", "2024-02-12", "2024-02-25"),
]

LEVELS = {
    "serie × día": (["store_id", "category", "date"], False),
    "serie × semana": (SERIES + ["semana"], True),
    "tienda × semana": (["store_id", "semana"], True),
    "categoría × semana": (["category", "semana"], True),
    "total × semana": (["semana"], True),
}


def add_week(df: pd.DataFrame) -> pd.DataFrame:
    """Semana ISO (lunes a domingo) identificada por su lunes."""
    out = df.copy()
    out["semana"] = out["date"].dt.to_period("W-SUN").dt.start_time
    return out


def complete_weeks(df: pd.DataFrame, pred: str) -> pd.DataFrame:
    """Conserva solo las series-semana con 7 días que tienen real y pronóstico."""
    d = add_week(df[df["y"].notna() & df[pred].notna()])
    n = d.groupby(SERIES + ["semana"])["date"].transform("size")
    return d[n == 7]


def aggregate(df: pd.DataFrame, pred: str, level: str) -> pd.DataFrame:
    """Suma real y pronóstico al nivel pedido (semanal solo con semanas completas)."""
    keys, weekly = LEVELS[level]
    d = complete_weeks(df, pred) if weekly else df[df["y"].notna() & df[pred].notna()]
    return d.groupby(keys, observed=True)[["y", pred]].sum().reset_index()


def evaluate(df: pd.DataFrame, pred: str, level: str = "serie × semana") -> dict[str, float]:
    a = aggregate(df, pred, level)
    return {**summarize(a["y"], a[pred]), "n": len(a)}


def evaluate_by(df: pd.DataFrame, pred: str, by: str, level: str = "serie × semana") -> pd.DataFrame:
    """Métricas por segmento (categoría, formato, región…), tomando el nivel `level` por serie."""
    keys, weekly = LEVELS[level]
    d = complete_weeks(df, pred) if weekly else df[df["y"].notna() & df[pred].notna()]
    gk = keys + [by] if by not in keys else keys
    a = d.groupby(gk, observed=True)[["y", pred]].sum().reset_index()
    rows = {k: summarize(g["y"], g[pred]) | {"n": len(g)} for k, g in a.groupby(by, observed=True)}
    return pd.DataFrame(rows).T
