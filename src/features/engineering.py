"""Ingeniería de variables para el pronóstico de ventas (tienda × categoría × día).

Regla anti-fuga (la más importante del módulo)
----------------------------------------------
Todo feature derivado del target para la fecha *t* usa solo datos con fecha <= t − `base_lag`,
donde `base_lag` es el múltiplo de 7 más pequeño que es >= horizonte (14 → 14, 28 → 28).
Con múltiplos de 7 los rezagos caen siempre en el mismo día de la semana.

- Rezagos y ventanas se calculan **por fecha** sobre el panel completo (malla diaria sin huecos),
  nunca por posición de fila: una fila ausente deja el rezago correspondiente en NaN.
- Ventanas móviles con `min_periods` explícito (no se rellenan faltantes).
- No se usa ninguna columna observada el mismo día (unidades, ticket, efectivo, transacciones)
  ni `replenishment_signal` (ver notebooks 01 y 03).
- `has_promotion` del día *t* se asume conocida al pronosticar (promoción planeada).
- No hay `month`, `quarter`, `week_of_year` ni `store_id`: con 14 meses cada valor de calendario
  anual se observa una sola vez y el modelo lo usaría como índice de tiempo.

Escala: los features de historia del target están en **logaritmos** (la escala del modelo);
las columnas `base_*` están en **niveles** (MXN) y son los baselines de referencia.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.load import PROCESSED_DIR
from src.data.preprocess import load_panel

KEY = ["store_id", "category", "date"]
SERIES = ["store_id", "category"]
# Festivos conocidos fuera de calendar.csv (que termina el 2024-02-29): evita NaN en los
# últimos días. Tercer lunes de marzo (Natalicio de Benito Juárez, LFT art. 74).
EXTRA_HOLIDAYS = ("2024-03-18",)
HOLIDAY_CAP = 14
BUEN_FIN_CAP = 60
CHRISTMAS_RANGE = (-14, 60)

# Ventanas de evento (hipótesis tomadas de 2023; editables). Se usan solo para identificar días
# de referencia "contaminados" por un evento, no para predecir su magnitud.
EVENT_BUEN_FIN_PAD = 3                 # días antes/después del Buen Fin
EVENT_XMAS_RANGE = (-13, 28)           # days_to_christmas: del 27-nov al 7-ene (-14 es el valor truncado)
CLEAN_N_REFS = 12                      # semanas de referencia para el promedio limpio
CLEAN_MIN_REFS = 2

CALENDAR_FEATURES = [
    "day_of_week", "day_of_month", "is_weekend", "is_official_holiday", "is_payday",
    "is_navidad_season", "is_buen_fin", "is_semana_santa", "days_to_payday", "days_since_payday",
    "days_to_holiday", "days_since_holiday", "days_to_christmas", "buen_fin_day",
    "days_to_buen_fin", "days_since_buen_fin",
]
STORE_NUMERIC = ("size_sqm", "num_checkouts", "opening_year")
STORE_BOOL = ("has_pharmacy", "has_fuel_station")
STORE_CATEGORICAL = ("store_format", "region")


@dataclass(frozen=True)
class FeatureConfig:
    horizon: int = 14
    n_seasonal_lags: int = 4              # rezagos base_lag, +7, +14, +21 …
    windows: tuple[int, ...] = (7, 28)    # ventanas móviles (días)
    min_periods_frac: float = 0.5
    # Base: formato y nivel socioeconómico (ver notebook 02). El resto se prueba por ablación.
    store_features: tuple[str, ...] = ("store_format", "socioeconomic_level")
    target: str = "amount_total"
    # Opciones que se evalúan por ablación (apagadas por defecto):
    use_seasonal_position: bool = False   # month y week_of_year (riesgo: índice de tiempo con 14 meses)
    event_aware: bool = False             # rezagos "limpios" que excluyen días de evento

    @property
    def base_lag(self) -> int:
        return int(math.ceil(self.horizon / 7) * 7)

    @property
    def seasonal_lags(self) -> list[int]:
        return [self.base_lag + 7 * k for k in range(self.n_seasonal_lags)]

    def _store_cols(self) -> list[str]:
        cols = ["category"] + list(self.store_features)
        if "size_sqm" in self.store_features and "num_checkouts" in self.store_features:
            cols.append("sqm_per_checkout")
        return cols

    def feature_columns(self) -> list[str]:
        w = list(self.windows)
        hist = [f"lag_{L}d" for L in self.seasonal_lags] + ["same_dow_mean"]
        hist += [f"roll_mean_{x}" for x in w] + [f"roll_std_{max(w)}", "trend_ratio"]
        extra = (["month", "week_of_year"] if self.use_seasonal_position else [])
        extra += (["clean_same_dow_mean", "ref_event_share"] if self.event_aware else [])
        return CALENDAR_FEATURES + hist + ["has_promotion", f"promo_share_{max(w)}"] + self._store_cols() + extra


# ───────────────────────── calendario ─────────────────────────
def _check_grid(df: pd.DataFrame) -> None:
    """El panel debe tener malla diaria completa y ordenada (los rezagos se calculan por fecha)."""
    n_days = df["date"].nunique()
    sizes = df.groupby(SERIES).size()
    if not (sizes == n_days).all():
        raise ValueError("El panel no tiene malla diaria completa por serie (usa build_panel).")
    span = (df["date"].max() - df["date"].min()).days + 1
    if span != n_days:
        raise ValueError("Hay fechas faltantes en la malla diaria.")


def _distance(dates: pd.DatetimeIndex, events: np.ndarray, *, forward: bool, cap: int,
              known_start: pd.Timestamp, known_end: pd.Timestamp) -> np.ndarray:
    """Días hasta el próximo (forward) o desde el último evento; NaN si no hay evento conocido
    y la ventana de búsqueda excede la cobertura del calendario."""
    d = dates.values.astype("datetime64[D]"); ev = np.sort(events.astype("datetime64[D]"))
    out = np.full(len(d), np.nan)
    for i, x in enumerate(d):
        if forward:
            j = np.searchsorted(ev, x, side="left")
            if j < len(ev):
                out[i] = min(int((ev[j] - x).astype(int)), cap)
            elif x + np.timedelta64(cap, "D") <= known_end.to_datetime64().astype("datetime64[D]"):
                out[i] = cap
        else:
            j = np.searchsorted(ev, x, side="right") - 1
            if j >= 0:
                out[i] = min(int((x - ev[j]).astype(int)), cap)
            elif x - np.timedelta64(cap, "D") >= known_start.to_datetime64().astype("datetime64[D]"):
                out[i] = cap
    return out


def calendar_features(day_flags: pd.DataFrame) -> pd.DataFrame:
    """Features de calendario por fecha. `day_flags` trae una fila por fecha con las banderas
    del calendario (is_official_holiday, is_buen_fin, …)."""
    f = day_flags.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(f["date"])
    out = pd.DataFrame({"date": dates})
    out["day_of_week"] = dates.day_of_week
    out["day_of_month"] = dates.day
    for c in ("is_weekend", "is_official_holiday", "is_payday", "is_navidad_season", "is_buen_fin", "is_semana_santa"):
        out[c] = f[c].astype(int).values

    # Quincena por regla (día 15 y último del mes), no por la bandera del calendario.
    month_end = dates + pd.offsets.MonthEnd(0)
    mid = dates.to_period("M").to_timestamp() + pd.Timedelta(days=14)
    prev_month_end = dates.to_period("M").to_timestamp() - pd.Timedelta(days=1)
    is_pay = (dates.day == 15) | (dates == month_end)
    nxt = np.where(dates.day <= 15, mid, month_end)
    prv = np.where(dates.day >= 15, mid, prev_month_end)
    nxt = pd.DatetimeIndex(np.where(is_pay, dates, nxt)); prv = pd.DatetimeIndex(np.where(is_pay, dates, prv))
    out["days_to_payday"] = (nxt - dates).days; out["days_since_payday"] = (dates - prv).days

    known_start, known_end = dates.min(), dates.max()
    hol = np.concatenate([dates[f["is_official_holiday"].astype(bool).values].values,
                          pd.to_datetime(list(EXTRA_HOLIDAYS)).values])
    out["days_to_holiday"] = _distance(dates, hol, forward=True, cap=HOLIDAY_CAP, known_start=known_start, known_end=known_end)
    out["days_since_holiday"] = _distance(dates, hol, forward=False, cap=HOLIDAY_CAP, known_start=known_start, known_end=known_end)

    # Navidad: días con signo hasta el 25-dic más cercano (rampa previa vs. caída posterior).
    yrs = dates.year.values
    xmas = np.stack([pd.to_datetime(pd.DataFrame({"year": yrs + k, "month": 12, "day": 25})).values for k in (-1, 0, 1)], axis=1)
    diff = (xmas - dates.values[:, None]) / np.timedelta64(1, "D")
    best = diff[np.arange(len(dates)), np.abs(diff).argmin(axis=1)]
    out["days_to_christmas"] = np.clip(best, *CHRISTMAS_RANGE)

    # Buen Fin: se toma el primer día del tramo observado y se repite ±364 días (mismo día de la
    # semana) para no tener NaN fuera del año observado. Solo describe *cuándo* ocurre.
    flag = f["is_buen_fin"].astype(bool).values
    starts, length = [], 0
    for i in range(len(dates)):
        if flag[i] and (i == 0 or not flag[i - 1]):
            starts.append(dates[i]); j = i
            while j < len(dates) and flag[j]: j += 1
            length = max(length, j - i)
    anchors = sorted({s + pd.Timedelta(days=364 * k) for s in starts for k in (-1, 0, 1)})
    day_in, d_to, d_since = np.zeros(len(dates)), np.full(len(dates), float(BUEN_FIN_CAP)), np.full(len(dates), float(BUEN_FIN_CAP))
    for i, d in enumerate(dates):
        for a in anchors:
            end = a + pd.Timedelta(days=length - 1)
            if a <= d <= end:
                day_in[i], d_to[i], d_since[i] = (d - a).days + 1, 0, 0; break
            if a > d: d_to[i] = min(d_to[i], (a - d).days)
            if end < d: d_since[i] = min(d_since[i], (d - end).days)
    out["buen_fin_day"] = day_in.astype(int)
    out["days_to_buen_fin"] = np.minimum(d_to, BUEN_FIN_CAP); out["days_since_buen_fin"] = np.minimum(d_since, BUEN_FIN_CAP)
    return out


# ───────────────────────── historia del target ─────────────────────────
def _history_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    g = df.groupby(SERIES, sort=False)
    y = df[cfg.target]
    h = pd.DataFrame(index=df.index)
    lags_level = {L: g[cfg.target].shift(L) for L in cfg.seasonal_lags}
    for L, s in lags_level.items():
        h[f"lag_{L}d"] = np.log(s)
    lv = pd.concat(lags_level.values(), axis=1)
    need = math.ceil(cfg.n_seasonal_lags / 2)
    mean_lv = lv.mean(axis=1).where(lv.notna().sum(axis=1) >= need)
    h["same_dow_mean"] = np.log(mean_lv)

    ly = np.log(y).rename("ly"); tmp = pd.concat([df[SERIES], y.rename("y"), ly], axis=1)
    gt = tmp.groupby(SERIES, sort=False)
    out_lvl = {}
    for w in cfg.windows:
        mp = max(2, math.ceil(cfg.min_periods_frac * w))
        m = gt["y"].transform(lambda s: s.shift(cfg.base_lag).rolling(w, min_periods=mp).mean())
        out_lvl[w] = m; h[f"roll_mean_{w}"] = np.log(m)
    wmax = max(cfg.windows); mp = max(2, math.ceil(cfg.min_periods_frac * wmax))
    h[f"roll_std_{wmax}"] = gt["ly"].transform(lambda s: s.shift(cfg.base_lag).rolling(wmax, min_periods=mp).std())
    h["trend_ratio"] = h[f"roll_mean_{min(cfg.windows)}"] - h[f"roll_mean_{wmax}"]

    # Baselines en niveles (MXN), con la misma información que el modelo.
    h["base_snaive"] = lags_level[cfg.base_lag]
    h["base_sma4"] = mean_lv
    h["base_ma28"] = out_lvl[wmax]
    return h


def event_day_flag(cal: pd.DataFrame) -> pd.Series:
    """True en los días dentro de una ventana de evento (Buen Fin ± pad; temporada navideña)."""
    c = cal.set_index("date")
    bf = (c.buen_fin_day > 0) | (c.days_to_buen_fin <= EVENT_BUEN_FIN_PAD) | (c.days_since_buen_fin <= EVENT_BUEN_FIN_PAD)
    xm = c.days_to_christmas.between(*EVENT_XMAS_RANGE)
    return (bf | xm).rename("is_event_day")


def _event_aware_features(df: pd.DataFrame, cfg: FeatureConfig, cal: pd.DataFrame) -> pd.DataFrame:
    """Promedio del mismo día de la semana de las últimas `CLEAN_N_REFS` semanas **excluyendo** los
    días de referencia que caen en un evento, y proporción de los rezagos estacionales contaminados."""
    ev = event_day_flag(cal); g = df.groupby(SERIES, sort=False)[cfg.target]
    lags = [cfg.base_lag + 7 * k for k in range(CLEAN_N_REFS)]
    vals = np.column_stack([g.shift(L).values for L in lags])
    flag = np.column_stack([ev.reindex(df["date"] - pd.Timedelta(days=L)).fillna(False).values for L in lags]).astype(bool)
    clean = np.where(flag, np.nan, vals); n_ok = (~np.isnan(clean)).sum(axis=1)
    mean = np.full(len(df), np.nan); okm = n_ok >= CLEAN_MIN_REFS
    mean[okm] = np.nanmean(clean[okm], axis=1)
    out = pd.DataFrame(index=df.index)
    out["clean_same_dow_mean"] = np.log(mean)
    k = cfg.n_seasonal_lags
    out["ref_event_share"] = flag[:, :k].mean(axis=1)
    return out


def build_features(panel: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Devuelve una fila por tienda × categoría × día con target, banderas, baselines y features."""
    cfg = cfg or FeatureConfig()
    df = panel.sort_values(KEY).reset_index(drop=True)
    _check_grid(df)

    cal = calendar_features(df.drop_duplicates("date")[["date", "is_official_holiday", "is_buen_fin",
                            "is_navidad_season", "is_payday", "is_weekend", "is_semana_santa"]])
    hist = _history_features(df, cfg)

    g = df.groupby(SERIES, sort=False)
    df["has_promotion"] = df["has_promotion"].astype(int)
    pm = max(cfg.windows); mp = max(2, math.ceil(cfg.min_periods_frac * pm))
    promo_share = g["has_promotion"].transform(lambda s: s.shift(cfg.base_lag).rolling(pm, min_periods=mp).mean())

    meta = df[KEY + ["is_missing_row", "region", "store_format", "socioeconomic_level"]].copy()
    meta["socioeconomic_level"] = meta["socioeconomic_level"].astype(str)
    out = pd.concat([meta, df[[cfg.target]].rename(columns={cfg.target: "y"}), hist.filter(like="base_")], axis=1)
    out = out.merge(cal, on="date", how="left")
    feats = pd.concat([hist.drop(columns=hist.filter(like="base_").columns), df[["has_promotion"]],
                       promo_share.rename(f"promo_share_{pm}")], axis=1)
    out = pd.concat([out, feats], axis=1)
    if cfg.use_seasonal_position:
        out["month"] = df["date"].dt.month.values; out["week_of_year"] = df["date"].dt.isocalendar().week.astype(int).values
    if cfg.event_aware:
        out = pd.concat([out, _event_aware_features(df, cfg, cal)], axis=1)

    # Variables de tienda / serie
    order = ["C", "C+", "B", "A/B"]
    out["socioeconomic_level_ord"] = out["socioeconomic_level"].map({k: i for i, k in enumerate(order)})
    extra = df[[c for c in STORE_NUMERIC + STORE_BOOL if c in df.columns]].copy()
    for c in STORE_BOOL: extra[c] = extra[c].astype(int)
    extra["sqm_per_checkout"] = extra["size_sqm"] / extra["num_checkouts"]
    out = pd.concat([out, extra], axis=1)
    out["category"] = df["category"].astype("category").values
    for c in STORE_CATEGORICAL: out[c] = out[c].astype("category")
    # `socioeconomic_level` como feature es ordinal (C < C+ < B < A/B)
    out["socioeconomic_level"] = out["socioeconomic_level_ord"].astype(float)
    out = out.drop(columns="socioeconomic_level_ord")

    out["eligible"] = (~out["is_missing_row"]) & out["y"].notna() & out["base_sma4"].notna() & out["base_ma28"].notna()
    cols = KEY + ["is_missing_row", "region", "store_format", "y", "eligible"] + [c for c in out.columns if c.startswith("base_")]
    cols += [c for c in cfg.feature_columns() if c not in cols]
    return out.loc[:, ~out.columns.duplicated()][cols]


def features_path(cfg: FeatureConfig):
    return PROCESSED_DIR / f"features_h{cfg.horizon}.parquet"


def load_features(cfg: FeatureConfig | None = None, rebuild: bool = False) -> pd.DataFrame:
    cfg = cfg or FeatureConfig(); path = features_path(cfg)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    feats = build_features(load_panel(), cfg)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True); feats.to_parquet(path, index=False)
    return feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera el dataset maestro de features")
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()
    cfg = FeatureConfig(horizon=args.horizon)
    f = load_features(cfg, rebuild=True)
    print(f"Features h={cfg.horizon} (base_lag={cfg.base_lag}): {f.shape} → {features_path(cfg)}")
    print(f"Columnas de modelo ({len(cfg.feature_columns())}):", ", ".join(cfg.feature_columns()))
    print(f"Filas elegibles: {int(f.eligible.sum()):,} de {len(f):,}")
