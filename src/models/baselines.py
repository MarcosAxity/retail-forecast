"""Baselines de referencia, con la misma información que el modelo (horizonte directo).

Para la fecha t, con base_lag = múltiplo de 7 >= horizonte:
- `snaive`: ingenuo estacional, valor del mismo día de la semana hace `base_lag` días.
- `sma4`:   promedio de los últimos 4 mismos días de la semana (base_lag, +7, +14, +21).
- `ma28`:   promedio móvil de 28 días terminado en t − base_lag (pronóstico plano).

Si el valor de referencia falta, el pronóstico es NaN y la fila no se evalúa (no se imputa).
El listón de la Fase 3 es `sma4`: ya cumple ≈ 8 % de WAPE semanal en periodos regulares.

Uso: `uv run python -m src.models.baselines --horizon 14`
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.features.engineering import FeatureConfig, load_features
from src.models.backtest import FOLDS, evaluate, evaluate_by
from src.data.load import ROOT

BASELINES = {"snaive": "base_snaive", "sma4": "base_sma4", "ma28": "base_ma28"}
REPORTS = ROOT / "outputs" / "reports"


def run_baselines(feats: pd.DataFrame) -> pd.DataFrame:
    """Métricas de cada baseline en cada fold y nivel de agregación."""
    rows = []
    for fold in FOLDS:
        te = feats[fold.test_mask(feats["date"]) & feats["eligible"]]
        for name, col in BASELINES.items():
            for level in ("serie × día", "serie × semana", "tienda × semana", "total × semana"):
                rows.append({"fold": fold.name, "tipo": fold.kind, "modelo": name, "nivel": level, **evaluate(te, col, level)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--horizon", type=int, default=14)
    cfg = FeatureConfig(horizon=ap.parse_args().horizon)
    res = run_baselines(load_features(cfg))
    REPORTS.mkdir(parents=True, exist_ok=True); out = REPORTS / f"baselines_h{cfg.horizon}.csv"
    res.to_csv(out, index=False)
    piv = res[res.nivel == "serie × semana"].pivot_table(index=["tipo", "fold"], columns="modelo", values="wape")
    print((piv * 100).round(1).to_string()); print("→", out)
