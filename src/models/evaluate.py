"""Métricas de evaluación de pronósticos (todas ignoran pares con NaN)."""
from __future__ import annotations

import numpy as np


def _clean(y, yhat):
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(yhat))
    return y[ok], yhat[ok]


def wape(y, yhat) -> float:
    """Σ|y − ŷ| / Σ|y|. Métrica principal: pondera por volumen."""
    y, yhat = _clean(y, yhat)
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def mape(y, yhat) -> float:
    """Media de |y − ŷ| / y. Da el mismo peso a cada fila; favorece sub-pronosticar."""
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat) / np.abs(y)))


def bias(y, yhat) -> float:
    """Σ(ŷ − y) / Σy. Positivo = sobre-pronóstico (riesgo de exceso de inventario)."""
    y, yhat = _clean(y, yhat)
    return float((yhat - y).sum() / y.sum())


def mae(y, yhat) -> float:
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def summarize(y, yhat) -> dict[str, float]:
    return {"wape": wape(y, yhat), "mape": mape(y, yhat), "bias": bias(y, yhat), "mae": mae(y, yhat)}
