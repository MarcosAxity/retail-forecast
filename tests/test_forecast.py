"""Pruebas del pronóstico a futuro: calendario por regla, paridad de features y coherencia del resultado."""
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import FeatureConfig, build_features
from src.models import forecast as FC


def test_calendario_por_regla_reproduce_el_observado(panel):
    cal = panel.drop_duplicates("date").set_index("date").sort_index()
    rule = FC.future_calendar(cal.index).set_index("date")
    for col in ("is_payday", "is_weekend", "is_navidad_season"):
        assert (rule[col].values == cal[col].astype(bool).values).all(), col
    # Festivos: Buen Fin y Semana Santa no se proyectan por diseño; se excluyen de la comparación.
    skip = cal["is_buen_fin"].astype(bool) | cal["is_semana_santa"].astype(bool) | \
        cal.index.isin(pd.to_datetime(["2023-04-06", "2023-04-07", "2023-11-17", "2023-11-18", "2023-11-19", "2023-11-20"]))
    assert (rule.loc[~skip.values, "is_official_holiday"].values == cal.loc[~skip.values, "is_official_holiday"].astype(bool).values).all()


def test_extend_panel_mantiene_malla_completa(panel):
    ext = FC.extend_panel(panel)
    assert ext["date"].max() == panel["date"].max() + pd.Timedelta(days=14)
    assert ext.groupby(["store_id", "category"]).size().nunique() == 1
    assert ext.loc[ext["date"] > panel["date"].max(), "amount_total"].isna().all()


def test_paridad_de_features_en_fechas_futuras(panel):
    """Pronosticar desde un corte anterior debe dar los mismos features que el panel completo
    (salvo ``has_promotion``, que a futuro se fija en 0). Prueba la ausencia de fugas y del calendario por regla."""
    cut = panel["date"].max() - pd.Timedelta(days=14)
    cfg = FeatureConfig(horizon=14)
    ext = FC.extend_panel(panel[panel["date"] <= cut])
    a = build_features(ext, cfg); b = build_features(panel, cfg)
    window = lambda d: d[(d["date"] > cut)].set_index(["store_id", "category", "date"]).sort_index()
    a, b = window(a), window(b)
    cols = [c for c in cfg.feature_columns() if c not in ("has_promotion", "category")]
    for c in cols:
        x, y = a[c], b[c]
        if x.dtype.name == "category":
            assert (x.astype(str) == y.astype(str)).all(), c
        else:
            assert np.allclose(x.astype(float), y.astype(float), equal_nan=True), c


@pytest.fixture(scope="module")
def fut(panel):
    return FC.forecast_next(panel)


def test_pronostico_forma_y_valores(panel, fut):
    n_series = panel[["store_id", "category"]].drop_duplicates().shape[0]
    assert len(fut) == n_series * 14
    assert fut["date"].min() == panel["date"].max() + pd.Timedelta(days=1)
    assert fut[["pred_amount", "pred_units", "price_unit"]].notna().all().all()
    assert (fut["pred_amount"] > 0).all() and (fut["pred_units"] > 0).all()
    assert (fut["reset_factor"] == 1.0).all()          # marzo: fuera de ventanas de evento


def test_pronostico_coherente_con_el_nivel_reciente(panel, fut):
    last14 = panel[panel["date"] > panel["date"].max() - pd.Timedelta(days=14)].groupby("date")["amount_total"].sum().mean()
    pred = fut.groupby("date")["pred_amount"].sum().mean()
    assert 0.85 < pred / last14 < 1.15
