"""Pruebas de las variables: foco en que NO haya fuga de información temporal."""
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import CALENDAR_FEATURES, KEY, FeatureConfig, build_features, calendar_features

FORBIDDEN = {"replenishment_signal", "total_transactions", "cash_transactions", "card_transactions",
             "amount_cash", "amount_card", "units_sold", "avg_ticket", "amount_total", "paid_transactions",
             "month", "quarter", "week_of_year", "store_id", "date", "y"}


@pytest.fixture(scope="module")
def feats14(panel):
    return build_features(panel, FeatureConfig(horizon=14))


def test_no_forbidden_columns_as_features():
    for h in (7, 14, 28):
        assert not set(FeatureConfig(horizon=h).feature_columns()) & FORBIDDEN


@pytest.mark.parametrize("horizon", [7, 14, 28])
def test_no_leakage_target_after_cutoff(panel, horizon):
    """Alterar el target posterior al corte C no cambia ningún feature (ni baseline) de las
    fechas <= C + base_lag, y sí cambia los de fechas posteriores (la prueba no es vacía)."""
    cfg = FeatureConfig(horizon=horizon); cut = pd.Timestamp("2023-09-30")
    base = build_features(panel, cfg)
    alt = panel.copy(); rng = np.random.default_rng(0)
    m = alt["date"] > cut
    alt.loc[m, "amount_total"] = alt.loc[m, "amount_total"] * rng.uniform(0.3, 3.0, m.sum())
    new = build_features(alt, cfg)
    cols = cfg.feature_columns() + [c for c in base.columns if c.startswith("base_")]
    keep = base["date"] <= cut + pd.Timedelta(days=cfg.base_lag)
    pd.testing.assert_frame_equal(base.loc[keep, KEY + cols].reset_index(drop=True),
                                  new.loc[keep, KEY + cols].reset_index(drop=True))
    after = (base["date"] > cut + pd.Timedelta(days=cfg.base_lag)) & (base["date"] <= cut + pd.Timedelta(days=cfg.base_lag + 7))
    assert not np.allclose(base.loc[after, f"lag_{cfg.base_lag}d"].fillna(0), new.loc[after, f"lag_{cfg.base_lag}d"].fillna(0))


def test_lags_are_by_date_not_by_position(panel, feats14):
    """STR_045 no tiene datos el 2023-02-01 y 02-02: el rezago de 14 días de 02-15 y 02-16 es NaN,
    no el valor de otra fecha."""
    r = feats14[(feats14.store_id == "STR_045") & (feats14.category == "Abarrotes")].set_index("date")
    assert r.loc[["2023-02-15", "2023-02-16"], "lag_14d"].isna().all()
    assert r.loc[["2023-02-14", "2023-02-17"], "lag_14d"].notna().all()
    ok = feats14[(feats14.store_id == "STR_001") & (feats14.category == "Abarrotes")].set_index("date")
    assert np.isclose(ok.loc["2023-03-20", "lag_14d"], np.log(ok.loc["2023-03-06", "y"]))
    assert np.isclose(ok.loc["2023-03-20", "base_sma4"], ok.loc[["2023-03-06", "2023-02-27", "2023-02-20", "2023-02-13"], "y"].mean())


def test_missing_rows_not_eligible(feats14):
    assert not feats14.loc[feats14.is_missing_row, "eligible"].any()
    assert feats14.loc[feats14.eligible, "y"].notna().all()


def test_calendar_features(panel):
    day = panel.drop_duplicates("date")[["date", "is_official_holiday", "is_buen_fin", "is_navidad_season",
                                         "is_payday", "is_weekend", "is_semana_santa"]]
    cal = calendar_features(day).set_index("date")
    # la quincena por regla coincide con la bandera del calendario
    assert ((cal.days_to_payday == 0).astype(int) == cal.is_payday).all()
    assert ((cal.days_since_payday == 0).astype(int) == cal.is_payday).all()
    # Buen Fin: 1..4 dentro del evento, 0 fuera
    assert cal.loc["2023-11-17":"2023-11-20", "buen_fin_day"].tolist() == [1, 2, 3, 4]
    assert (cal.loc[cal.is_buen_fin == 0, "buen_fin_day"] == 0).all()
    assert cal.loc["2023-11-16", "days_to_buen_fin"] == 1 and cal.loc["2023-11-21", "days_since_buen_fin"] == 1
    # sin NaN en los últimos días (festivo extra) ni en los primeros
    assert cal[CALENDAR_FEATURES[2:]].notna().all().all()
    # Navidad con signo: antes positivo, después negativo
    assert cal.loc["2023-12-10", "days_to_christmas"] == 15 and cal.loc["2023-12-28", "days_to_christmas"] == -3
