import numpy as np
import pandas as pd

from src.models.backtest import FOLDS, add_week, aggregate, evaluate
from src.models.evaluate import bias, mape, wape


def test_folds_are_whole_weeks_and_do_not_overlap():
    prev_end = None
    for f in sorted(FOLDS, key=lambda f: f.start):
        s, e = pd.Timestamp(f.start), pd.Timestamp(f.end)
        assert s.dayofweek == 0 and e.dayofweek == 6 and ((e - s).days + 1) % 7 == 0
        assert f.kind in ("regular", "evento", "post_evento")
        if prev_end is not None:
            assert s > prev_end
        prev_end = e


def test_train_cutoff_precedes_test_origin():
    for f in FOLDS:
        for h in (7, 14, 28):
            assert f.train_cutoff(h) + pd.Timedelta(days=h) == pd.Timestamp(f.start)


def _toy(n_days=14):
    d = pd.date_range("2023-01-02", periods=n_days)   # lunes a domingo, dos semanas
    return pd.DataFrame({"store_id": "S", "category": "C", "date": d, "y": 100.0, "pred": 110.0})


def test_weekly_aggregation_keeps_only_complete_weeks():
    df = _toy().drop(index=3)                          # falta un día de la semana 1
    a = aggregate(df, "pred", "serie × semana")
    assert len(a) == 1 and a.iloc[0]["y"] == 700.0 and a.iloc[0]["pred"] == 770.0


def test_metrics():
    r = evaluate(_toy(), "pred", "serie × semana")
    assert np.isclose(r["wape"], 0.10) and np.isclose(r["bias"], 0.10) and np.isclose(r["mape"], 0.10)
    assert np.isclose(wape([100, 200], [110, 180]), 30 / 300) and np.isclose(bias([100, 200], [110, 180]), -10 / 300)
    assert np.isclose(mape([100, 200], [110, 180]), (0.1 + 0.1) / 2)
    assert add_week(_toy())["semana"].nunique() == 2


def test_regular_folds_do_not_touch_events():
    """Regla de los folds regulares: ventana de prueba y rezagos (14-35 días) fuera de Buen Fin y Navidad."""
    events = [("2023-11-17", "2023-11-20"), ("2023-12-15", "2024-01-06")]
    for f in FOLDS:
        if f.kind != "regular":
            continue
        lo, hi = pd.Timestamp(f.start) - pd.Timedelta(days=35), pd.Timestamp(f.end)
        for a, b in events:
            assert hi < pd.Timestamp(a) or lo > pd.Timestamp(b), f.name


def test_evaluate_by_segment_that_is_part_of_the_key():
    from src.models.backtest import evaluate_by
    df = _toy().assign(category="C")
    out = evaluate_by(df, "pred", "category")
    assert np.isclose(out.loc["C", "wape"], 0.10)
