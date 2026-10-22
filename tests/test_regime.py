import numpy as np
import pandas as pd

from src.models.regime import final_factor, make_window, reset_factor


def _toy(spike=True):
    """Dos series con nivel 100 y, entre el 15-nov y el 31-dic de 2023, nivel 300 (temporada alta)."""
    d = pd.date_range("2023-06-01", "2024-03-31")
    rows = []
    for s in ("A", "B"):
        y = np.full(len(d), 100.0 if s == "A" else 50.0)
        if spike:
            y[(d >= "2023-11-15") & (d <= "2023-12-31")] *= 3
        rows.append(pd.DataFrame({"store_id": s, "category": "C", "date": d, "y": y, "is_buen_fin": (d >= "2023-11-17") & (d <= "2023-11-20")}))
    df = pd.concat(rows, ignore_index=True)
    g = df.groupby(["store_id", "category"], sort=False)["y"]
    df["base_sma4"] = pd.concat([g.shift(L) for L in (14, 21, 28, 35)], axis=1).mean(axis=1)
    return df


def test_factor_is_one_without_events():
    df = _toy(spike=False)
    w = pd.Series(False, index=pd.DatetimeIndex(sorted(df.date.unique())))
    assert (reset_factor(df, w, w) == 1.0).all()


def test_factor_resets_level_after_season_and_is_neutral_elsewhere():
    df = _toy(); f = final_factor(df); df["f"] = f
    d = df[df.store_id == "A"].set_index("date")
    assert d.loc["2023-07-01":"2023-10-15", "f"].eq(1.0).all()          # regular: sin cambio
    assert d.loc["2023-11-20":"2023-12-31", "f"].eq(1.0).all()          # dentro de la temporada: sin cambio
    post = d.loc["2024-01-10", "f"]                                      # rezagos 27, 20, 13 y 6-dic = 300 (contaminados); limpio = 100
    assert np.isclose(post, 100 / 300)
    assert d.loc["2024-03-01", "f"] == 1.0                               # rezagos ya fuera de las ventanas


def test_factor_has_no_lookahead():
    df = _toy(); t = pd.Timestamp("2024-01-10")
    f0 = pd.Series(final_factor(df), index=df.index)
    df2 = df.copy(); df2.loc[df2.date > t - pd.Timedelta(days=14), "y"] = 9999.0     # altera el futuro respecto al origen
    # base_sma4 se recalcula con el mismo historial permitido (rezagos >= 14 d), así que debe coincidir en t
    g = df2.groupby(["store_id", "category"], sort=False)["y"]
    df2["base_sma4"] = pd.concat([g.shift(L) for L in (14, 21, 28, 35)], axis=1).mean(axis=1)
    f1 = pd.Series(final_factor(df2), index=df2.index)
    m = df.date == t
    assert np.allclose(f0[m], f1[m])


def test_window_covers_season_without_gaps():
    df = _toy(); w = make_window(df.date, [pd.Timestamp("2023-11-17")])
    season = w.loc["2023-10-27":"2024-01-08"]
    assert season.all()                                                   # Buen Fin (−21) y Navidad (+14) enlazan
