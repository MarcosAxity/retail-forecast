import numpy as np
import pandas as pd

from src.features.targets import derive_units, unit_price_by_series


def test_unit_price_ignores_rows_without_units(panel):
    price = unit_price_by_series(panel)
    d = panel[panel.units_sold.notna()]
    s = d[(d.store_id == "STR_001") & (d.category == "Abarrotes")]
    assert np.isclose(price[("STR_001", "Abarrotes")], s.amount_total.sum() / s.units_sold.sum())


def test_unit_price_uses_only_data_up_to_cutoff(panel):
    cut = "2023-06-30"; p1 = unit_price_by_series(panel, cut)
    alt = panel.copy(); m = alt.date > cut; alt.loc[m, "units_sold"] = alt.loc[m, "units_sold"] * 5
    pd.testing.assert_series_equal(p1, unit_price_by_series(alt, cut))


def test_derive_units():
    price = pd.Series([10.0, 20.0], index=pd.MultiIndex.from_tuples([("A", "x"), ("B", "y")], names=["store_id", "category"]), name="unit_price")
    df = pd.DataFrame({"store_id": ["A", "B"], "category": ["x", "y"], "amount_hat": [100.0, 100.0]})
    assert derive_units(df, "amount_hat", price).tolist() == [10.0, 5.0]
