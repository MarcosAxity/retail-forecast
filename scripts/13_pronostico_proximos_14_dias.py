"""13 · Pronóstico de los 14 días siguientes al último dato (insumo del Simulador del dashboard)

Aplica el modelo final (artifacts/model/lgbm_h14.joblib) a las 480 series tienda × categoría para
las fechas T+1 … T+14 (T = último día con datos, 2024-02-29), con la capa de régimen y una
estimación de unidades. No reentrena nada.

Ejecutar desde la raíz del repo:  uv run python scripts/13_pronostico_proximos_14_dias.py
Entradas : artifacts/model/lgbm_h14.joblib (paso 12) · data/raw/*.csv
Salidas  : outputs/13_pronostico_proximos_14_dias/ (log.txt, pronostico_14d.csv, tabla_XX.csv, fig_XX.png)
"""
from _common import stage, show, savefig  # noqa: E402
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, matplotlib.pyplot as plt

from src.data.preprocess import load_panel
from src.models import forecast as FC

d = stage("13_pronostico_proximos_14_dias")
pd.set_option("display.width", 170, "display.max_columns", 30, "display.float_format", "{:,.1f}".format)

# ## 1. Pronóstico
panel = load_panel()
bundle = FC.load_bundle()
last = panel["date"].max()
print(f"Último dato: {last.date()} | modelo entrenado hasta {bundle['train_end']} | horizonte {FC.HORIZON} días")
fut = FC.forecast_next(panel, bundle)
fut.to_csv(d / "pronostico_14d.csv", index=False)
print(f"{len(fut):,} filas ({fut[['store_id','category']].drop_duplicates().shape[0]} series × {fut['date'].nunique()} días) "
      f"→ outputs/13_pronostico_proximos_14_dias/pronostico_14d.csv")
print(f"Capa de régimen: factor = 1.0 en todas las filas ({(fut['reset_factor'] == 1).all()}); el horizonte no toca Buen Fin ni Navidad.")

# ## 2. Verificaciones de coherencia
obs = panel[panel["date"] > last - pd.Timedelta(days=14)].groupby("date")["amount_total"].sum()
pred = fut.groupby("date")["pred_amount"].sum()
ratio = pred.mean() / obs.mean()
print(f"\nVenta diaria media de la cadena: últimos 14 días observados {obs.mean():,.0f} MXN | pronóstico {pred.mean():,.0f} MXN "
      f"(razón {ratio:.3f})")
assert 0.85 < ratio < 1.15, "El pronóstico se aleja del nivel reciente: revisar"
assert fut[["pred_amount", "pred_units"]].notna().all().all() and (fut["pred_amount"] > 0).all()

# ## 3. Resúmenes por dimensión (monto en MXN y unidades estimadas)
tot = fut.groupby("date")[["pred_amount", "pred_units"]].sum().rename(columns={"pred_amount": "monto MXN", "pred_units": "unidades"})
tot.insert(0, "día", tot.index.day_name())
print("\nTotal de la cadena por día:"); show(tot)
for col, name in [("category", "categoría"), ("store_format", "formato de tienda"), ("nse", "nivel socioeconómico"), ("region", "región")]:
    g = fut.groupby(col).agg(**{"monto MXN": ("pred_amount", "sum"), "unidades": ("pred_units", "sum"),
                                "tiendas": ("store_id", "nunique")})
    g["% del monto"] = g["monto MXN"] / g["monto MXN"].sum() * 100
    print(f"\nPor {name}:"); show(g.sort_values("monto MXN", ascending=False))

# ## 4. Figura: historia reciente y pronóstico
hist = panel[panel["date"] > last - pd.Timedelta(days=56)].groupby("date")["amount_total"].sum() / 1e6
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(hist.index, hist.values, color="#2a78d6", label="observado")
ax.plot(pred.index, pred.values / 1e6, color="#eb6834", ls="--", marker="o", ms=3, label="pronóstico (14 días)")
ax.axvline(last, color="gray", lw=0.8); ax.set_ylabel("MXN millones"); ax.set_title("Venta diaria de la cadena: observado y pronóstico"); ax.legend()
savefig("Venta diaria de la cadena")
