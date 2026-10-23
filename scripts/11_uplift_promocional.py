"""11 · Fase 4 — Uplift promocional (Capa 2): ¿las promociones generan venta incremental?

Pregunta de negocio: en un día con promoción activa, ¿cuánto más se vende (amount_total) que sin ella,
y dónde (categoría, formato, región, evento) o cuándo (anticipación, canibalización) cambia ese efecto?

Diseño (derivado del EDA, scripts 01-03): la promoción se comporta como un experimento aleatorizado a
nivel serie-día (Bernoulli ≈ 0.20, sin memoria ni dependencia de segmentos o de la venta reciente).
Por eso el estimador es un contraste con efectos fijos, no un modelo predictivo:

    log(y_it) = β · promo_it + α_serie(i) + γ_fecha(t) + ε_it      (serie = tienda × categoría)

- Efectos fijos de serie y de fecha (por proyecciones alternadas) → β es la diferencia dentro de cada
  serie y día, sin sesgo por nivel, calendario ni tendencia común.
- Errores estándar agrupados por tienda (80 clusters) + inferencia por aleatorización
  (permutaciones dentro de cada serie) que no depende de supuestos asintóticos.
- Equivalencia (TOST) y efecto mínimo detectable: «no significativo» ≠ «cero»; se acota el efecto.
- Heterogeneidad con corrección por múltiples pruebas (Benjamini-Hochberg) y contra el azar esperado.
- Dinámica: adelantos y rezagos de la promoción (anticipación / canibalización / efecto acumulado).
- Dosis: primer día, segundo día y 3+ días consecutivos de promoción.
- Resultados alternos: unidades, transacciones pagadas, ticket y `total_transactions` (artefacto conocido).

Ejecutar desde la raíz del repo:  uv run python scripts/11_uplift_promocional.py
Entradas : data/processed/panel.parquet (script 00)
Salidas  : outputs/11_uplift_promocional/  (log.txt, tabla_XX.csv, fig_XX.png, resumen.json)
"""
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from _common import ROOT, stage, show, savefig, out_dir

from src.data.load import load_transactions, load_stores, load_calendar
from src.data.preprocess import build_panel

warnings.filterwarnings("ignore")
OUT = stage("11_uplift_promocional")
plt.rcParams.update({"figure.figsize": (9, 3.6), "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.width", 160, "display.max_columns", 30, "display.float_format", "{:,.3f}".format)
RNG = np.random.default_rng(0)
N_PERM = 1000
RESUMEN: dict = {}

# ---------------------------------------------------------------------------------------------
# 0. Datos
# ---------------------------------------------------------------------------------------------
p = build_panel(load_transactions(), load_stores(), load_calendar())
p = p.sort_values(["store_id", "category", "date"]).reset_index(drop=True)
p["serie"] = p.store_id + "|" + p.category
p["promo"] = p.has_promotion.astype(float).where(~p.is_missing_row)          # tratamiento desconocido si falta la fila
p["paid"] = p.cash_transactions + p.card_transactions
obs = p[~p.is_missing_row & p.amount_total.notna()].copy()
print(f"{len(obs):,} filas observadas | {obs.serie.nunique()} series | {obs.store_id.nunique()} tiendas | "
      f"{obs.date.min().date()} → {obs.date.max().date()} | promoción en {obs.promo.mean():.2%} de las filas")


# ---------------------------------------------------------------------------------------------
# Herramientas: residualización con efectos fijos y estimador con errores por cluster
# ---------------------------------------------------------------------------------------------
def residualize(x: np.ndarray, g1: np.ndarray, g2: np.ndarray, iters: int = 30, tol: float = 1e-10) -> np.ndarray:
    """Quita efectos fijos de dos factores (proyecciones alternadas; panel casi balanceado)."""
    r = x.astype(float).copy()
    n1, n2 = g1.max() + 1, g2.max() + 1
    c1, c2 = np.bincount(g1, minlength=n1), np.bincount(g2, minlength=n2)
    for _ in range(iters):
        before = r.copy()
        r -= (np.bincount(g1, r, n1) / c1)[g1]
        r -= (np.bincount(g2, r, n2) / c2)[g2]
        if np.abs(r - before).max() < tol:
            break
    return r


def fe_codes(df: pd.DataFrame):
    return pd.factorize(df.serie)[0], pd.factorize(df.date)[0], pd.factorize(df.store_id)[0]


def ols_cluster(Y: np.ndarray, X: np.ndarray, cl: np.ndarray):
    """OLS (Y, X ya residualizados) con varianza agrupada por cluster. Devuelve β, error estándar."""
    X = X.reshape(len(Y), -1)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    e = Y - X @ beta
    meat = np.zeros((X.shape[1],) * 2)
    for k in np.unique(cl):
        s = X[cl == k].T @ e[cl == k]
        meat += np.outer(s, s)
    G = len(np.unique(cl))
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1)
    return beta, np.sqrt(np.diag(V))


def effect(df: pd.DataFrame, outcome: str, treat="promo", log=True, controls: list[str] | None = None):
    """Efecto (en %) del tratamiento sobre `outcome`, con FE serie + fecha y error por tienda."""
    cols = [outcome, treat] + (controls or [])
    d = df.dropna(subset=cols)
    y = np.log(d[outcome].clip(lower=1e-9)).values if log else d[outcome].values
    g1, g2, cl = fe_codes(d)
    Xs = np.column_stack([residualize(d[c].values, g1, g2) for c in [treat] + (controls or [])])
    Y = residualize(y, g1, g2)
    b, se = ols_cluster(Y, Xs, cl)
    b0, s0 = b[0], se[0]
    return {"n": len(d), "beta_log": b0, "efecto %": np.expm1(b0) * 100, "IC95 inf %": np.expm1(b0 - 1.96 * s0) * 100,
            "IC95 sup %": np.expm1(b0 + 1.96 * s0) * 100, "ee (log)": s0, "p": 2 * stats.norm.sf(abs(b0 / s0))}


# ---------------------------------------------------------------------------------------------
# 1. Efecto promedio sobre la venta (resultado principal) y robustez a la especificación
# ---------------------------------------------------------------------------------------------
print("\n## 1. Efecto promedio de la promoción sobre amount_total")
naive = np.log(obs.amount_total)[obs.promo == 1].mean() - np.log(obs.amount_total)[obs.promo == 0].mean()
obs["dow_serie"] = obs.serie + "|" + obs.day_of_week.astype(str)
rows = {"sin controles (diferencia de medias, log)": {"efecto %": np.expm1(naive) * 100}}
rows["FE serie + fecha"] = effect(obs, "amount_total")
# robustez: FE serie×día de la semana en lugar de serie (controla estacionalidad semanal propia de cada serie)
o2 = obs.assign(serie=obs.dow_serie)
rows["FE (serie × día de la semana) + fecha"] = effect(o2, "amount_total")
# robustez: solo días normales (sin Buen Fin, temporada navideña ni festivos)
normal = obs[~(obs.is_buen_fin | obs.is_navidad_season | obs.is_official_holiday)]
rows["FE serie + fecha · solo días normales"] = effect(normal, "amount_total")
# robustez: nivel (MXN) en lugar de log; el efecto se expresa como % de la venta media
mxn, mu = effect(obs, "amount_total", log=False), obs.amount_total.mean()
rows["FE serie + fecha · en MXN (efecto como % de la venta media)"] = {
    "n": mxn["n"], "efecto %": mxn["beta_log"] / mu * 100, "p": mxn["p"],
    "IC95 inf %": (mxn["beta_log"] - 1.96 * mxn["ee (log)"]) / mu * 100, "IC95 sup %": (mxn["beta_log"] + 1.96 * mxn["ee (log)"]) / mu * 100}
t1 = pd.DataFrame(rows).T
show(t1[["n", "efecto %", "IC95 inf %", "IC95 sup %", "p"]], "Efecto de la promoción sobre amount_total (% vs. día sin promoción)")
main = t1.loc["FE serie + fecha"]
RESUMEN["efecto_amount_total_pct"] = {k: float(main[k]) for k in ["efecto %", "IC95 inf %", "IC95 sup %", "p", "n"]}

# ---------------------------------------------------------------------------------------------
# 2. Inferencia por aleatorización, equivalencia (TOST) y efecto mínimo detectable
# ---------------------------------------------------------------------------------------------
print("\n## 2. Aleatorización, equivalencia y potencia")
g1, g2, cl = fe_codes(obs)
y_res = residualize(np.log(obs.amount_total.values), g1, g2)
d0 = obs.promo.values
stat = lambda d: y_res[d == 1].mean() - y_res[d == 0].mean()
t_obs = stat(d0)
# permutaciones DENTRO de cada serie (se conserva el número de días con promoción de cada serie)
perm = np.empty(N_PERM)
for k in range(N_PERM):
    order = np.lexsort((RNG.random(len(obs)), g1))
    dp = np.empty_like(d0)
    dp[order] = d0[np.argsort(g1, kind="stable")]
    perm[k] = stat(dp)
p_perm = (np.sum(np.abs(perm) >= abs(t_obs)) + 1) / (N_PERM + 1)
fig, ax = plt.subplots()
ax.hist(perm * 100, bins=40, color="#9aa5b1"); ax.axvline(t_obs * 100, color="#c0392b", lw=2, label=f"observado ({t_obs*100:+.3f} %)")
ax.set(title=f"Prueba de aleatorización ({N_PERM} permutaciones dentro de serie): p = {p_perm:.2f}", xlabel="diferencia de medias residualizada (%)")
ax.legend(frameon=False); savefig("Distribución nula del efecto bajo reasignación aleatoria de la promoción")
se = main["ee (log)"]
mde = 2.8 * se                                              # 80 % de potencia, α = 0.05 (dos colas)
tost = {}
for m_ in (0.5, 1.0, 2.0):                                  # márgenes de equivalencia en %
    mm = np.log1p(m_ / 100)
    p_tost = max(stats.norm.sf((main["beta_log"] + mm) / se), stats.norm.cdf((main["beta_log"] - mm) / se))
    tost[f"±{m_} %"] = {"p (TOST)": p_tost, "¿equivalente a 0?": "sí" if p_tost < 0.05 else "no"}
t2 = pd.DataFrame(tost).T
show(t2, "Equivalencia con cero (TOST) para distintos márgenes")
print(f"p de aleatorización = {p_perm:.3f} | efecto mínimo detectable (80 % de potencia) = ±{np.expm1(mde)*100:.2f} % de la venta diaria de la serie")
RESUMEN.update(p_aleatorizacion=float(p_perm), mde_pct=float(np.expm1(mde) * 100),
               tost={k: v["p (TOST)"] for k, v in tost.items()})

# ---------------------------------------------------------------------------------------------
# 3. Otras métricas: ¿qué pasa con unidades, transacciones pagadas, ticket y total_transactions?
# ---------------------------------------------------------------------------------------------
print("\n## 3. Resultados alternos")
alt = {n: effect(obs, c) for n, c in [("amount_total (monto)", "amount_total"), ("units_sold (unidades)", "units_sold"),
                                      ("efectivo + tarjeta (transacciones pagadas)", "paid"), ("avg_ticket (ticket)", "avg_ticket"),
                                      ("total_transactions (artefacto)", "total_transactions")]}
t3 = pd.DataFrame(alt).T[["n", "efecto %", "IC95 inf %", "IC95 sup %", "p"]]
show(t3, "Efecto de la promoción por variable de resultado")
RESUMEN["efecto_total_transactions_pct"] = float(t3.loc["total_transactions (artefacto)", "efecto %"])
real = t3.drop("total_transactions (artefacto)")
print(f"Lectura: total_transactions sube {t3.loc['total_transactions (artefacto)', 'efecto %']:+.1f} %, pero ese conteo se infla solo con promoción (scripts 01 y 02) y no mide demanda real. "
      f"En las variables que sí reflejan demanda el efecto está entre {real['efecto %'].min():+.2f} % y {real['efecto %'].max():+.2f} % "
      f"(límites de IC95: {real['IC95 inf %'].min():+.2f} % a {real['IC95 sup %'].max():+.2f} %).")

# ---------------------------------------------------------------------------------------------
# 4. Heterogeneidad: ¿dónde funciona mejor la promoción?
# ---------------------------------------------------------------------------------------------
print("\n## 4. Heterogeneidad del efecto")
d_t = obs.promo.values; d_res = residualize(d_t, g1, g2)


def subgroup(mask: np.ndarray):
    """Efecto en un subconjunto con los residuos globales (FWL) y error agrupado por tienda."""
    X, Y, c = d_res[mask], y_res[mask], cl[mask]
    b = (X @ Y) / (X @ X); e = Y - b * X
    s = np.bincount(c, X * e)
    se_ = np.sqrt((s ** 2).sum() / (X @ X) ** 2 * (len(np.unique(c)) / (len(np.unique(c)) - 1)))
    return b, se_


obs["pago_quincena"] = np.where(obs.is_payday, "quincena", "otro día")
obs["fin_semana"] = np.where(obs.is_weekend, "fin de semana", "entre semana")
obs["evento"] = np.select([obs.is_buen_fin, obs.is_navidad_season, obs.is_semana_santa, obs.is_official_holiday],
                          ["Buen Fin", "temporada navideña", "Semana Santa", "festivo"], "día normal")
obs["nivel_venta"] = pd.qcut(obs.groupby("serie").amount_total.transform("median"), 3, labels=["bajo", "medio", "alto"])
obs["trimestre"] = obs.date.dt.to_period("Q").astype(str)
DIMS = ["category", "store_format", "region", "socioeconomic_level", "evento", "pago_quincena", "fin_semana", "nivel_venta", "trimestre"]
het = []
for dim in DIMS:
    for lvl, idx in obs.groupby(dim, observed=True).indices.items():
        m = np.zeros(len(obs), bool); m[idx] = True
        b, s_ = subgroup(m)
        het.append({"dimensión": dim, "nivel": str(lvl), "n": int(m.sum()), "efecto %": np.expm1(b) * 100,
                    "IC95 inf %": np.expm1(b - 1.96 * s_) * 100, "IC95 sup %": np.expm1(b + 1.96 * s_) * 100, "z": b / s_,
                    "p": 2 * stats.norm.sf(abs(b / s_))})
H = pd.DataFrame(het)
order_ = H.p.argsort().values
H["p ajustado (BH)"] = np.minimum(1, np.minimum.accumulate((H.p.values[order_] * len(H) / np.arange(1, len(H) + 1))[::-1])[::-1][np.argsort(order_)])
show(H.set_index(["dimensión", "nivel"]), "Efecto por subgrupo (error agrupado por tienda; p ajustado por Benjamini-Hochberg)")
n_sig_raw, n_sig_bh = int((H.p < .05).sum()), int((H["p ajustado (BH)"] < .05).sum())
print(f"Subgrupos evaluados: {len(H)} | p < 0.05 sin ajustar: {n_sig_raw} (esperados por azar ≈ {0.05 * len(H):.1f}) | significativos tras BH: {n_sig_bh}")
fig, ax = plt.subplots(figsize=(8, 0.28 * len(H) + 1.2))
yy = np.arange(len(H))[::-1]
ax.errorbar(H["efecto %"], yy, xerr=[H["efecto %"] - H["IC95 inf %"], H["IC95 sup %"] - H["efecto %"]], fmt="o", color="#2a6fbb", ms=3.5, lw=1)
ax.axvline(0, color="k", lw=.8); ax.set_yticks(yy); ax.set_yticklabels([f"{a}: {b}" for a, b in zip(H["dimensión"], H["nivel"])], fontsize=7.5)
ax.set(title="Efecto de la promoción por subgrupo (% de la venta, IC 95 %)", xlabel="%"); savefig("Forest plot de subgrupos")
RESUMEN["subgrupos"] = {"evaluados": len(H), "p<0.05_sin_ajustar": n_sig_raw, "significativos_BH": n_sig_bh}

# Heterogeneidad a nivel tienda vs. lo que produce el azar (permutando la promoción dentro de cada serie)
store_ids = pd.factorize(obs.store_id)[0]; ns = store_ids.max() + 1


def store_effects(dv):
    dr = residualize(dv, g1, g2)
    num, den = np.bincount(store_ids, dr * y_res, ns), np.bincount(store_ids, dr * dr, ns)
    return num / den, den


b_s, w_s = store_effects(d_t)
Q_obs = np.sum(w_s * (b_s - np.average(b_s, weights=w_s)) ** 2)
Q_null = np.empty(200)
for k in range(200):
    order = np.lexsort((RNG.random(len(obs)), g1)); dp = np.empty_like(d_t); dp[order] = d_t[np.argsort(g1, kind="stable")]
    bs_, ws_ = store_effects(dp); Q_null[k] = np.sum(ws_ * (bs_ - np.average(bs_, weights=ws_)) ** 2)
p_het = (np.sum(Q_null >= Q_obs) + 1) / (len(Q_null) + 1)
print(f"Heterogeneidad entre las {ns} tiendas: estadístico Q = {Q_obs:.2f} | mediana bajo el azar = {np.median(Q_null):.2f} | p (permutación) = {p_het:.2f}")
fig, ax = plt.subplots()
ax.hist(np.expm1(b_s) * 100, bins=25, color="#2a6fbb", alpha=.8)
ax.set(title=f"Efecto por tienda (80 tiendas): dispersión compatible con el azar (p = {p_het:.2f})", xlabel="efecto % estimado por tienda"); savefig("Efecto por tienda")
RESUMEN["heterogeneidad_tiendas_p"] = float(p_het)

# ---------------------------------------------------------------------------------------------
# 5. Dinámica: anticipación, canibalización y efecto acumulado (event study de ±7 días)
# ---------------------------------------------------------------------------------------------
print("\n## 5. Adelantos y rezagos de la promoción")
K = 7
g = p.groupby("serie").promo
for k in range(-K, K + 1):
    p[f"d{k:+d}"] = g.shift(-k)                    # promoción k días en el futuro (k>0) o en el pasado (k<0): promo_{t+k}
lead_cols = [f"d{k:+d}" for k in range(-K, K + 1)]
dd = p[~p.is_missing_row & p.amount_total.notna()].dropna(subset=lead_cols).copy()
a1, a2, acl = fe_codes(dd)
Xs = np.column_stack([residualize(dd[c].values, a1, a2) for c in lead_cols]); Ys = residualize(np.log(dd.amount_total.values), a1, a2)
bb, ss = ols_cluster(Ys, Xs, acl)
ES = pd.DataFrame({"k": range(-K, K + 1), "efecto %": np.expm1(bb) * 100, "IC95 inf %": np.expm1(bb - 1.96 * ss) * 100, "IC95 sup %": np.expm1(bb + 1.96 * ss) * 100}).set_index("k")
show(ES, "Efecto sobre la venta de t de una promoción en t+k (k<0: promoción pasada · k=0: mismo día · k>0: promoción futura)")
fig, ax = plt.subplots()
ax.errorbar(ES.index, ES["efecto %"], yerr=[ES["efecto %"] - ES["IC95 inf %"], ES["IC95 sup %"] - ES["efecto %"]], fmt="o-", color="#2a6fbb", ms=4, lw=1)
ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="gray", ls=":")
ax.set(title="Event study: efecto de la promoción en t+k sobre la venta del día t", xlabel="k (días)", ylabel="% de la venta"); savefig("Event study")
# efecto acumulado de una promoción: suma de los efectos en la ventana -7..+7 (delta method con la covarianza agrupada)
w = np.ones(len(bb)); cum = w @ bb
X_ = Xs; Xi = np.linalg.inv(X_.T @ X_); e_ = Ys - X_ @ bb
meat = sum(np.outer(X_[acl == c].T @ e_[acl == c], X_[acl == c].T @ e_[acl == c]) for c in np.unique(acl))
V_ = Xi @ meat @ Xi * len(np.unique(acl)) / (len(np.unique(acl)) - 1); se_cum = np.sqrt(w @ V_ @ w)
print(f"Efecto acumulado de una promoción (suma de t-7…t+7): {np.expm1(cum)*100:+.2f} % (IC95 [{np.expm1(cum-1.96*se_cum)*100:+.2f}, {np.expm1(cum+1.96*se_cum)*100:+.2f}] %) "
      f"| p = {2*stats.norm.sf(abs(cum/se_cum)):.2f}")
pre = [f"d{k:+d}" for k in range(1, K + 1)]; post = [f"d{k:+d}" for k in range(-K, 0)]
idx_pre = [lead_cols.index(c) for c in pre]; idx_post = [lead_cols.index(c) for c in post]
for nm, idx in [("anticipación (promoción en t+1…t+7)", idx_pre), ("canibalización/arrastre (promoción en t-7…t-1)", idx_post)]:
    ww = np.zeros(len(bb)); ww[idx] = 1; est = ww @ bb; s_ = np.sqrt(ww @ V_ @ ww)
    print(f"Suma de efectos · {nm}: {np.expm1(est)*100:+.2f} % (IC95 [{np.expm1(est-1.96*s_)*100:+.2f}, {np.expm1(est+1.96*s_)*100:+.2f}] %) | p = {2*stats.norm.sf(abs(est/s_)):.2f}")
RESUMEN["efecto_acumulado_pct"] = float(np.expm1(cum) * 100)

# ---------------------------------------------------------------------------------------------
# 6. Dosis: días consecutivos de promoción
# ---------------------------------------------------------------------------------------------
print("\n## 6. Promociones consecutivas (fatiga o refuerzo)")
p["promo_prev"] = p.groupby("serie").promo.shift(1); p["promo_prev2"] = p.groupby("serie").promo.shift(2)
p["run1"] = ((p.promo == 1) & (p.promo_prev == 0)).astype(float).where(p.promo_prev.notna() & p.promo.notna())
p["run2"] = ((p.promo == 1) & (p.promo_prev == 1) & (p.promo_prev2 == 0)).astype(float).where(p.promo_prev2.notna() & p.promo.notna())
p["run3p"] = ((p.promo == 1) & (p.promo_prev == 1) & (p.promo_prev2 == 1)).astype(float).where(p.promo_prev2.notna() & p.promo.notna())
dm = p[~p.is_missing_row & p.amount_total.notna()].dropna(subset=["run1", "run2", "run3p"]).copy()
b1, b2, bcl = fe_codes(dm)
Xr = np.column_stack([residualize(dm[c].values, b1, b2) for c in ["run1", "run2", "run3p"]]); Yr = residualize(np.log(dm.amount_total.values), b1, b2)
br, sr = ols_cluster(Yr, Xr, bcl)
t6 = pd.DataFrame({"filas": [int(dm[c].sum()) for c in ["run1", "run2", "run3p"]], "efecto %": np.expm1(br) * 100,
                   "IC95 inf %": np.expm1(br - 1.96 * sr) * 100, "IC95 sup %": np.expm1(br + 1.96 * sr) * 100,
                   "p": 2 * stats.norm.sf(np.abs(br / sr))}, index=["1.er día de promoción", "2.º día consecutivo", "3.er día o más"])
show(t6, "Efecto según días consecutivos con promoción")

# ---------------------------------------------------------------------------------------------
# 7. Traducción a negocio: ¿cuánto margen puede pagar una promoción?
# ---------------------------------------------------------------------------------------------
print("\n## 7. Traducción a negocio")
venta_dia = obs.amount_total.mean(); n_promo = int((obs.promo == 1).sum()); venta_promo = obs.loc[obs.promo == 1, "amount_total"].sum()
spec = t1.drop(index=[i for i in t1.index if i.startswith("sin controles")])
lo, pt, hi = spec["IC95 inf %"].min() / 100, main["efecto %"] / 100, spec["IC95 sup %"].max() / 100   # cotas más amplias entre especificaciones
inc = pd.DataFrame({"escenario": ["cota inferior (IC95 más amplio)", "estimación puntual (FE serie + fecha)", "cota superior (IC95 más amplio)"], "efecto %": [lo * 100, pt * 100, hi * 100]})
inc["venta incremental por serie-día con promoción (MXN)"] = [x * venta_dia for x in (lo, pt, hi)]
inc["venta incremental total de las promociones del periodo (MXN)"] = [x * venta_promo / (1 + x) for x in (lo, pt, hi)]
show(inc.set_index("escenario"), "Venta incremental atribuible a las promociones (periodo completo)")
rows = []
for margen in (0.15, 0.25, 0.35):
    rows.append({"margen bruto asumido": f"{margen:.0%}", "utilidad bruta máx. recuperable por promoción (MXN, con la cota superior del efecto)": max(hi, 0) * venta_dia * margen,
                 "como % de la venta del día": max(hi, 0) * margen * 100})
show(pd.DataFrame(rows).set_index("margen bruto asumido"),
     "Punto de equilibrio: si el costo de una promoción (descuento + operación) por serie-día supera esta cifra, no se paga sola")
print(f"Venta diaria media por serie: {venta_dia:,.0f} MXN | días-serie con promoción: {n_promo:,} ({n_promo / len(obs):.1%}) | "
      f"venta bajo promoción: {venta_promo / 1e6:,.1f} M MXN")
print("Supuestos: el dataset no trae costos ni descuentos de la promoción ni márgenes; los márgenes de la tabla son escenarios ilustrativos.")
RESUMEN["venta_diaria_media_mxn"] = float(venta_dia)

# ---------------------------------------------------------------------------------------------
# 8. Conclusiones
# ---------------------------------------------------------------------------------------------
print("\n## 8. Conclusiones")
lo_, hi_ = main["IC95 inf %"], main["IC95 sup %"]
if lo_ < 0 < hi_:
    veredicto = "no se distingue de cero"
elif hi_ < 0:
    veredicto = "es ligeramente negativo y su IC95 excluye el cero, pero la magnitud es económicamente irrelevante"
else:
    veredicto = "es positivo"
p_mxn = float(t1.loc[[i for i in t1.index if "MXN" in i][0], "p"])
print(f"1. Efecto promedio sobre la venta: {main['efecto %']:+.2f} % (IC95 [{lo_:+.2f}, {hi_:+.2f}] %); {veredicto}. "
      f"p de aleatorización = {p_perm:.3f}; en especificación en niveles (MXN) p = {p_mxn:.2f}. Cotas más amplias entre especificaciones: [{lo*100:+.2f}, {hi*100:+.2f}] %.")
eq = [k for k, v in tost.items() if v["p (TOST)"] < .05]
print(f"2. Precisión: efecto mínimo detectable (80 % de potencia) = ±{np.expm1(mde)*100:.2f} %. Equivalente a cero (TOST) con márgenes de {', '.join(eq) or 'ninguno de los márgenes probados'}. "
      "No hay evidencia de un uplift positivo.")
cand = H[H["p ajustado (BH)"] < 0.10].sort_values("p")
print(f"3. Heterogeneidad: {n_sig_raw} de {len(H)} subgrupos con p < 0.05 sin ajustar (≈ {0.05 * len(H):.1f} esperados por azar) y {n_sig_bh} tras Benjamini-Hochberg; "
      f"dispersión entre tiendas compatible con el azar (p = {p_het:.2f})."
      + (" Candidatos a vigilar (BH < 0.10, hipótesis y no conclusión): " + "; ".join(f"{r['dimensión']}={r['nivel']} ({r['efecto %']:+.2f} %)" for _, r in cand.iterrows()) + "." if len(cand) else ""))
print(f"4. Dinámica: efecto acumulado en ±7 días {np.expm1(cum)*100:+.2f} % (IC95 [{np.expm1(cum-1.96*se_cum)*100:+.2f}, {np.expm1(cum+1.96*se_cum)*100:+.2f}] %), "
      "sin patrón de anticipación ni canibalización consistente en el event study.")
print(f"5. total_transactions sube {t3.loc['total_transactions (artefacto)', 'efecto %']:+.1f} % con promoción pero es un artefacto de medición (script 02); la demanda real "
      "(monto, unidades, transacciones pagadas, ticket) no sube.")
print("6. Implicación: con estos datos las promociones no generan venta incremental medible. Antes de escalarlas conviene (a) conocer su costo real y margen, "
      "(b) probar con grupos de control planeados (holdout) por categoría o tienda.")
(OUT / "resumen.json").write_text(json.dumps(RESUMEN, indent=2, ensure_ascii=False))
