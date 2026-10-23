"""03 · EDA 3/3 · Atributos de tienda, asignación de promociones y ruido irreducible

Generado a partir de notebooks/02_eda_complementario.ipynb (el notebook se conserva como referencia).
Ejecutar desde la raíz del repo:  uv run python scripts/03_eda_tiendas_promocion_ruido.py
Entradas : Ninguno (lee data/raw).
Salidas  : outputs/03_eda_tiendas_promocion_ruido/  (log.txt, tabla_XX.csv, fig_XX.png)
"""
# # 03 · EDA 3/3: atributos de tienda, asignación de promociones y ruido irreducible
# Continúa el script 02 (nulos, consistencias y calendario; secciones 1 a 3) con las pruebas que sustentan las decisiones de modelado:
# 4. ¿Qué atributos de tienda importan y para qué (nivel vs. dinámica)?
# 5. ¿La promoción se asignó de forma aleatoria? (pruebas formales)
# 6. ¿Cuánto error es irreducible y a qué granularidad tiene sentido exigir una meta?

from _common import ROOT, stage, show, display, savefig  # noqa: E402
import sys
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from scipy import stats
from src.data.load import load_transactions, load_stores, load_calendar
from src.data.preprocess import build_panel
from src.models.evaluate import wape, mape, bias

stage('03_eda_tiendas_promocion_ruido')
plt.rcParams.update({'figure.figsize': (10, 3.4), 'axes.grid': True, 'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False})
pd.set_option('display.width', 140, 'display.max_columns', 30, 'display.float_format', '{:,.3f}'.format)
rng = np.random.default_rng(0)
t, s, c = load_transactions(), load_stores(), load_calendar()
p = build_panel(t, s, c)
obs = p[~p.is_missing_row].copy()          # filas realmente observadas
obs['serie'] = obs.store_id + '|' + obs.category
print(obs.shape)

# ## 4. Atributos de tienda: ¿qué explica el nivel y qué la dinámica?

sd_ = obs.groupby(['store_id', 'date']).amount_total.sum().rename('amt').reset_index()
level = np.log(sd_.groupby('store_id').amt.median()).rename('level')
S = s.set_index('store_id').join(level)
S['has_pharmacy'] = S.has_pharmacy.astype(int); S['has_fuel_station'] = S.has_fuel_station.astype(int)
CAT = ['store_format', 'region', 'socioeconomic_level']; NUM = ['has_pharmacy', 'has_fuel_station', 'opening_year', 'size_sqm', 'num_checkouts']
def design(cols, df=S):
    parts = [np.ones((len(df), 1))]
    for k in cols:
        parts.append(pd.get_dummies(df[k].astype(str), drop_first=True).astype(float).values if k in CAT else df[[k]].astype(float).values)
    return np.hstack(parts)
def r2(cols, y=S.level.values, df=S):
    X = design(cols, df); b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    return 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
r2_fmt = r2(['store_format']); rows = {}
for a in CAT[1:] + NUM:
    obs_gain = r2(['store_format', a]) - r2_fmt
    null = []
    for _ in range(1000):
        S2 = S.copy(); S2[a] = rng.permutation(S2[a].values); null.append(r2(['store_format', a], df=S2) - r2_fmt)
    rows[a] = {'R² solo': r2([a]), 'ΔR² sobre formato': obs_gain, 'p (permutación)': (np.sum(np.array(null) >= obs_gain) + 1) / 1001}
print(f'R² del formato solo: {r2_fmt:.3f} | con todos los atributos: {r2(CAT + NUM):.3f}')
show(pd.DataFrame(rows).T)

print('Correlación del nivel con atributos numéricos DENTRO de cada formato (media removida por formato):')
res = S.level - S.groupby('store_format')['level'].transform('mean')
print({k: round(np.corrcoef(S[k] - S.groupby('store_format')[k].transform('mean'), res)[0, 1], 3) for k in ['opening_year', 'size_sqm', 'num_checkouts']})
print('\nFormato × (farmacia, gasolinera) — nº de tiendas:'); print(pd.crosstab(S.store_format, [S.has_pharmacy.rename('farmacia'), S.has_fuel_station.rename('gasolinera')]))
X = design(['store_format', 'socioeconomic_level']); b = np.linalg.lstsq(X, S.level.values, rcond=None)[0]
names_ = ['const'] + ['formato=' + v for v in sorted(S.store_format.unique())[1:]] + ['socio=' + v for v in sorted(S.socioeconomic_level.astype(str).unique())[1:]]
print('\nEfecto multiplicativo sobre la venta diaria (base: Bodega y nivel A/B):'); print((np.expm1(pd.Series(b, index=names_).drop('const')) * 100).round(1).astype(str) + ' %')

# Dinámica: multiplicador de evento por formato (ventas del evento / ventas de los mismos días de la semana 4-7 semanas antes)
sd_ = sd_.merge(s[['store_id', 'store_format']], on='store_id')
piv = sd_.pivot(index='date', columns='store_id', values='amt')
def mult(days):
    days = pd.DatetimeIndex(days); refs = [days - pd.Timedelta(days=7 * k) for k in (4, 5, 6, 7)]
    ref = sum(piv.reindex(r).values for r in refs) / 4
    return pd.Series(piv.reindex(days).values.sum(0) / ref.sum(0), index=piv.columns)
ev = {'Buen Fin (4 días)': c.loc[c.is_buen_fin, 'date'], 'Navidad (18-31 dic)': pd.date_range('2023-12-18', '2023-12-31')}
M = pd.DataFrame({k: mult(v) for k, v in ev.items()}).join(s.set_index('store_id').store_format)
print('Multiplicador medio de la tienda (× nivel de referencia) por formato:'); show(M.groupby('store_format').mean().round(2))

# Tendencia común: los formatos se mueven en paralelo, pero el nivel común cambia mucho a lo largo del año
normal = sd_[~sd_.date.isin(c.loc[c.is_holiday | c.is_buen_fin | c.is_navidad_season, 'date'])].copy()
normal['lc'] = np.log(normal.amt); normal['lc'] -= normal.groupby('store_id').lc.transform('mean')
mm = normal.assign(mes=normal.date.dt.to_period('M').astype(str)).groupby(['mes', 'store_format']).lc.mean().unstack()
ax = mm.plot(marker='o', ms=3, title='Nivel mensual (días normales, log relativo a la media de la tienda) por formato'); ax.set_xlabel('')
plt.xticks(rotation=60); savefig()
print(f'Rango máx−mín entre formatos dentro de un mismo mes: {(mm.max(axis=1) - mm.min(axis=1)).max():.3f} log | variación del nivel común entre el mes más bajo y el más alto: {mm.mean(axis=1).max() - mm.mean(axis=1).min():.2f} log')
mix = obs.groupby(['store_id', 'category']).amount_total.sum().unstack(); mix = mix.div(mix.sum(axis=1), axis=0) * 100
print('\nParticipación por categoría entre tiendas (%):'); show(pd.DataFrame({'media': mix.mean(), 'desv. est. entre tiendas': mix.std(), 'mín': mix.min(), 'máx': mix.max()}).round(2))

# **Lectura.**
# - **Nivel:** el formato explica el 78 % de la varianza del nivel entre tiendas; con todos los atributos llega al 89 %. Frente a una Bodega, un Supercenter vende ≈ +61 % y un Express ≈ −52 %. El **nivel socioeconómico** aporta además del formato y de forma ordenada (ΔR² = 0.075, p = 0.001): frente a A/B, B ≈ −14 %, C+ ≈ −29 %, C ≈ −37 % (solo hay 5 tiendas A/B, así que esa referencia es imprecisa).
# - **Nada más aporta** una vez conocido el formato (p > 0.08): región, farmacia, gasolinera, año de apertura, m² y cajas. `size_sqm` y `num_checkouts` parecen explicar mucho por sí solos (R² ≈ 0.6) pero dentro de un mismo formato su correlación con el nivel es ≈ 0: son el formato disfrazado. Además ningún Express tiene farmacia ni gasolinera, así que con 80 tiendas no se pueden separar esos efectos del formato.
# - **Dinámica:** los eventos pegan distinto por formato (Buen Fin ≈ ×3.3–3.4 en Bodega y Supercenter vs. ×2.7 en Express, frente a los mismos días de la semana 4–7 semanas antes; esta base incluye la rampa de otoño, por eso las magnitudes son mayores que contra septiembre). Los tres formatos se mueven en paralelo mes a mes (diferencia máx. 0.013 log), pero el **nivel común cambia ≈ 0.64 log (≈ 90 %) entre el mes más bajo y el más alto**: con 14 meses no se puede separar tendencia de estacionalidad anual.
# - **Mix de categorías:** prácticamente idéntico en todas las tiendas (p. ej. Abarrotes 25.1 % ± 0.2 pp).
#
# **Decisión.** Modelo **global** (no uno por formato) con `store_format` y `socioeconomic_level` como variables y sus interacciones con calendario/eventos; sin `region`, farmacia, gasolinera ni tamaño en la versión base (se pueden probar con ablación en la Fase 3). El nivel propio de cada tienda que los atributos no explican lo capturan los rezagos de su propia serie. Nunca se usa `month`/`week_of_year` como variable: con 14 meses cada valor se observa una vez y el modelo lo usaría como índice de tiempo.

# ## 5. Promoción: ¿se asignó al azar? Pruebas formales

rows = []
for col in ['category', 'store_format', 'region', 'socioeconomic_level', 'has_pharmacy', 'has_fuel_station', 'day_of_week', 'is_payday', 'is_official_holiday', 'is_buen_fin', 'is_navidad_season', 'month']:
    ct = pd.crosstab(obs[col], obs.has_promotion); chi2, pv, *_ = stats.chi2_contingency(ct)
    rate = obs.groupby(col, observed=True).has_promotion.mean() * 100
    rows.append({'variable': col, 'tasa mín %': rate.min(), 'tasa máx %': rate.max(), 'p (chi²)': pv})
R = pd.DataFrame(rows).set_index('variable'); print(f'Tasa global de promoción: {obs.has_promotion.mean():.2%}'); print(R.to_string())
print(f'\nKS de los p-valores contra U(0,1): p = {stats.kstest(R["p (chi²)"], "uniform").pvalue:.2f}   (si fuera asignación dirigida esperaríamos p-valores pequeños)')

pr = obs.groupby('store_id').has_promotion.agg(['mean', 'size']); p0 = obs.has_promotion.mean()
print(f'Dispersión de la tasa entre tiendas: observada {pr["mean"].std()*100:.2f} pp | esperada por azar {np.sqrt(p0*(1-p0)/pr["size"].mean())*100:.2f} pp')
day = obs.groupby('date').has_promotion.agg(['mean', 'size']).query('size == 480')
print(f'Dispersión de la fracción diaria de series en promoción: observada {day["mean"].std()*100:.2f} pp | esperada {np.sqrt(p0*(1-p0)/480)*100:.2f} pp')
q = obs.sort_values(['serie', 'date']); prev = q.groupby('serie').has_promotion.shift()
print(f'P(promo | promo ayer) = {q.has_promotion[prev == 1].mean():.3f} | P(promo | sin promo ayer) = {q.has_promotion[prev == 0].mean():.3f}')
runs = q.groupby('serie').has_promotion.apply(lambda x: x.groupby((x != x.shift()).cumsum()).agg(['first', 'size']).query('first == 1')['size'])
print(f'Duración de una promoción: media {runs.mean():.3f} días | esperada si cada día es Bernoulli({p0:.2f}): {1/(1-p0):.3f}')
# ¿se promociona según el desempeño reciente? (asignación dirigida / causalidad inversa)
q['lc'] = np.log(q.amount_total) - q.groupby('serie').amount_total.transform(lambda x: np.log(x).mean())
for k in (1, 7, 14):
    lag = q.groupby('serie').lc.shift(k); dif = lag[q.has_promotion == 1].mean() - lag[q.has_promotion == 0].mean()
    se = np.sqrt(lag[q.has_promotion == 1].var() / (q.has_promotion == 1).sum() + lag[q.has_promotion == 0].var() / (q.has_promotion == 0).sum())
    print(f'Venta (log, centrada) {k:2d} días antes: promo − no promo = {dif:+.4f} (±{1.96*se:.4f})')

# **Lectura.** Cada prueba es consistente con una asignación **aleatoria** (Bernoulli ≈ 0.20 por serie y día):
# - La tasa de promoción es ≈ 20 % en todos los segmentos; los p-valores de las 12 pruebas χ² no se distinguen de uniformes (KS p = 0.42). El único p < 0.05 (nivel socioeconómico, p = 0.019; tasas de 19.6 % a 20.3 %) es lo esperable al hacer 12 pruebas.
# - La dispersión de la tasa entre tiendas (0.83 vs. 0.79 pp) y de la fracción diaria de series en promoción (1.81 vs. 1.82 pp) coincide con lo esperado por azar.
# - No hay memoria: P(promo | promo ayer) = P(promo | sin promo ayer) = 0.200, y la duración media de una promoción (1.249 días) iguala la geométrica esperada (1.250).
# - **No se promociona según el desempeño reciente:** la venta de 1, 7 y 14 días antes no difiere entre filas con y sin promoción (|dif.| < 0.002 log, dentro de ±0.0045).
#
# **Implicaciones para la Capa 2.**
# 1. El riesgo de sesgo de selección (confusión con calendario o tienda) es bajo: la promoción funciona como un experimento aleatorizado a nivel serie-día. Aun así el modelo controla por efectos fijos y calendario y se documenta como supuesto.
# 2. Como la promoción casi no tiene señal predecible, no debe esperarse que mejore el pronóstico.
# 3. El resultado puede ser un efecto nulo o muy pequeño: se reporta con intervalo de confianza y no se asume positivo.

# ## 6. Ruido irreducible y granularidad de la meta
# Si el error diario fuera ruido sin estructura, ningún modelo podría bajar de cierto WAPE. Se estima con un modelo estructural en logaritmos **con efecto fijo por fecha** (supone conocido el choque común de la cadena cada día, así que es un piso optimista): efectos de serie, de fecha, categoría × día de la semana y promoción.

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
import scipy.sparse as sp
r = obs.copy(); r['y'] = np.log(r.amount_total); r['cat_dow'] = r.category + '|' + r.day_of_week.astype(str); r['fecha'] = r.date.astype(str)
X = sp.hstack([OneHotEncoder().fit_transform(r[['serie', 'fecha', 'cat_dow']]), sp.csr_matrix(r[['has_promotion']].values.astype(float))]).tocsr()
mod = Ridge(alpha=1e-3, solver='sparse_cg', max_iter=2000).fit(X, r.y); r['yhat'] = np.exp(mod.predict(X)); resid = r.y - mod.predict(X)
print(f'σ residual (log): {resid.std():.3f} | autocorrelación lag-1/7/28 del residuo por serie:',
      [round(resid.groupby(r.serie).apply(lambda x: x.autocorr(k)).mean(), 3) for k in (1, 7, 28)])
r['semana'] = r.date.dt.to_period('W'); full = r.groupby(['serie', 'semana']).date.transform('size') == 7
levels = {'serie × día': ['serie', 'date'], 'tienda × día': ['store_id', 'date'], 'categoría × día': ['category', 'date'], 'total × día': ['date']}
wk = {'serie × semana': ['serie', 'semana'], 'tienda × semana': ['store_id', 'semana'], 'categoría × semana': ['category', 'semana'], 'total × semana': ['semana']}
out = {}
for name, k in {**levels, **wk}.items():
    base_df = r[full] if 'semana' in k else r
    g = base_df.groupby(k)[['amount_total', 'yhat']].sum(); out[name] = {'WAPE piso %': wape(g.amount_total, g.yhat) * 100, 'MAPE piso %': mape(g.amount_total, g.yhat) * 100}
show(pd.DataFrame(out).T.round(1))

# **Lectura.**
# - Tras quitar nivel de serie, choque diario de la cadena, perfil semanal por categoría y promoción, el residuo tiene **σ ≈ 0.20 en logaritmos y no tiene autocorrelación** (0.003, 0.001 y 0.000 a 1, 7 y 28 días): es ruido sin patrón que un modelo pueda aprender.
# - Al agregar, el ruido se compensa. **Aun con un pronóstico perfecto, a tienda × categoría × día no se baja de WAPE ≈ 15.6 %**; a tienda × categoría × **semana** el piso es ≈ 6.5 %, a tienda × semana ≈ 3.4 % y a nivel categoría o total ≈ 2 %.
# - Son pisos **optimistas** (se ajustan en la muestra y suponen conocido el choque común de la cadena cada día): un pronóstico real queda por encima, y un error diario por debajo de ≈ 15 % debe leerse como alarma de fuga de información.
#
# **Implicaciones para las métricas.**
# - La meta **MAPE < 15 % a nivel serie-día** del plan está en el piso de ruido: no es exigible.
# - La métrica principal será **WAPE** (pondera por volumen), con **MAPE secundaria y sesgo siempre a la vista**: el MAPE favorece sub-pronosticar, lo cual en inventario implica riesgo de desabasto.
# - El modelo se entrena a nivel diario y la meta se mide en **tienda × categoría × semana** (grano de la decisión de reposición, con piso ≈ 6.5 %), contra baselines simples.

# ## Conclusiones
# | Tema | Hallazgo | Decisión |
# |---|---|---|
# | Nulos | MCAR; efectivo siempre junto con su monto | `NaN` + bandera; no son features |
# | `total_transactions` | ≠ efectivo+tarjeta ⇔ promoción (38,295/38,295); ticket coherente con transacciones pagadas | No usar como target ni para uplift; monto como medida |
# | Calendario | Consistente; `is_holiday` incluye Buen Fin; Semana Santa solo 2 días | Usar `is_official_holiday`; no modelar Semana Santa 2024 |
# | Extremos | 98 % en eventos reales | No recortar; evaluar eventos aparte |
# | Atributos de tienda | Formato (78 %) + nivel socioeconómico; el resto es proxy del formato | Modelo global con formato y nivel socioeconómico |
# | Promoción | Compatible con asignación aleatoria (χ², dispersión, memoria, desempeño previo) | Efectos fijos + calendario; reportar IC; esperar efecto pequeño |
# | Ruido | Piso WAPE ≈ 15.6 % serie-día, ≈ 6.5 % serie-semana | Entrenar diario, evaluar en serie × semana con WAPE |
