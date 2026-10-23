"""02 · EDA 2/3 · Nulos, consistencias entre variables, calendario y valores extremos

Generado a partir de notebooks/02_eda_complementario.ipynb (el notebook se conserva como referencia).
Ejecutar desde la raíz del repo:  uv run python scripts/02_eda_nulos_consistencias_calendario.py
Entradas : Ninguno (lee data/raw).
Salidas  : outputs/02_eda_nulos_consistencias_calendario/  (log.txt, tabla_XX.csv, fig_XX.png)
"""
# # 02 · EDA complementario: calidad, atributos de tienda, promoción y ruido
# Complementa `01_eda.ipynb` con las pruebas que sustentan las decisiones de modelado:
# 1. ¿Los nulos son aleatorios o estructurados?
# 2. ¿Qué campos son consistentes entre sí? (monto, transacciones, ticket)
# 3. ¿El calendario es confiable? ¿Los valores extremos son errores o eventos reales?
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

stage('02_eda_nulos_consistencias_calendario')
plt.rcParams.update({'figure.figsize': (10, 3.4), 'axes.grid': True, 'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False})
pd.set_option('display.width', 140, 'display.max_columns', 30, 'display.float_format', '{:,.3f}'.format)
rng = np.random.default_rng(0)
t, s, c = load_transactions(), load_stores(), load_calendar()
p = build_panel(t, s, c)
obs = p[~p.is_missing_row].copy()          # filas realmente observadas
obs['serie'] = obs.store_id + '|' + obs.category
print(obs.shape)

# ## 1. Nulos: ¿aleatorios (MCAR) o estructurados?

print('¿cash_transactions y amount_cash son nulos siempre juntos?')
print(pd.crosstab(t.cash_transactions.isna().rename('cash_tx nulo'), t.amount_cash.isna().rename('amount_cash nulo')))
flags = obs[['cash_missing', 'units_missing', 'ticket_missing']].astype(int)
print('\nFilas con nulo simultáneo (diagonal = nulos de la columna):'); show((flags.T @ flags))

base = obs.cash_missing.mean()
seg = {}
for col in ['category', 'store_format', 'region', 'socioeconomic_level', 'has_promotion', 'is_official_holiday', 'day_of_week', 'month']:
    r = obs.groupby(col, observed=True).cash_missing.mean() * 100
    seg[col] = f'{r.min():.1f}% – {r.max():.1f}%'
print(f'Tasa de nulos de efectivo por segmento (mín – máx entre niveles). Global: {base:.2%}')
print(pd.Series(seg, name='rango').to_string())

# dispersión entre tiendas vs lo esperable por azar
st = obs.groupby('store_id').cash_missing.agg(['mean', 'size'])
sd_exp = np.sqrt(base * (1 - base) / st['size'].mean()); z = (st['mean'] - base) / sd_exp
print(f'\nDesv. estándar entre tiendas: observada {st["mean"].std()*100:.2f} pp | esperada por azar {sd_exp*100:.2f} pp | |z| máx {z.abs().max():.2f}')

# persistencia: ¿un nulo hoy anticipa un nulo mañana?
q = obs.sort_values(['serie', 'date'])
for col in ['cash_missing', 'units_missing', 'ticket_missing']:
    prev = q.groupby('serie')[col].shift()
    print(f'{col:14s} P(nulo)={q[col].mean():.4f} | P(nulo | nulo ayer)={q.loc[prev == True, col].mean():.4f} | P(nulo | no nulo ayer)={q.loc[prev == False, col].mean():.4f}')

# **Lectura.** Los nulos se comportan como fallas puntuales independientes (MCAR), no como apagones:
# - `cash_transactions` y `amount_cash` son nulos **siempre juntos** (un solo evento: falla en el registro de efectivo). `units_sold` y `avg_ticket` aparentan ser eventos independientes (el cruce fuera de la diagonal es pequeño).
# - La tasa de nulos de efectivo es ≈ uniforme entre segmentos (5.6 %–6.4 %); la dispersión entre tiendas es la esperable por azar (0.55 pp observado vs. 0.47 pp esperado, |z| máx 2.3) y **un nulo ayer no eleva la probabilidad hoy** (5.7 % vs. 6.0 %).
#
# **Decisión.** Como no hay estructura que explotar y son variables del mismo día que se pronostica, se conservan como `NaN` con bandera (ya construidas en el panel) y no se usan como features.

# ## 2. Consistencias entre monto, transacciones y ticket

d = t.dropna(subset=['cash_transactions']).copy()
d['paid'] = d.cash_transactions + d.card_transactions
d['inflada'] = d.total_transactions != d.paid
print('amount_total vs efectivo + tarjeta (filas comparables):', int(t.amount_cash.notna().sum()),
      '| diferencia máx. abs:', f'{(t.amount_total - t.amount_cash - t.amount_card).abs().max():.1e}', 'MXN')
print('\nhas_promotion × (total_transactions ≠ efectivo + tarjeta):')
print(pd.crosstab(d.has_promotion, d.inflada.rename('total ≠ efectivo+tarjeta'), margins=True))
ex = (d.total_transactions / d.paid - 1)[d.has_promotion == 1]
print('\nExceso sobre efectivo + tarjeta en filas con promoción:'); print(ex.describe(percentiles=[.05, .5, .95]).round(3).to_frame().T.to_string())

tk = t.dropna(subset=['avg_ticket']).copy()
tk['paid'] = tk.cash_transactions + tk.card_transactions
tk['mm_total'] = (tk.amount_total / tk.total_transactions - tk.avg_ticket).abs() > 0.011
tk['mm_paid'] = (tk.amount_total / tk.paid - tk.avg_ticket).abs() > 0.011
print(f'avg_ticket coincide con monto / total_transactions en {(~tk.mm_total).mean():.1%} de las filas')
print(f'avg_ticket coincide con monto / (efectivo+tarjeta)  en {(~tk.mm_paid[tk.paid.notna()]).mean():.1%} de las filas con efectivo')
print('\nDonde NO coincide con total_transactions:'); print(pd.crosstab(tk.has_promotion, tk.mm_total.rename('ticket ≠ monto/total_tx')))
ok = tk.dropna(subset=['paid']); err = (ok.amount_total / ok.avg_ticket - ok.paid).abs()
print(f'\nReconstrucción de transacciones pagadas como monto/avg_ticket: error máx {err.max():.2f} transacciones')

cmp = d.assign(**{'efectivo+tarjeta': d.paid}).groupby('has_promotion')[['total_transactions', 'efectivo+tarjeta', 'amount_total', 'units_sold', 'avg_ticket']].mean().T
cmp.columns = ['sin promo', 'con promo']; cmp['diferencia %'] = (cmp['con promo'] / cmp['sin promo'] - 1) * 100
print('Medias simples (sin controles) por estatus de promoción:'); show(cmp)

# **Lectura.**
# - `amount_total` = efectivo + tarjeta de forma exacta (diferencia máxima 3.5e-10 MXN): el target es confiable.
# - `total_transactions` ≠ efectivo + tarjeta **si y solo si** `has_promotion = 1`: 38,295 de 38,295 filas con promoción y 0 de 153,540 sin ella. El exceso es de entre ≈ 8 % y 35 % (media 22 %).
# - `avg_ticket` coincide **siempre** con `amount_total / (efectivo + tarjeta)` y solo discrepa de `amount_total / total_transactions` en filas con promoción. Es decir: monto, ticket y transacciones pagadas son coherentes entre sí y **el campo que se aparta es `total_transactions`, solo con promoción**.
# - Con promoción, `total_transactions` sube ≈ 22 %, pero transacciones pagadas, monto, unidades y ticket no cambian (|dif.| ≤ 0.3 %). Si la promoción atrajera más clientes, esos clientes pagarían y el monto subiría.
# - Cuando falta el efectivo, las transacciones pagadas se reconstruyen como `monto / avg_ticket` con error < 0.3 transacciones.
#
# **Hipótesis (no verificable con estos datos):** en días con promoción se cuentan transacciones sin monto asociado (p. ej. canjes de cupón de valor cero o conteos duplicados). Se documenta como supuesto.
#
# **Decisión.** `total_transactions` no es válido como target ni como medida de uplift. El uplift se mide sobre `amount_total`.

# ## 3. Calendario y valores extremos

cc = c.copy()
reglas = {
    'day_of_week coincide con la fecha': (cc.date.dt.dayofweek == cc.day_of_week).all(),
    'is_weekend = sábado o domingo': ((cc.day_of_week >= 5) == cc.is_weekend).all(),
    'week_of_year = semana ISO': (cc.date.dt.isocalendar().week.astype(int) == cc.week_of_year).all(),
    'is_payday = día 15 o fin de mes': (((cc.date.dt.day == 15) | cc.date.dt.is_month_end) == cc.is_payday).all(),
    'is_navidad_season = 15-dic a 6-ene': ((((cc.date.dt.month == 12) & (cc.date.dt.day >= 15)) | ((cc.date.dt.month == 1) & (cc.date.dt.day <= 6))) == cc.is_navidad_season).all(),
}
print(pd.Series(reglas, name='cumple').to_string())
print('\nBuen Fin     :', cc.loc[cc.is_buen_fin, 'date'].dt.strftime('%Y-%m-%d').tolist())
print('Semana Santa :', cc.loc[cc.is_semana_santa, 'date'].dt.strftime('%Y-%m-%d').tolist())
print('Días marcados is_holiday que además son Buen Fin:', int((cc.is_holiday & cc.is_buen_fin).sum()), '→', cc.loc[cc.is_holiday & cc.is_buen_fin, 'holiday_name'].unique().tolist())
print('Festivos:', sorted(cc.holiday_name.dropna().unique()))

q = obs.sort_values(['serie', 'date']).copy()
q['med28'] = q.groupby('serie').amount_total.transform(lambda x: x.rolling(28, center=True, min_periods=7).median())
q['ratio'] = q.amount_total / q.med28
print(q.ratio.describe(percentiles=[.001, .01, .99, .999]).round(2).to_frame().T.to_string())
names = c.holiday_name.fillna('')
ev_dates = c.loc[c.is_buen_fin | names.str.contains('Nochebuena|Fin de Año|Navidad'), 'date']
alto, bajo = q[q.ratio > 3], q[q.ratio < 0.3]
print(f'\nFilas con ratio > 3: {len(alto):,} | % en Buen Fin/Nochebuena/Navidad/Fin de Año: {alto.date.isin(ev_dates).mean():.1%}')
print(f'Filas con ratio < 0.3: {len(bajo):,} | fechas más frecuentes:', bajo.date.dt.strftime('%Y-%m-%d').value_counts().head(4).to_dict())
print('Plausibilidad → unidades/transacción:', round((obs.units_sold / obs.total_transactions).min(), 2), 'a', round((obs.units_sold / obs.total_transactions).max(), 2),
      '| ticket:', round(obs.avg_ticket.min()), 'a', round(obs.avg_ticket.max()), '| duplicados:', int(t.duplicated(['date', 'store_id', 'category']).sum()),
      '| montos idénticos al día anterior:', int((t.sort_values(['store_id', 'category', 'date']).groupby(['store_id', 'category']).amount_total.diff() == 0).sum()))

# **Lectura.**
# - El calendario es internamente consistente (5 de 5 reglas verificables).
# - **Limitaciones:** `is_holiday` también marca los 4 días del Buen Fin (por eso el preprocesamiento crea `is_official_holiday`, que los excluye) y `is_semana_santa` solo marca Jueves y Viernes Santo de 2023; la Semana Santa de 2024 cae fuera del periodo.
# - **Valores extremos = eventos, no errores:** el 98.3 % de las filas con venta > 3× su mediana de 28 días cae en Buen Fin, Nochebuena, Navidad o Fin de Año; las caídas (< 0.3×) son sobre todo el 1-ene-2024 (449 de 532). No hay duplicados, montos congelados ni valores implausibles (unidades por transacción de 0.9 a 2.8).
#
# **Decisión.** No se recortan ni se eliminan valores extremos: se modelan con calendario y los eventos se evalúan aparte.
