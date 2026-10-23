"""01 · EDA 1/3 · Estructura, calidad, segmentos, dinámica temporal y promociones

Generado a partir de notebooks/01_eda.ipynb (el notebook se conserva como referencia).
Ejecutar desde la raíz del repo:  uv run python scripts/01_eda_calidad_y_dinamica.py
Entradas : Ninguno (lee data/raw).
Salidas  : outputs/01_eda_calidad_y_dinamica/  (log.txt, tabla_XX.csv, fig_XX.png)
"""
# # 01 · EDA y calidad de datos
# Objetivo: entender los datos, documentar problemas de calidad y fijar decisiones de preprocesamiento antes de modelar.

from _common import ROOT, stage, show, display, savefig  # noqa: E402
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt


from src.data.load import load_transactions, load_stores, load_calendar
from src.data.preprocess import build_panel

stage('01_eda_calidad_y_dinamica')
plt.rcParams.update({'figure.figsize': (10, 3.6), 'axes.grid': True, 'grid.alpha': .3})
t, s, c = load_transactions(), load_stores(), load_calendar()
p = build_panel(t, s, c)
print(t.shape, s.shape, c.shape, '| periodo', t.date.min().date(), '->', t.date.max().date())
print('tiendas', t.store_id.nunique(), '| categorías', t.category.nunique(), '| series', p.groupby(['store_id','category']).ngroups)

# ## 1. Estructura y completitud

esperadas = p.shape[0]
print(f'Filas esperadas (malla completa): {esperadas:,} | observadas: {len(t):,} | ausentes: {int(p.is_missing_row.sum())}')
aus = p[p.is_missing_row].groupby('store_id').agg(desde=('date','min'), hasta=('date','max'), filas=('date','size'))
show(aus)

# **Hallazgo.** Los «periodos sin datos» son pocos: 3 tiendas con 2–3 días completos ausentes (42 filas, 0.02 %). No se imputan: se reincorporan a la malla con `is_missing_row` y se excluyen de entrenamiento y métricas.

# ## 2. Nulos por columna y por tienda

nulos = t.isna().mean().mul(100).round(2); show(nulos[nulos > 0])

por_tienda = t.groupby('store_id')[['cash_transactions','units_sold','avg_ticket']].apply(lambda d: d.isna().mean()*100)
ax = por_tienda.plot(kind='box', title='% de nulos por tienda'); savefig()
# ¿hay semanas completas sin dato de efectivo en alguna tienda?
sem = t.assign(w=t.date.dt.to_period('W')).groupby(['store_id','w']).cash_transactions.apply(lambda x: x.isna().mean())
print('tienda-semanas con efectivo 100% nulo:', int((sem == 1).sum()), 'de', len(sem))

# **Hallazgo.** Los nulos de `cash_transactions`/`amount_cash` (5.9 %), `units_sold` (3.0 %) y `avg_ticket` (2.4 %) están dispersos de forma homogénea entre tiendas (máx. ~7 % por tienda): son fallas puntuales, no apagones. `amount_total` y `amount_card` no tienen nulos.
#
# **Decisión (con criterio de negocio).**
# | Columna | Tratamiento | Razón |
# |---|---|---|
# | `amount_total` | target, sin nulos | completa y = efectivo + tarjeta exacto |
# | `cash_*`, `avg_ticket`, `units_sold` | se mantienen NaN + bandera | son del mismo día que se pronostica: no existen al predecir, imputarlas no mejora el modelo |
# | filas ausentes | NaN + `is_missing_row` | imputar el target crearía datos que el modelo aprendería e inflaría métricas |
# | `replenishment_signal` | se descarta | ver §7 |

# ## 3. Distribución de ventas por segmento

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for a, col in zip(ax, ['store_format','region','socioeconomic_level']):
    p.groupby(col, observed=True).amount_total.mean().div(1e3).plot(kind='bar', ax=a, title=f'Venta diaria media por serie (miles MXN) · {col}')
plt.tight_layout(); savefig()
show(p.groupby('category').amount_total.agg(['mean','median','sum']).round(0).assign(share=lambda d: (d['sum']/d['sum'].sum()).round(3)))

print(s.groupby(['store_format']).agg(n=('store_id','size'), sqm=('size_sqm','median'), cajas=('num_checkouts','median')))
print(np.log(p.amount_total).groupby(p.store_id).mean().pipe(lambda x: f'CV del nivel entre tiendas (log): {x.std()/x.mean():.2%}'))

# **Hallazgo.** El nivel de venta difiere claramente por categoría (Abarrotes ≈ 25 % del monto) y por formato (el tamaño mediano va de ~2 mil m² en Express a ~12 mil en Supercenter). Hay mucha heterogeneidad de escala entre series, por lo que las métricas deben ser relativas (WAPE/sesgo), no RMSE ni R² globales, que solo reflejan la diferencia de tamaño entre tiendas.

# ## 4. Dinámica temporal: estacionalidad semanal y eventos

d = p.groupby('date').amount_total.sum() / 1e6
ax = d.plot(title='Venta diaria total de la cadena (millones MXN)')
for col, color in [('is_buen_fin','red'), ('is_navidad_season','green'), ('is_semana_santa','orange')]:
    f = c.loc[c[col], 'date']; ax.scatter(f.values, d.reindex(f).values, s=10, color=color, label=col)
ax.legend(); savefig()
dow = p.groupby('day_of_week').amount_total.mean(); show((dow/dow.mean()).round(3))

sem = d.resample('W').sum(); show((sem / sem['2023-09-04':'2023-09-30'].mean()).round(2).loc['2023-10-22':'2024-01-31'])

# **Hallazgo.** Hay fuerte efecto de día de la semana y dos eventos dominantes: **Buen Fin** y **Navidad**, con saltos de la cadena de más del doble en las semanas del evento, seguidos de una caída en enero. Cada evento aparece **una sola vez** en 14 meses, así que ningún modelo entrenado antes del evento puede aprender su magnitud: se evaluarán aparte de los periodos regulares y se reportarán como prueba de estrés.

# ## 5. Promociones: frecuencia, asignación y patrón

print('% filas con promoción:', round(p.has_promotion.mean()*100, 2))
rows = []
for col in ['day_of_week','is_payday','is_buen_fin','is_navidad_season','store_format','region','socioeconomic_level','category']:
    g = p.groupby(col, observed=True).has_promotion.mean().mul(100).round(1)
    rows.append(pd.DataFrame({'variable': col, 'nivel': g.index.astype(str), '% promo': g.values}))
show(pd.concat(rows).set_index(['variable','nivel']))

q = p.sort_values(['store_id','category','date'])
ac = q.groupby(['store_id','category']).has_promotion.apply(lambda x: x.autocorr(1)).mean()
runs = (q.groupby(['store_id','category']).has_promotion
          .apply(lambda x: x.groupby((x != x.shift()).cumsum()).agg(['first','size']).query('first == 1')['size']))
print(f'autocorrelación lag-1 de la promoción: {ac:.3f} | duración media de una promo: {runs.mean():.2f} días | % de promos de 1 día: {(runs==1).mean():.0%}')
m = p.groupby(p.date.dt.to_period('M')).has_promotion.mean().mul(100); m.plot(kind='bar', title='% filas con promoción por mes'); plt.ylim(0, 40); savefig()

# **Hallazgo clave para la Capa 2.** La promoción cubre ≈20 % de las filas en *todos* los segmentos, meses, días de la semana, quincenas y eventos, sin autocorrelación (≈0) y con duración típica de 1 día. Eso es lo que se vería con una asignación **aleatoria** (Bernoulli ≈ 0.2 por serie-día), no con una decisión comercial dirigida. Implicaciones:
# - El riesgo de sesgo de selección (planteado en el plan) es bajo; aun así el diseño controla por efectos fijos y calendario y se documenta como supuesto.
# - No hay periodos prolongados de promoción: el efecto es un efecto «de día»; se probarán rezagos/adelantos del tratamiento en la Fase 4 (anticipación y canibalización).

# ## 6. ¿Qué efecto aparente tiene la promoción? Transacciones vs. monto

q = p.dropna(subset=['amount_total']).copy()
for v in ['amount_total','total_transactions']:
    q['l'] = np.log(q[v].clip(lower=1)); q['lc'] = q.l - q.groupby(['store_id','category']).l.transform('mean')
    dif = q[q.has_promotion==1].lc.mean() - q[q.has_promotion==0].lc.mean()
    print(f'{v:20s} efecto aparente de la promo: {np.expm1(dif):+.2%}')
chk = q.dropna(subset=['tx_inflated']).groupby('has_promotion').tx_inflated.mean()
print('\n% filas donde total_transactions != efectivo + tarjeta:'); print(chk.mul(100).round(1))

# **Hallazgo.** `total_transactions` sube ~22 % en días con promoción **pero el monto cobrado no cambia** (~0 %). Además `total_transactions ≠ efectivo + tarjeta` en el 100 % de las filas con promoción donde se puede verificar y en 0 % de las demás: el conteo está inflado. Por eso **el target y el uplift se miden sobre `amount_total`** y no sobre transacciones.

# ## 7. `replenishment_signal`: ¿es válida como feature?

w = t.sort_values(['store_id','category','date']).copy()
g = w.groupby(['store_id','category']).total_transactions
w['ma'] = sum(g.shift(k) for k in range(-1, 6)) / 7          # ventana [t-5, t+1]
ok = w.dropna(subset=['ma','replenishment_signal'])
print(f"coincide con el promedio de total_transactions en [t-5, t+1]: {(np.abs(ok.ma - ok.replenishment_signal) <= 1).mean():.2%}")
print('fechas con nulo (primeras):', t[t.replenishment_signal.isna()].date.value_counts().sort_index().head(3).to_dict())

# **Hallazgo (fuga de información).** La señal es el promedio móvil de `total_transactions` de los 7 días `[t−5, t+1]`: contiene la demanda del día que se quiere predecir y la del siguiente. Usarla haría que las métricas fueran artificialmente buenas. **Se excluye** (sin importar el rezago) y sus nulos no se imputan. Sus nulos están al *inicio* del periodo y no al final como indica el diccionario.
#
# > Para mayor detalle ver `scripts/04_auditoria_replenishment.py` y la sección 7 de `docs/EDA.md`

# ## 8. Outliers

q = p.dropna(subset=['amount_total']).copy(); q['l'] = np.log(q.amount_total)
q['res'] = q.l - q.groupby(['store_id','category','day_of_week']).l.transform('median')
q['evento'] = q.is_buen_fin | q.is_navidad_season
print('|residuo| > 1 (en log) por tipo de día:'); print(q.assign(out=q.res.abs() > 1).groupby('evento').out.agg(['sum','mean']))
print('mínimo amount_total:', round(q.amount_total.min()), '| ceros:', int((q.amount_total==0).sum()))

# **Hallazgo.** No hay ceros ni negativos. Los valores extremos se concentran en eventos y son demanda real, no errores: no se recortan ni se eliminan. Se modelan con calendario y se evalúan por separado.

# ## 9. Conclusiones y decisiones
# 1. **Target:** `amount_total` (sin nulos, consistente con efectivo + tarjeta).
# 2. **No imputar** el target ni descartar filas; la malla se completa con banderas. Los nulos de variables del mismo día no pueden ser features.
# 3. **Excluir `replenishment_signal`** (fuga) y **no usar `total_transactions`** como medida de promoción (inflado).
# 4. **Métricas relativas** (WAPE, sesgo) y evaluación en **tienda × categoría × semana**; periodos regulares y eventos por separado.
# 5. **Features** del target con rezago mínimo = horizonte (14 días) para evitar fuga temporal.
# 6. **Promoción** con apariencia de aleatoria y efecto aparente ≈ 0 en monto: la Capa 2 debe reportar el efecto con intervalo de confianza y probar heterogeneidad, sin asumir que es positivo.

# ## Scripts complementarios
# - **`scripts/02_eda_nulos_consistencias_calendario.py` y `scripts/03_eda_tiendas_promocion_ruido.py`:** pruebas formales de nulos (MCAR), consistencias monto/transacciones/ticket, calendario, atributos de tienda, aleatoriedad de la promoción y piso de ruido por granularidad.
# - **`scripts/04_auditoria_replenishment.py`:** auditoría completa de la fuga de información de `replenishment_signal` (correlación cruzada, reconstrucción exacta e indirecta y prueba predictiva).
