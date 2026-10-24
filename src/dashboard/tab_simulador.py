"""Pestaña 2 · Simulador: pronóstico de los próximos 14 días por categoría y, opcionalmente, por tienda."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config as C
from . import data_access as D
from . import ui

TODAS_CAT, TODAS_TIENDAS = "Todas las categorías", "Todas las tiendas (cadena)"
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
DIAS_LARGO = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
NSE_ORDER = ["C", "C+", "B", "A/B"]
FORMAT_ORDER = ["Supercenter", "Bodega", "Express"]
DIMS = [("store_format", "Formato de tienda"), ("nse", "Nivel socioeconómico"), ("region", "Región"), ("category", "Categoría")]


def _compact(x: float, money: bool = False) -> str:
    p = "$" if money else ""
    if abs(x) >= 1e6:
        return f"{p}{x / 1e6:,.1f} M"
    if abs(x) >= 1e3:
        return f"{p}{x / 1e3:,.0f} k"
    return f"{p}{x:,.0f}"


def _fecha(ts: pd.Timestamp) -> str:
    return f"{ts.day} {MESES[ts.month - 1]}"


def _selector(stores: pd.DataFrame) -> tuple[str | None, str | None]:
    cats = sorted(D.pronostico()["category"].unique())
    labels = {r.store_id: f"{r.store_id} · {r.store_format} · {r.region} · NSE {r.nse}" for r in stores.itertuples()}
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Categoría", [TODAS_CAT] + cats, key="sim_cat", format_func=ui.pretty)
    sid = c2.selectbox("Tienda (opcional)", [TODAS_TIENDAS] + list(labels), key="sim_store",
                       format_func=lambda s: s if s == TODAS_TIENDAS else labels[s])
    return (None if cat == TODAS_CAT else cat), (None if sid == TODAS_TIENDAS else sid)


def _perfil_tienda(row: pd.Series) -> None:
    extras = ", ".join(x for x, f in (("farmacia", row.has_pharmacy), ("gasolinera", row.has_fuel_station)) if f) or "sin farmacia ni gasolinera"
    ui.callout(f"<b>{row.store_id}</b> · formato <b>{row.store_format}</b> · región <b>{row.region}</b> · nivel socioeconómico "
               f"<b>{row.nse}</b> · {row.size_sqm:,} m² · {row.num_checkouts} cajas · abierta en {row.opening_year} · {extras}.")


def _kpis(fc: pd.DataFrame, hist: pd.DataFrame, cat: str | None, sid: str | None, n_series: int) -> None:
    daily = fc.groupby("date")["pred_amount"].sum()
    total, units = fc["pred_amount"].sum(), fc["pred_units"].sum()
    obs_mean = hist.groupby("date")["amount_total"].sum().tail(14).mean()
    delta = (daily.mean() / obs_mean - 1) * 100 if obs_mean else float("nan")
    wape = D.wape_historico(cat, sid)
    nivel = ("serie × semana" if (cat and sid) else "tienda × semana" if sid else "categoría × semana" if cat else "total × semana")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.kpi("Venta pronosticada · 14 días", _compact(total, True),
               f"{n_series} serie{'s' if n_series != 1 else ''} · del {_fecha(daily.index.min())} al {_fecha(daily.index.max())}", accent=True)
    with c2:
        ui.kpi("Unidades estimadas", _compact(units), f"Precio medio ≈ ${total / units:,.0f} por unidad (últimas 8 semanas).")
    with c3:
        ui.kpi("Contra las 2 semanas previas", f"{delta:+.1f} %",
               f"Venta diaria media: {_compact(daily.mean(), True)} pronosticada vs {_compact(obs_mean, True)} observada.")
    with c4:
        ui.kpi("Error histórico del modelo", f"{wape:.1f} %" if wape is not None else "n/d",
               f"WAPE semanal en periodos regulares a nivel {nivel}." if wape is not None else "Ejecuta el paso 08 para calcularlo.")


def _serie_diaria(fc: pd.DataFrame, hist: pd.DataFrame) -> None:
    ui.section("Venta diaria: últimas 8 semanas y próximos 14 días", "Línea sólida: observado. Línea punteada: pronóstico del modelo.")
    h = hist.groupby("date")["amount_total"].sum()
    f = fc.groupby("date")["pred_amount"].sum()
    fig = go.Figure()
    lab = lambda idx: [f"{DIAS[d.day_of_week]} {_fecha(d)}" for d in idx]
    fig.add_scatter(x=h.index, y=h.values, mode="lines", name="Observado", line=dict(color=C.SERIES_BLUE, width=2),
                    text=lab(h.index), hovertemplate="%{text}<br>observado: $%{y:,.0f}<extra></extra>")
    # Une el último día observado con el primer pronosticado para que no quede hueco visual.
    fx = [h.index[-1]] + list(f.index)
    fy = [h.values[-1]] + list(f.values)
    fig.add_scatter(x=fx, y=fy, mode="lines+markers", name="Pronóstico", line=dict(color=C.SERIES_ORANGE, width=2, dash="dash"),
                    marker=dict(size=6, color=C.SERIES_ORANGE), text=[""] + lab(f.index),
                    hovertemplate="%{text}<br>pronóstico: $%{y:,.0f}<extra></extra>")
    cut = h.index[-1].to_pydatetime()
    fig.add_shape(type="line", x0=cut, x1=cut, y0=0, y1=1, yref="paper", line=dict(color=C.MUTED, width=1, dash="dot"))
    fig.add_annotation(x=cut, y=1, yref="paper", text="último dato", showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=11, color=C.MUTED))
    ticks = list(pd.date_range(f.index.max(), h.index.min(), freq="-14D"))[::-1]
    fig.update_xaxes(tickvals=ticks, ticktext=[_fecha(t) for t in ticks])
    fig.update_yaxes(title="MXN por día", rangemode="tozero", tickformat="~s")
    ui.show(ui.style_fig(fig, height=360, legend=True).update_layout(margin=dict(l=8, r=16, t=44, b=8)))

    dow = f.groupby(f.index.day_of_week).mean()
    peak, low = int(dow.idxmax()), int(dow.idxmin())
    ui.callout(f"<b>Patrón semanal.</b> El día más fuerte es el <b>{DIAS_LARGO[peak]}</b> ({_compact(dow.max(), True)} al día en promedio) "
               f"y el más débil el <b>{DIAS_LARGO[low]}</b> ({_compact(dow.min(), True)}): una diferencia de {dow.max() / dow.min():.1f} veces. "
               "Sirve para escalonar surtido y personal dentro de la semana.")


def _desglose(fc: pd.DataFrame, sid: str | None) -> None:
    ui.section("¿Dónde se vende?", "Pronóstico a 14 días desglosado por las características de las tiendas.")
    c1, c2 = st.columns([2, 3])
    metric = c1.radio("Medida", ["Monto (MXN)", "Unidades"], horizontal=True, key="sim_metric")
    base = c2.radio("Base", ["Total", "Promedio por tienda"], horizontal=True, key="sim_base",
                    help="El promedio por tienda permite comparar grupos con distinto número de tiendas.")
    col = "pred_amount" if metric.startswith("Monto") else "pred_units"
    money = metric.startswith("Monto")
    dims = [(k, t) for k, t in DIMS if fc[k].nunique() > 1]
    if sid:
        dims = [(k, t) for k, t in dims if k == "category"]
        if dims:
            st.caption("Con una tienda seleccionada solo tiene sentido desglosar por categoría; elige «Todas las tiendas» para ver formato, NSE y región.")
    if not dims:
        st.caption("La selección es una sola serie, así que en lugar de un desglose se muestra el pronóstico día por día.")
        d = fc.assign(Día=fc["date"].map(lambda x: f"{DIAS_LARGO[x.day_of_week]} {_fecha(x)}"))[["Día", "pred_amount", "pred_units"]]
        st.dataframe(d.rename(columns={"pred_amount": "Monto (MXN)", "pred_units": "Unidades estimadas"}), hide_index=True,
                     width="stretch", column_config={"Monto (MXN)": st.column_config.NumberColumn(format="$%,.0f"),
                                                      "Unidades estimadas": st.column_config.NumberColumn(format="%,.0f")})
        return
    for i in range(0, len(dims), 2):
        cols = st.columns(2)
        for (k, title), slot in zip(dims[i:i + 2], cols):
            g = fc.groupby(k).agg(valor=(col, "sum"), tiendas=("store_id", "nunique"))
            g["valor"] = g["valor"] / g["tiendas"] if base.startswith("Promedio") else g["valor"]
            order = {"nse": NSE_ORDER, "store_format": FORMAT_ORDER}.get(k)
            g = g.reindex([o for o in order if o in g.index]) if order else g.sort_values("valor", ascending=False)
            share = g["valor"] / g["valor"].sum() * 100
            fig = go.Figure(go.Bar(
                x=[ui.pretty(x) for x in g.index], y=g["valor"], marker_color=C.SERIES_BLUE,
                text=[_compact(v, money) for v in g["valor"]], textposition="outside", cliponaxis=False,
                customdata=list(zip(g["tiendas"], share)),
                hovertemplate="<b>%{x}</b><br>%{customdata[0]} tienda(s)<br>%{customdata[1]:.1f} % del total<extra></extra>"))
            fig.update_yaxes(visible=False, range=[0, g["valor"].max() * 1.18])
            fig.update_xaxes(showgrid=False)
            fig.update_layout(bargap=0.35)
            with slot:
                st.markdown(f"**{title}**")
                ui.show(ui.style_fig(fig, height=260).update_layout(margin=dict(l=8, r=8, t=8, b=8)))

def _csv_transparente(fc: pd.DataFrame, hist: pd.DataFrame, cat: str | None, sid: str | None, n_series: int) -> bytes:
    """CSV de descarga: primero la explicación de cómo se llegó al pronóstico (líneas ``#``) y luego los datos.

    Las líneas de explicación empiezan con ``#`` y no llevan comas, para que se lean bien en Excel y se puedan
    omitir al cargar con ``pd.read_csv(..., comment="#")``.
    """
    spec = D.model_spec()
    hp = spec["hiperparametros"]
    imp = D.importancia_ganancia().head(5)
    wape, hm = D.wape_historico(cat, sid), D.headline_metrics()
    nivel = ("serie x semana" if (cat and sid) else "tienda x semana" if sid else "categoria x semana" if cat else "total x semana")
    ultimo = hist["date"].max()
    daily = fc.groupby("date")["pred_amount"].sum()
    obs_mean = hist.groupby("date")["amount_total"].sum().tail(14).mean()
    total, units = fc["pred_amount"].sum(), fc["pred_units"].sum()
    top = "; ".join(f"{f} ({g:.1f} %)" for f, g in zip(imp["feature"], imp["% ganancia"]))
    sel = (f"categoria = {cat or 'todas'}; tienda = {sid or 'todas'} ({n_series} serie(s) tienda x categoria)")
    L = [
        "PRONOSTICO DE DEMANDA A 14 DIAS - EXPLICACION DE COMO SE OBTUVO Y RESULTADO",
        "",
        "1. QUE SE PRONOSTICA",
        f"Seleccion: {sel}.",
        f"Ultimo dia con datos reales: {ultimo.date()}. Periodo pronosticado: {fc['date'].min().date()} a {fc['date'].max().date()} (14 dias).",
        "Se pronostica la venta diaria en MXN (monto) por tienda y categoria. Cada fila del resultado es un dia de una serie tienda x categoria.",
        "",
        "2. COMO SE CALCULO",
        f"Modelo: {spec['modelo']}. Es un modelo directo a {spec['horizonte_dias']} dias: no pronostica dia por dia encadenando predicciones.",
        f"Entrenado con datos hasta {spec['entrenado_hasta']}. Usa {spec['n_features']} variables y {hp['n_estimators']} arboles (semilla {spec['semilla']}).",
        "Cada variable usa solo informacion con al menos 14 dias de antiguedad respecto al dia pronosticado; por eso no hay fuga de informacion del futuro.",
        "Variables: calendario (dia de la semana; quincena; feriados; Semana Santa; Buen Fin; Navidad); ventas pasadas (rezagos de 14 a 35 dias; promedios moviles de 7 y 28 dias; promedio del mismo dia de la semana; tendencia); promocion; y atributos de la tienda (formato; nivel socioeconomico) y categoria.",
        f"Las variables que mas pesan (ganancia del modelo): {top}. Es decir el pronostico se apoya sobre todo en lo que se vendio en el mismo dia de la semana en semanas anteriores.",
        "Capa de regimen: ajusta la prediccion solo despues del Buen Fin y de Navidad. En este horizonte no hay esos eventos y el factor es 1.0 en todas las filas (columna factor_regimen).",
        "Promociones: se supone que no hay promocion planeada. El analisis de uplift del proyecto muestra que en promedio las promociones no mueven la venta de forma medible.",
        "Unidades: el modelo pronostica monto. unidades_estimadas = monto_mxn / precio_unitario_mxn, con el precio medio por unidad de esa tienda y categoria de las ultimas 8 semanas. Asume que el precio no cambia.",
        "",
        "3. QUE TAN CONFIABLE ES",
        (f"Error historico (WAPE semanal en periodos regulares; backtest) para esta seleccion a nivel {nivel}: {wape:.1f} %." if wape is not None
         else "Error historico para esta seleccion: no disponible (ejecuta el paso 08 del pipeline)."),
        f"Referencia de toda la cadena: WAPE {hm['wape_reg']:.1f} % a nivel serie x semana; mejora de {abs(hm['vs_sma4_pts']):.1f} puntos frente a un promedio simple de 4 semanas.",
        "WAPE = suma de errores absolutos / suma de ventas reales. Un WAPE de 10 % significa que en promedio el pronostico se desvia 10 % de la venta real.",
        "Este pronostico no incluye intervalos de confianza. Los errores diarios individuales suelen ser mayores que el WAPE semanal.",
        "",
        "4. RESUMEN DEL RESULTADO",
        f"Venta pronosticada a 14 dias: {total:.0f} MXN. Unidades estimadas: {units:.0f}.",
        f"Venta diaria media pronosticada: {daily.mean():.0f} MXN frente a {obs_mean:.0f} MXN observados en los 14 dias previos ({(daily.mean() / obs_mean - 1) * 100:+.1f} %).",
        "",
        "5. COLUMNAS DEL DETALLE",
        "fecha | tienda | categoria | formato | region | nse: identifican el dia y la serie.",
        "monto_mxn: venta pronosticada del dia (prediccion del modelo x factor_regimen).",
        "factor_regimen: ajuste por Buen Fin o Navidad (1.0 = sin ajuste).",
        "precio_unitario_mxn: precio medio por unidad usado para estimar unidades.",
        "unidades_estimadas: monto_mxn / precio_unitario_mxn.",
        "",
        "6. RESULTADO (detalle diario)",
    ]
    head = "\n".join(("# " + ln).rstrip().replace(",", ";") for ln in L) + "\n"
    out = fc[["date", "store_id", "category", "store_format", "region", "nse", "pred_amount", "reset_factor", "price_unit", "pred_units"]].rename(
        columns={"date": "fecha", "store_id": "tienda", "category": "categoria", "store_format": "formato", "pred_amount": "monto_mxn",
                 "reset_factor": "factor_regimen", "price_unit": "precio_unitario_mxn", "pred_units": "unidades_estimadas"}
    ).round({"monto_mxn": 1, "factor_regimen": 3, "precio_unitario_mxn": 2, "unidades_estimadas": 1})
    out["fecha"] = out["fecha"].dt.strftime("%Y-%m-%d")
    return (head + out.to_csv(index=False)).encode("utf-8-sig")


def _detalle(fc: pd.DataFrame, hist: pd.DataFrame, cat: str | None, sid: str | None, n_series: int) -> None:
    ui.section("Detalle y descarga")
    if not sid:
        top = (fc.groupby(["store_id", "store_format", "region", "nse"], as_index=False)
                 .agg(**{"Monto (MXN)": ("pred_amount", "sum"), "Unidades": ("pred_units", "sum")})
                 .sort_values("Monto (MXN)", ascending=False).head(15)
                 .rename(columns={"store_id": "Tienda", "store_format": "Formato", "region": "Región", "nse": "NSE"}))
        st.markdown("**Las 15 tiendas con mayor venta pronosticada**")
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            "Monto (MXN)": st.column_config.NumberColumn(format="$%,.0f"), "Unidades": st.column_config.NumberColumn(format="%,.0f")})
    csv = _csv_transparente(fc, hist, cat, sid, n_series)
    st.download_button("Descargar pronóstico diario (CSV)", csv, file_name="pronostico_14_dias.csv", mime="text/csv")
    st.caption("El archivo abre con una explicación de cómo se obtuvo el pronóstico (líneas que empiezan con #) y después los datos.")


def _notas() -> None:
    with st.expander("¿Cómo se calcula este pronóstico?"):
        st.markdown(
            "- **Modelo:** el LightGBM final de la pestaña Modelo, aplicado al último día con datos. Es un modelo *directo* a 14 días: "
            "cada variable usa solo información con al menos 14 días de antigüedad, así que no hay recursión ni fuga.\n"
            "- **Capa de régimen:** en marzo no hay Buen Fin ni Navidad cerca, por lo que su factor es exactamente 1.0.\n"
            "- **Unidades:** el modelo pronostica monto; las unidades son una *estimación* = monto ÷ precio medio por unidad de esa tienda y "
            "categoría en las últimas 8 semanas. Asume que el precio medio no cambia.\n"
            "- **Promociones:** se supone que no hay promoción planeada (y el análisis de uplift muestra que, en promedio, no mueve la venta).\n"
            "- **Sin intervalos de confianza:** los cuantiles p10/p90 siguen pendientes; el error esperado se resume con el WAPE histórico.")


def render() -> None:
    try:
        fc_all, hist_all, stores = D.pronostico(), D.historia(), D.tiendas()
    except D.ArtifactMissing as e:
        st.error(str(e))
        return
    ui.section("Simulador de demanda", "Elige una categoría (y, si quieres, una tienda): el tablero devuelve el pronóstico de los próximos 14 días.")
    cat, sid = _selector(stores)
    if sid:
        _perfil_tienda(stores.set_index("store_id", drop=False).loc[sid])

    fc, hist = fc_all, hist_all
    if cat:
        fc, hist = fc[fc["category"] == cat], hist[hist["category"] == cat]
    if sid:
        fc, hist = fc[fc["store_id"] == sid], hist[hist["store_id"] == sid]
    n_series = fc[["store_id", "category"]].drop_duplicates().shape[0]

    _kpis(fc, hist, cat, sid, n_series)
    _serie_diaria(fc, hist)
    _desglose(fc, sid)
    _detalle(fc, hist, cat, sid, n_series)
    _notas()
