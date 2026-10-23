"""Pestaña 3 · Uplift promocional: qué es, qué encontramos y qué acciones detona."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from . import config as C
from . import data_access as D
from . import ui

DIM_LABEL = {
    "category": "Categoría", "store_format": "Formato de tienda", "region": "Región", "socioeconomic_level": "Nivel socioeconómico",
    "evento": "Tipo de día", "pago_quincena": "Quincena", "fin_semana": "Día de la semana", "nivel_venta": "Nivel de venta de la serie",
    "trimestre": "Trimestre",
}
RESULT_LABEL = {
    "amount_total (monto)": "Monto vendido", "units_sold (unidades)": "Unidades vendidas",
    "efectivo + tarjeta (transacciones pagadas)": "Transacciones pagadas", "avg_ticket (ticket)": "Ticket promedio",
}
EQ_MARGIN = 0.5      # margen de equivalencia (%) usado en la lectura principal
WATCH_BH = 0.10      # umbral BH para "candidatos a vigilar" (criterio del paso 11)


def _forest(df, label_col: str, height: int, highlight_col: str | None = None, equiv: bool = True) -> go.Figure:
    """Forest plot horizontal: punto = efecto, barra = IC95; el cero es la línea de 'sin efecto'."""
    df = df.iloc[::-1]
    flags = df[highlight_col] if highlight_col else [False] * len(df)
    colors = [C.SERIES_ORANGE if f else C.SERIES_BLUE for f in flags]
    fig = go.Figure()
    if equiv:
        fig.add_vrect(x0=-EQ_MARGIN, x1=EQ_MARGIN, fillcolor="#E2E8F0", opacity=0.55, line_width=0, layer="below")
    fig.add_vline(x=0, line_color=C.MUTED, line_width=1)
    fig.add_trace(go.Scatter(
        x=df["efecto %"], y=df[label_col], mode="markers", marker=dict(size=9, color=colors),
        error_x=dict(type="data", symmetric=False, array=df["IC95 sup %"] - df["efecto %"], arrayminus=df["efecto %"] - df["IC95 inf %"],
                     color="#94A3B8", thickness=1.6, width=4),
        hovertemplate="<b>%{y}</b><br>efecto %{x:+.2f} %<extra></extra>"))
    fig.update_xaxes(title="Efecto de la promoción sobre la venta (%)", zeroline=False, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return ui.style_fig(fig, height=height).update_layout(margin=dict(l=8, r=16, t=16, b=8))


# ───────────────────────── secciones ─────────────────────────
def _que_es() -> None:
    ui.section("¿Qué es el uplift promocional?", "La pregunta que importa no es cuánto se vendió en promoción, sino cuánto se vendió gracias a ella.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            "**Uplift = venta con promoción − venta que se habría tenido sin ella.** Es la venta *incremental*: la única que justifica "
            "el costo de una promoción (descuento, material, operación).\n\n"
            "Comparar a simple vista los días con y sin promoción engaña, porque las promociones suelen concentrarse en días o tiendas "
            "que ya vendían más. Aquí hay una ventaja: **la promoción se asignó al azar** (≈ 20 % de las series-día, sin memoria y sin relación "
            "con el segmento ni con la venta reciente). Eso la convierte en un experimento natural y permite medir el efecto sin sesgo de selección."
        )
    with c2:
        st.markdown(
            "**Cómo se midió**\n"
            "1. Se compara cada serie contra sí misma y contra el mismo día en las demás (efectos fijos de serie y fecha).\n"
            "2. Los errores se agrupan por tienda y se contrasta con permutaciones aleatorias.\n"
            "3. Se prueba si el efecto es *equivalente a cero* (TOST) y qué tan pequeño podría detectarse (efecto mínimo detectable)."
        )


def _respuesta(r: dict) -> None:
    ui.section("La respuesta corta")
    ef = r["efecto_amount_total_pct"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.kpi("Efecto sobre la venta", f"{ef['efecto %']:+.2f} %", f"IC95: {ef['IC95 inf %']:+.2f} % a {ef['IC95 sup %']:+.2f} %.", accent=True)
    with c2:
        ui.kpi("Efecto mínimo detectable", f"±{r['mde_pct']:.2f} %", "Con 80 % de potencia: efectos mayores se habrían visto.")
    with c3:
        ui.kpi("Equivalente a cero", "Sí", f"TOST con margen ±{EQ_MARGIN} %: p = {r['tost']['±0.5 %']:.3f}.")
    with c4:
        s = r["subgrupos"]
        ui.kpi("Subgrupos con efecto real", f"{s['significativos_BH']} de {s['evaluados']}",
               f"{s['p<0.05_sin_ajustar']} salen «significativos» sin corregir; por azar se esperaban ≈ 2.")
    ui.callout(
        "<b>Lectura.</b> Las promociones <b>no generan venta incremental medible</b>. El efecto promedio es ligeramente negativo y su intervalo "
        "excluye el cero, pero la magnitud (menos de 0.3 % de la venta del día) es económicamente irrelevante y cae dentro de la zona "
        "donde el efecto se considera nulo. No se afirma que la promoción <i>dañe</i> la venta: se afirma que <i>no la aumenta</i>.")


def _por_resultado(r: dict) -> None:
    ui.section("¿Cambia algo en otras medidas de venta?", "Cada punto es el efecto estimado y la barra su intervalo de confianza al 95 %. La banda gris es la zona de equivalencia (±0.5 %).")
    res = D.uplift_resultados()
    real = res[res["resultado"].isin(RESULT_LABEL)].copy()
    real["Medida"] = real["resultado"].map(RESULT_LABEL)
    ui.show(_forest(real[["Medida", "efecto %", "IC95 inf %", "IC95 sup %"]], "Medida", height=240))
    art = res[res["resultado"].str.contains("artefacto")].iloc[0]
    ui.callout(
        f"<b>Ojo con las transacciones totales.</b> El conteo <code>total_transactions</code> sube <b>{art['efecto %']:+.1f} %</b> en promoción, "
        "pero se infla artificialmente solo en esos días (no coincide con efectivo + tarjeta). Si se usara como KPI, <i>parecería</i> que la "
        "promoción funciona. En todo lo que sí mide demanda real (monto, unidades, transacciones pagadas, ticket) el efecto es cero o ligeramente negativo.",
        warn=True)


def _heterogeneidad(r: dict) -> None:
    ui.section("¿Funciona en algún segmento?", "Se buscó si la promoción ayuda en alguna categoría, formato, región, nivel socioeconómico o tipo de día.")
    sub = D.uplift_subgrupos()
    dim = st.selectbox("Cortar por", list(DIM_LABEL), format_func=DIM_LABEL.get, key="up_dim")
    s = sub[sub["dimensión"] == dim].copy().sort_values("efecto %")
    s["nivel"] = s["nivel"].map(ui.pretty)
    s["vigilar"] = s["p ajustado (BH)"] < WATCH_BH
    ui.show(_forest(s, "nivel", height=max(220, 44 * len(s) + 70), highlight_col="vigilar"))
    st.caption("Naranja: candidatos a vigilar (p ajustado por Benjamini-Hochberg < 0.10). Son hipótesis para un experimento, no conclusiones.")
    watch = sub[sub["p ajustado (BH)"] < WATCH_BH]
    names = ", ".join(f"{DIM_LABEL[a]}: <b>{ui.pretty(b)}</b> ({c:+.2f} %)" for a, b, c in zip(watch["dimensión"], watch["nivel"], watch["efecto %"]))
    s_ = r["subgrupos"]
    ui.callout(
        f"<b>Sin segmento ganador.</b> Con {s_['evaluados']} subgrupos evaluados, {s_['p<0.05_sin_ajustar']} salen «significativos» sin corregir "
        f"(por azar se esperaban ≈ 1.8) y <b>{s_['significativos_BH']} tras la corrección por comparaciones múltiples</b>. La dispersión entre las 80 tiendas también es "
        f"compatible con el azar (p = {r['heterogeneidad_tiendas_p']:.2f}). Los únicos a vigilar, y en dirección <i>negativa</i>, son {names}.")


def _dinamica(r: dict) -> None:
    ui.section("¿Adelanta o roba ventas de otros días?", "Efecto de una promoción en el día t sobre la venta de los 7 días anteriores y posteriores (k = 0 es el mismo día).")
    es = D.uplift_event_study()
    fig = go.Figure()
    fig.add_hline(y=0, line_color=C.MUTED, line_width=1)
    fig.add_scatter(x=es["k"], y=es["IC95 sup %"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=es["k"], y=es["IC95 inf %"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(42,120,214,0.18)",
                    hoverinfo="skip", name="IC95")
    fig.add_scatter(x=es["k"], y=es["efecto %"], mode="lines+markers", name="Efecto", line=dict(color=C.SERIES_BLUE, width=2),
                    marker=dict(size=6), hovertemplate="k = %{x}<br>efecto %{y:+.2f} %<extra></extra>")
    fig.update_xaxes(title="Días respecto a la promoción (k)", dtick=1)
    fig.update_yaxes(title="Efecto (%)", ticksuffix=" %")
    ui.show(ui.style_fig(fig, height=320, legend=True).update_layout(margin=dict(l=8, r=16, t=40, b=8)))
    ui.callout(
        f"<b>Sin patrón consistente.</b> El efecto acumulado en ±7 días es <b>{r['efecto_acumulado_pct']:+.2f} %</b> (IC95 −1.20 a +0.60 %): no hay evidencia de que la "
        "promoción adelante compras de días previos ni de que reste ventas a los días siguientes. Tampoco hay fatiga o refuerzo claro cuando la promoción dura varios días seguidos.")


def _negocio() -> None:
    ui.section("Traducción a negocio: ¿se paga sola una promoción?",
               "El dataset no trae costos ni márgenes: el cálculo es un ejercicio con supuestos que puedes mover.")
    neg = D.uplift_negocio().set_index("escenario")
    col_inc = [c for c in neg.columns if c.startswith("venta incremental por serie-día")][0]
    col_tot = [c for c in neg.columns if c.startswith("venta incremental total")][0]
    c1, c2 = st.columns(2)
    margen = c1.slider("Margen bruto de la categoría (%)", 5, 50, 25, key="up_margen")
    costo = c2.number_input("Costo de una promoción por tienda-categoría-día (MXN: descuento + operación)", min_value=0, value=150, step=50, key="up_costo")
    rows = []
    for esc in neg.index:
        inc = float(neg.loc[esc, col_inc])
        util = inc * margen / 100
        rows.append({"Escenario": esc.replace(" (IC95 más amplio)", "").replace(" (FE serie + fecha)", "").capitalize(),
                     "Venta incremental por promoción (MXN)": inc, "Utilidad bruta incremental (MXN)": util,
                     "Resultado tras costo (MXN)": util - costo,
                     "Venta incremental total del periodo (MXN)": float(neg.loc[esc, col_tot])})
    table = [{k: (v if k == "Escenario" else ui.money(v)) for k, v in r.items()} for r in rows]
    st.dataframe(table, hide_index=True, width="stretch")
    best = max(r["Utilidad bruta incremental (MXN)"] for r in rows)
    be = max(best, 0.0)
    ui.callout(
        f"<b>Punto de equilibrio.</b> Aun en el escenario más favorable (límite superior del intervalo), una promoción recupera como máximo "
        f"<b>{ui.money(be)}</b> de utilidad bruta con un margen de {margen} %. Si su costo real supera esa cifra, <b>no se paga sola</b>; "
        f"con el costo de {ui.money(costo)} que ingresaste, {'el resultado solo sería positivo en el escenario más favorable; con la estimación puntual es negativo' if best - costo > 0 else 'el resultado es negativo incluso en el mejor caso'}.", warn=best - costo <= 0)


def _acciones() -> None:
    ui.section("¿Qué acciones detona?", "Del resultado a la decisión.")
    imp = D.importancia_promocion()
    cards = [
        ("1 · Pausar la expansión de promociones generales",
         "Sin venta incremental medible, escalar el programa tal como está no tiene sustento. Decisión: congelar nuevas campañas amplias hasta conocer su costo.",
         "Evidencia: efecto −0.25 %, equivalente a cero."),
        ("2 · Montar un experimento con grupo de control (holdout)",
         "Reservar por categoría un grupo de tiendas-día sin promoción de forma planeada. Con la muestra actual se detectan efectos mayores a ±0.31 %; "
         "el holdout permite medir también promociones nuevas o con descuentos más profundos.",
         "Decisión: aprobar el diseño del piloto."),
        ("3 · Revisar primero Electrónica y las series de alta venta",
         "Son los únicos candidatos a vigilar (−0.74 % y −0.52 %, p ajustado ≈ 0.05), en dirección negativa. Es una hipótesis: probar pausar o rediseñar la promoción ahí.",
         "Decisión: priorizar esas categorías en el piloto."),
        ("4 · Registrar costo, descuento y margen de cada promoción",
         "Sin esos datos no se puede calcular rentabilidad ni punto de equilibrio. Hoy solo se sabe que la venta no sube.",
         "Decisión: incorporar los campos al registro de promociones."),
        ("5 · Dejar de medir promociones con «transacciones totales»",
         "Ese conteo sube ≈ 22 % solo por cómo se registra en promoción y daría una lectura falsamente positiva. Usar monto, unidades o transacciones pagadas.",
         "Decisión: cambiar el KPI de seguimiento."),
        ("6 · Pronosticar sin supuestos de promoción",
         f"El modelo de pronóstico, entrenado solo para predecir, asigna a las variables de promoción {imp['has_promotion']:.4f} % y {imp['promo_share_28']:.4f} % de su importancia: "
         "coincide con el uplift en que la promoción no mueve la venta. Por eso el simulador no necesita conocer el calendario promocional.",
         "Decisión: no condicionar el plan de reposición a la agenda de promociones."),
    ]
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for (title, body, decision), slot in zip(cards[i:i + 2], cols):
            with slot, st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(body)
                st.caption(decision)


def _limites() -> None:
    with st.expander("Límites y supuestos del análisis"):
        st.markdown(
            "- **Experimento natural, no controlado:** la asignación al azar se verificó (sin memoria, sin relación con segmento ni venta previa), pero la promoción real incluye mecánicas que el dataset no describe.\n"
            "- **Sin costos ni márgenes:** la rentabilidad se ilustra con escenarios; no se puede concluir que una promoción sea «rentable» o «no rentable» sin esos datos.\n"
            "- **Equivalencia ≠ ausencia total:** el análisis descarta efectos mayores a ±0.5 % (y, según la especificación, de −0.74 % a +0.18 %), no efectos de cualquier tamaño.\n"
            "- **Horizonte:** 14 meses, 80 tiendas; una promoción distinta (descuento más profundo, otra mecánica) podría comportarse diferente.\n"
            "- **Subgrupos:** son exploratorios; por eso se corrige por comparaciones múltiples y los candidatos se presentan como hipótesis.")


def render() -> None:
    try:
        r = D.uplift_resumen()
        _que_es()
        _respuesta(r)
        _por_resultado(r)
        _heterogeneidad(r)
        _dinamica(r)
        _negocio()
        _acciones()
        _limites()
    except D.ArtifactMissing as e:
        st.error(str(e))
