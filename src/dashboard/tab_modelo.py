"""Pestaña 1 · Modelo: métricas globales, competencia de modelos, backtest e importancia de variables."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import config as C
from . import data_access as D
from . import ui

PROPUESTA = "LightGBM (propuesta)"
FOLD_LABEL = {
    "semana_santa_2023": "Semana Santa 2023", "may_jun_2023": "May–Jun 2023", "sep_2023": "Sep 2023",
    "oct_2023": "Oct 2023", "feb_2024": "Feb 2024", "buen_fin_2023": "Buen Fin 2023",
    "navidad_2023": "Navidad 2023", "post_navidad_2024": "Post-Navidad 2024",
}
MODEL_LABEL = {"modelo": "Modelo (LightGBM)", "sma4": "sma4", "ma28": "ma28", "snaive": "snaive"}
MODEL_COLOR = {"modelo": C.SERIES_BLUE, "sma4": C.SERIES_ORANGE, "ma28": C.SERIES_AQUA}
SEGMENT_LABEL = {"category": "Categoría", "store_format": "Formato de tienda", "region": "Región"}
FEATURE_GROUPS = {
    "Rezagos y ventanas de venta": ("lag_", "roll_", "same_dow", "trend_ratio"),
    "Calendario y eventos": ("day_", "is_", "days_", "buen_fin_day"),
    "Promoción": ("promo", "has_promotion"),
    "Tienda y categoría": ("category", "store_format", "socioeconomic"),
}


def _pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f} %"


def _ficha(spec: dict) -> None:
    ui.section("El modelo", "Una sola ecuación global para todas las series; el resto del tablero depende de ella.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f"**Qué pronostica.** Venta diaria (`amount_total`, MXN) de cada tienda × categoría con **{spec['horizonte_dias']} días** "
            "de anticipación, para decidir cuánto reponer. Se modela el logaritmo de la venta y se regresa a pesos.\n\n"
            f"**Cómo.** {spec['modelo']}, con **{spec['n_features']} variables** y una capa de reinicio de régimen que corrige "
            "los días posteriores a Buen Fin y Navidad (fuera de esas ventanas el factor es 1.0).\n\n"
            f"**Qué se excluyó a propósito.** {spec['exclusiones']}."
        )
    with c2:
        st.markdown(
            f"- **Entrenado hasta:** {spec['entrenado_hasta']}\n"
            f"- **Métrica principal:** WAPE (con sesgo al lado)\n"
            f"- **Validación:** backtest continuo de 48 semanas, 14 folds\n"
            f"- **Semilla:** {spec['semilla']} (reproducible)"
        )
        with st.expander(f"Ver las {spec['n_features']} variables por tipo"):
            feats = list(spec["features"])
            for group, prefixes in FEATURE_GROUPS.items():
                members = [f for f in feats if f.startswith(prefixes) or f in prefixes]
                feats = [f for f in feats if f not in members]
                if members:
                    st.markdown(f"**{group}** ({len(members)}): " + ", ".join(f"`{m}`" for m in members))
            if feats:
                st.markdown("**Otras:** " + ", ".join(f"`{m}`" for m in feats))


def _kpis(h: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.kpi("WAPE · periodos regulares", _pct(h["wape_reg"]),
               f"{abs(h['vs_sma4_pts']):.2f} pts mejor que el baseline sma4 ({_pct(h['wape_sma4'])}).", accent=True)
    with c2:
        ui.kpi("Sesgo · regulares", f"{h['sesgo_reg']:+.2f} %", "Casi neutro: no sobre ni sub-pronostica de forma sistemática.")
    with c3:
        gap = h["wape_reg"] - h["piso_serie_semana"]
        ui.kpi("Piso de ruido", _pct(h["piso_serie_semana"]),
               f"Error mínimo alcanzable en serie × semana; el modelo está {gap:.1f} pts por encima.")
    with c4:
        ui.kpi("WAPE · eventos (con capa)", _pct(h["wape_evento_capa"]),
               f"Buen Fin y Navidad siguen siendo la limitación: sma4 con capa logra {_pct(h['wape_evento_sma4_capa'])}.")


def _competencia(h: dict) -> None:
    ui.section("Competencia de modelos", "Ocho modelos, mismo backtest continuo de 48 semanas. Menor WAPE es mejor.")
    reg = D.comparacion_regulares().copy()
    reg["modelo"] = reg["modelo"].replace({"LightGBM (paso 07)": PROPUESTA})
    reg = reg.sort_values("WAPE %", ascending=False)
    colors = [C.SERIES_BLUE if m == PROPUESTA else C.NEUTRAL for m in reg["modelo"]]

    left, right = st.columns(2)
    with left:
        st.markdown("**Periodos regulares** (9 folds)")
        fig = go.Figure(go.Bar(
            x=reg["WAPE %"], y=reg["modelo"], orientation="h", marker_color=colors, text=reg["WAPE %"].map(lambda v: f"{v:.2f} %"),
            textposition="outside", cliponaxis=False,
            customdata=reg[["sesgo %", "MAE (MXN)"]],
            hovertemplate="<b>%{y}</b><br>WAPE %{x:.2f} %<br>Sesgo %{customdata[0]:+.2f} %<br>MAE $%{customdata[1]:,.0f}<extra></extra>",
        ))
        fig.add_vline(x=h["piso_serie_semana"], line_dash="dash", line_color=C.MUTED, line_width=1)
        fig.update_xaxes(range=[0, max(reg["WAPE %"]) * 1.18], title="WAPE serie × semana (%)")
        fig.update_yaxes(showgrid=False)
        ui.show(ui.style_fig(fig, height=360).update_layout(margin=dict(l=8, r=16, t=44, b=8)))
        st.caption(f"Línea punteada: piso de ruido ({h['piso_serie_semana']:.1f} %), el error mínimo alcanzable en este nivel.")
    with right:
        st.markdown("**Eventos** (Buen Fin y Navidad, 4 folds)")
        ev = D.comparacion_eventos().copy()
        ev["modelo"] = ev["modelo"].replace({"LightGBM (paso 07)": PROPUESTA})
        ev = ev.sort_values("con capa", ascending=False)
        fig = go.Figure()
        fig.add_bar(y=ev["modelo"], x=ev["sin capa"], name="Sin capa de régimen", marker_color="#9ec5f4", orientation="h",
                    hovertemplate="<b>%{y}</b><br>sin capa: %{x:.2f} %<extra></extra>")
        fig.add_bar(y=ev["modelo"], x=ev["con capa"], name="Con capa de régimen", marker_color=C.SERIES_BLUE, orientation="h",
                    text=ev["con capa"].map(lambda v: f"{v:.1f} %"), textposition="outside", cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>con capa: %{x:.2f} %<extra></extra>")
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
        fig.update_xaxes(title="WAPE medio (%)", rangemode="tozero", range=[0, ev["sin capa"].max() * 1.15])
        fig.update_yaxes(showgrid=False)
        ui.show(ui.style_fig(fig, height=360, legend=True).update_layout(margin=dict(l=8, r=16, t=44, b=8)))

    sig = D.significancia()
    ties = sig[sig["p Holm"] > 0.05]["competidor"].str.replace(" + capa", "", regex=False).tolist()
    ui.callout(
        "<b>Lectura honesta.</b> En periodos regulares la propuesta supera de forma significativa a los baselines "
        "(sma4, ma28, snaive) y a ETS, pero <b>empata técnicamente con " + ", ".join(ties) + "</b>. "
        "En eventos, los baselines simples (ma28 y sma4) todavía ganan a los modelos de ML: es la limitación principal "
        "del modelo y está documentada, no oculta.", warn=False)
    with st.expander("Ver pruebas de significancia (bootstrap por semana, Wilcoxon y Holm)"):
        view = sig.rename(columns={"competidor": "Competidor"}).round(4)
        st.dataframe(view, hide_index=True, width="stretch")


def _backtest() -> None:
    ui.section("Backtest oficial por periodo", "Cada barra es un periodo no visto durante el entrenamiento.")
    bt = D.backtest_oficial()
    bt = bt[bt["modelo"].isin(MODEL_COLOR)]
    metric = st.radio("Métrica", ["WAPE (%)", "Sesgo (%)"], horizontal=True, key="bt_metric")
    col = "wape" if metric.startswith("WAPE") else "bias"

    groups = [("Periodos regulares", ["regular"]), ("Eventos y post-evento", ["evento", "post_evento"])]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[g[0] for g in groups], horizontal_spacing=0.08)
    for ci, (_, tipos) in enumerate(groups, start=1):
        sub = bt[bt["tipo"].isin(tipos)]
        order = [f for f in FOLD_LABEL if f in set(sub["fold"])]
        for modelo, color in MODEL_COLOR.items():
            s = sub[sub["modelo"] == modelo].set_index("fold").reindex(order)
            fig.add_trace(go.Bar(
                x=[FOLD_LABEL[f] for f in order], y=s[col], name=MODEL_LABEL[modelo], marker_color=color,
                legendgroup=modelo, showlegend=(ci == 1),
                hovertemplate=f"<b>{MODEL_LABEL[modelo]}</b><br>%{{x}}<br>{metric.split(' ')[0]}: %{{y:.1f}} %<extra></extra>"),
                row=1, col=ci)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title_text=metric, row=1, col=1)
    ui.show(ui.style_fig(fig, height=400, legend=True).update_layout(
        margin=dict(l=8, r=16, t=70, b=8), legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0)))
    st.caption("En post-Navidad 2024 todos los modelos sobre-pronostican (el nivel cae tras la temporada): "
               "la capa de régimen existe para corregirlo, y aun así `snaive` es el mejor en esa ventana puntual.")
    with st.expander("Ver tabla completa del backtest"):
        t = D.backtest_oficial().assign(fold=lambda d: d["fold"].map(FOLD_LABEL))
        st.dataframe(t.rename(columns={"fold": "Periodo", "tipo": "Tipo", "modelo": "Modelo", "wape": "WAPE %",
                                       "mape": "MAPE %", "bias": "Sesgo %", "mae": "MAE (MXN)", "n": "n"}).round(2),
                     hide_index=True, width="stretch")


def _importancia() -> None:
    ui.section("¿Qué variables explican el pronóstico?",
               "Dos lecturas complementarias: ganancia (cuánto reduce el error al dividir) y SHAP (cuánto mueve la predicción).")
    kind = st.radio("Método", ["Ganancia (LightGBM)", "SHAP (media del |valor|)"], horizontal=True, key="imp_kind")
    if kind.startswith("Ganancia"):
        df = D.importancia_ganancia().rename(columns={"% ganancia": "pct"})[["feature", "pct"]]
    else:
        df = D.importancia_shap().rename(columns={"% SHAP": "pct"})
    df = df.sort_values("pct", ascending=False).head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=df["pct"], y=df["feature"], orientation="h", marker_color=C.SERIES_BLUE,
                           text=df["pct"].map(lambda v: f"{v:.1f} %"), textposition="outside", cliponaxis=False,
                           hovertemplate="<b>%{y}</b><br>%{x:.1f} % de la importancia<extra></extra>"))
    fig.update_xaxes(title="% de la importancia total", range=[0, df["pct"].max() * 1.15])
    fig.update_yaxes(showgrid=False)
    ui.show(ui.style_fig(fig, height=420))

    g = D.importancia_ganancia().set_index("feature")["% ganancia"]
    persist = g[[c for c in g.index if c.startswith(("same_dow", "lag_", "roll_"))]].sum()
    cal = g[[c for c in g.index if c.startswith(("is_", "days_", "day_", "buen_fin_day"))]].sum()
    ui.callout(
        f"<b>Qué nos dice.</b> Por ganancia, el {persist:.0f} % de la información viene de la <i>persistencia</i> de la propia serie "
        f"(promedio del mismo día de la semana, rezagos y ventanas móviles) y el {cal:.0f} % del calendario "
        "(Buen Fin, Navidad, quincena, festivos). Las características de tienda aportan poco en ganancia, "
        "lo que es coherente con que el nivel de cada serie ya queda capturado por su propio historial.")


def _segmentos() -> None:
    ui.section("Desempeño por segmento", "El modelo gana a sma4 en todos los cortes; no hay un segmento donde falle.")
    seg = D.metricas_por_segmento()
    which = st.selectbox("Cortar por", list(SEGMENT_LABEL), format_func=SEGMENT_LABEL.get, key="seg_pick")
    s = seg[seg["segmento"] == which]
    fig = go.Figure()
    fig.add_bar(x=s["valor"], y=s["wape"], name="Modelo (LightGBM)", marker_color=C.SERIES_BLUE,
                text=s["wape"].map(lambda v: f"{v:.2f}"), textposition="outside",
                hovertemplate="<b>%{x}</b><br>Modelo: %{y:.2f} %<extra></extra>")
    fig.add_bar(x=s["valor"], y=s["wape_sma4"], name="sma4", marker_color=C.SERIES_ORANGE,
                text=s["wape_sma4"].map(lambda v: f"{v:.2f}"), textposition="outside",
                hovertemplate="<b>%{x}</b><br>sma4: %{y:.2f} %<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    fig.update_yaxes(title="WAPE (%)", range=[0, max(s["wape_sma4"].max(), s["wape"].max()) * 1.2])
    ui.show(ui.style_fig(fig, height=340, legend=True))
    st.caption("Fuente: desarrollo del modelo (paso 07), 5 folds regulares del backtest oficial.")


def _notas() -> None:
    ui.section("Notas de método")
    with st.expander("¿Por qué WAPE y no MAPE, MAE o RMSE?"):
        st.markdown(
            "- Se interpreta directo: **% de la venta que se planeó mal**.\n"
            "- Es comparable entre series de niveles muy distintos y no depende de la escala en pesos.\n"
            "- **MAPE** premia el sub-pronóstico (se comprobó con modelos sesgados a propósito), por eso no se usa.\n"
            "- En estos datos MAE y RMSE darían el mismo ranking; el sesgo se reporta aparte porque solo, engaña "
            "(ETS en eventos tiene menor |sesgo| y peor WAPE)."
        )
    with st.expander("Límites que conviene declarar"):
        st.markdown(
            "- **Eventos:** Buen Fin y Navidad no se pronostican mejor que con baselines simples; el criterio fijado de "
            "antemano (quedar entre los 2 mejores post-evento) no se cumple.\n"
            "- **No se afirma que sea «el mejor»:** en regulares empata con Ridge, Random Forest y XGBoost; la propuesta se "
            "justifica por superar a los baselines y por robustez en rampas.\n"
            "- **Histórico corto:** 14 meses, sin una segunda ocurrencia de los eventos anuales.\n"
            "- **Pendiente:** cuantiles p10/p90, ensamble Ridge + LightGBM y política híbrida en ventanas de evento."
        )


def render() -> None:
    try:
        spec = D.model_spec()
        h = D.headline_metrics()
        _ficha(spec)
        _kpis(h)
        _competencia(h)
        _backtest()
        _importancia()
        _segmentos()
        _notas()
    except D.ArtifactMissing as e:
        st.error(str(e))
