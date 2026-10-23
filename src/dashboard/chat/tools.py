"""Herramientas analíticas del chat (compartidas por el modo Claude y el modo guiado sin LLM).

Cada herramienta devuelve ``{"ok", "summary", "data", "figure", "meta"}``:
- ``summary``: texto con las cifras clave (lo que el LLM puede citar; el modo guiado lo muestra tal cual);
- ``data``: filas acotadas (para el LLM y para la tabla del detalle técnico);
- ``figure``: figura Plotly (solo para la UI; nunca se envía al LLM).
Los números salen siempre de pandas, nunca de un LLM. Ninguna herramienta ejecuta código libre.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import data_layer as L
from .. import config as C

PALETTE = [C.SERIES_BLUE, C.SERIES_ORANGE, C.SERIES_AQUA, "#eda100", "#e87ba4", "#008300"]
MAX_ROWS = 25
TIME_DIMS = {"mes", "semana"}
GROUP_DIMS = ["categoria", "formato", "region", "nse", "tienda", "mes", "semana", "dia_semana", "evento", "promocion", "total"]
SPLIT_DIMS = ["categoria", "formato", "region", "nse"]
METRICS = list(L.METRIC_LABEL)
ANALYSES = ["tendencia", "eventos", "patron_semanal", "anomalias", "comparar", "promocion"]


# ───────────────────────── utilidades ─────────────────────────
DIMLAB = {"dia_semana": "día de la semana", "nse": "NSE", "total": "total"}
PRETTY = {"Electronica": "Electrónica", "Cuidado_Personal": "Cuidado personal"}


def _nm(x) -> str:
    return PRETTY.get(str(x), str(x).replace("_", " "))


def _low(s: str) -> str:
    return s[:1].lower() + s[1:]


def _err(msg: str) -> dict:
    return {"ok": False, "error": msg}


def _money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f} mil M"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f} M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.1f} k"
    return f"${x:,.0f}"


def _num(x: float) -> str:
    return f"{x / 1e6:,.2f} M" if abs(x) >= 1e6 else f"{x / 1e3:,.1f} k" if abs(x) >= 1e4 else f"{x:,.0f}"


def _fmt(metric: str, x: float) -> str:
    return _money(x) if metric in ("monto", "ticket") else _num(x)


def _pct(x: float, signed: bool = True) -> str:
    return f"{x:+.1f} %" if signed else f"{x:.1f} %"


def _color_map(dim: str, values) -> dict:
    uni = L.universe(dim) if dim in SPLIT_DIMS or dim in ("promocion",) else sorted(set(values))
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(uni)}


def _style(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=16, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=12, color=C.INK), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      title=dict(font=dict(size=14), x=0, xanchor="left"))
    fig.update_xaxes(showgrid=False, linecolor=C.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=C.GRID, zeroline=False)
    return fig


def _rows(df: pd.DataFrame, n: int = MAX_ROWS) -> list[dict]:
    out = df.head(n).copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(2)
    return out.to_dict("records")


def _match(value: str, universe: list[str]) -> str | None:
    """Empata un texto del usuario/LLM con un valor válido (sin acentos, coincidencia exacta o parcial única)."""
    v = L.norm(value)
    exact = [u for u in universe if L.norm(u) == v]
    if exact:
        return exact[0]
    part = [u for u in universe if v and (v in L.norm(u) or L.norm(u) in v)]
    return part[0] if len(part) == 1 else None


def apply_filters(df: pd.DataFrame, filters: dict | None) -> tuple[pd.DataFrame, list[str]]:
    """Aplica filtros validados. Devuelve (df, descripción). Lanza ValueError con ayuda si un valor no existe."""
    notes: list[str] = []
    for key, raw in (filters or {}).items():
        if raw in (None, "", []):
            continue
        vals = raw if isinstance(raw, list) else [raw]
        if key in ("categoria", "formato", "region", "nse", "evento"):
            ok = []
            for v in vals:
                m = _match(v, L.universe(key))
                if m is None:
                    raise ValueError(f"Valor '{v}' no válido para '{key}'. Opciones: {', '.join(L.universe(key))}")
                ok.append(m)
            df = df[df[key].isin(ok)]
            notes.append(f"{key}={'/'.join(ok)}")
        elif key == "tienda":
            ids = []
            for v in vals:
                s = str(v).upper().replace("STR_", "").strip()
                if not s.isdigit() or f"STR_{int(s):03d}" not in set(df["tienda"]):
                    raise ValueError(f"Tienda '{v}' no existe (formato STR_001 … STR_080).")
                ids.append(f"STR_{int(s):03d}")
            df = df[df["tienda"].isin(ids)]
            notes.append(f"tienda={'/'.join(ids)}")
        elif key in ("fecha_desde", "fecha_hasta"):
            try:
                d = pd.Timestamp(str(vals[0]))
            except Exception as e:
                raise ValueError(f"Fecha inválida en '{key}': usa YYYY-MM-DD") from e
            df = df[df["fecha"] >= d] if key == "fecha_desde" else df[df["fecha"] <= d]
            notes.append(f"{key.replace('fecha_', '')} {d:%Y-%m-%d}")
        elif key == "promocion":
            m = _match(str(vals[0]), ["con promoción", "sin promoción"])
            if m is None:
                raise ValueError("promocion debe ser 'con' o 'sin'")
            df = df[df["promocion"] == m]
            notes.append(m)
        else:
            raise ValueError(f"Filtro desconocido '{key}'. Válidos: categoria, formato, region, nse, tienda, evento, promocion, fecha_desde, fecha_hasta")
    if df.empty:
        raise ValueError("Los filtros no dejan ningún registro.")
    return df, notes


def measure(df: pd.DataFrame, metric: str, by=None):
    """Métrica agregada. ``by`` = None → escalar; lista/columna → Series por grupo."""
    def calc(g):
        if metric == "ticket":
            tx = g["transacciones"].sum()
            return g["monto_con_tx"].sum() / tx if tx else np.nan
        return g[metric].sum()
    if by is None:
        return calc(df)
    if metric == "ticket":
        s = df.groupby(by, observed=True)[["monto_con_tx", "transacciones"]].sum()
        return (s["monto_con_tx"] / s["transacciones"]).rename("valor")
    return df.groupby(by, observed=True)[metric].sum().rename("valor")


def _periodo(df: pd.DataFrame) -> str:
    return f"{df['fecha'].min():%d-%b-%Y} a {df['fecha'].max():%d-%b-%Y}"


def _check_metric(metric: str) -> str | None:
    return None if metric in METRICS else f"metric debe ser una de: {', '.join(METRICS)}"


# ───────────────────────── Nivel 1 · descriptivo ─────────────────────────
def query_data(metric: str = "monto", group_by: str = "categoria", top_n: int | None = None, order: str = "desc",
               filters: dict | None = None, split_by: str | None = None) -> dict:
    """Totales, rankings y series de tiempo del histórico."""
    if (m := _check_metric(metric)):
        return _err(m)
    if group_by not in GROUP_DIMS:
        return _err(f"group_by debe ser uno de: {', '.join(GROUP_DIMS)}")
    if order not in ("asc", "desc"):
        return _err("order debe ser 'asc' o 'desc'")
    if split_by and (split_by not in SPLIT_DIMS or group_by not in TIME_DIMS):
        return _err("split_by solo aplica con group_by='mes' o 'semana' y debe ser categoria, formato, region o nse")
    try:
        df, notes = apply_filters(L.get_df(), filters)
    except ValueError as e:
        return _err(str(e))
    label, where = L.METRIC_LABEL[metric], (f" · filtros: {', '.join(notes)}" if notes else "")
    per = _periodo(df)

    if group_by == "total":
        v = measure(df, metric)
        return {"ok": True, "summary": f"{label} total ({per}){where}: {_fmt(metric, v)}.", "data": [{"valor": float(v)}],
                "figure": None, "meta": {"tool": "query_data", "metric": metric}}

    if group_by in TIME_DIMS:
        by = [group_by] + ([split_by] if split_by else [])
        s = measure(df, metric, by)
        if group_by == "mes":                      # evita meses parciales distorsionando la serie
            pass
        wide = s.unstack(split_by) if split_by else s.to_frame("total")
        wide = wide.sort_index()
        fig = go.Figure()
        cmap = _color_map(split_by, wide.columns) if split_by else {"total": C.SERIES_BLUE}
        for col in wide.columns:
            x = wide.index.astype(str) if group_by == "mes" else wide.index
            fig.add_scatter(x=x, y=wide[col], mode="lines+markers", name=_nm(col) if split_by else label,
                            line=dict(color=cmap[col], width=2), marker=dict(size=5),
                            hovertemplate=f"{col}<br>%{{x}}<br>%{{y:,.0f}}<extra></extra>")
        fig.update_layout(title=f"{label} por {DIMLAB.get(group_by, group_by)}")
        _style(fig, legend=bool(split_by))
        tot = wide.sum(axis=1) if metric in L.ADDITIVE else wide.mean(axis=1)
        first, last = tot.iloc[0], tot.iloc[-1]
        pk, lo = tot.idxmax(), tot.idxmin()
        summ = (f"{label} por {DIMLAB.get(group_by, group_by)} ({per}){where}: máximo en {pk} ({_fmt(metric, tot.max())}), mínimo en {lo} ({_fmt(metric, tot.min())}); "
                f"del primer al último periodo {_pct((last / first - 1) * 100)}"
                + (" (el primero/último puede ser parcial)." if group_by == "semana" or True else "."))
        data = wide.reset_index().rename(columns={group_by: group_by})
        return {"ok": True, "summary": summ, "data": _rows(data, 60), "figure": fig, "meta": {"tool": "query_data", "metric": metric}}

    s = measure(df, metric, group_by)
    if group_by == "dia_semana":
        s = s.reindex(L.DIAS)
    elif group_by == "nse":
        s = s.reindex([n for n in L.NSE_ORDER if n in s.index])
    elif group_by == "formato":
        s = s.reindex([n for n in L.FORMATOS if n in s.index])
    else:
        s = s.sort_values(ascending=(order == "asc"))
    s = s.dropna()
    keep = s.head(top_n or (10 if group_by in ("tienda", "categoria", "region") and len(s) > 10 else len(s)))
    shown = keep.copy()
    total = s.sum()
    labels = [_nm(x) for x in shown.index]
    fig = go.Figure(go.Bar(y=labels[::-1], x=shown.values[::-1], orientation="h", marker_color=C.SERIES_BLUE,
                           text=[_fmt(metric, v) for v in shown.values[::-1]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>")) if group_by not in ("dia_semana", "nse", "formato", "evento", "promocion") else \
        go.Figure(go.Bar(x=labels, y=shown.values, marker_color=C.SERIES_BLUE, text=[_fmt(metric, v) for v in shown.values],
                         textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
    fig.update_layout(title=f"{label} por {DIMLAB.get(group_by, group_by)}")
    _style(fig, height=max(260, 34 * len(shown) + 80) if group_by in ("tienda", "categoria", "region") else 320)
    if group_by in ("tienda", "categoria", "region"):
        fig.update_xaxes(visible=False); fig.update_yaxes(showgrid=False)
    else:
        fig.update_yaxes(visible=False)
    parts = [f"{_nm(k)}: {_fmt(metric, v)}" + (f" ({v / total * 100:.1f} %)" if metric in L.ADDITIVE else "") for k, v in shown.head(7).items()]
    head = "Mayores" if order == "desc" else "Menores"
    summ = f"{label} por {DIMLAB.get(group_by, group_by)} ({per}){where}. " + (f"{head} {len(shown)} de {len(s)}: " if group_by in ("tienda", "categoria", "region") else "") + "; ".join(parts) + \
           (f". Total: {_fmt(metric, total)}." if metric in L.ADDITIVE else ".")
    data = shown.rename("valor").reset_index()
    return {"ok": True, "summary": summ, "data": _rows(data), "figure": fig, "meta": {"tool": "query_data", "metric": metric}}


# ───────────────────────── Nivel 2 · diagnóstico ─────────────────────────
def _tendencia(metric: str, group_by: str | None, df: pd.DataFrame, where: str) -> dict:
    """Crecimiento interanual de los meses que ya tienen su equivalente del año anterior (ene–feb 2024 vs ene–feb 2023)."""
    months = sorted(df["mes"].unique())
    last_year = max(int(m[:4]) for m in months)
    cur = [m for m in months if int(m[:4]) == last_year and f"{last_year - 1}{m[4:]}" in months]
    if not cur:
        return _err("No hay meses con su equivalente del año anterior para calcular crecimiento interanual.")
    prev = [f"{last_year - 1}{m[4:]}" for m in cur]
    d_cur, d_prev = df[df["mes"].isin(cur)], df[df["mes"].isin(prev)]
    n_cur, n_prev = d_cur["fecha"].nunique(), d_prev["fecha"].nunique()
    key = group_by if group_by in SPLIT_DIMS else None
    a, b = measure(d_cur, metric, key), measure(d_prev, metric, key)
    if metric in L.ADDITIVE:                              # se normaliza por número de días (2024 es bisiesto)
        a, b = a / n_cur, b / n_prev
    g = (a / b - 1) * 100
    label = L.METRIC_LABEL[metric]
    periodo = f"{', '.join(cur)} vs {', '.join(prev)}"
    if key:
        g = g.sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=[_nm(i) for i in g.index], y=g.values, marker_color=C.SERIES_BLUE,
                               text=[f"{v:+.1f} %" for v in g.values], textposition="outside", cliponaxis=False))
        fig.update_layout(title=f"Crecimiento interanual de {_low(label)} ({periodo})")
        _style(fig); fig.update_yaxes(ticksuffix=" %")
        summ = (f"Crecimiento interanual de {_low(label)} ({periodo}{', por día' if metric in L.ADDITIVE else ''}){where}: " +
                "; ".join(f"{_nm(k)} {v:+.1f} %" for k, v in g.items()) + ".")
        data = g.rename("crecimiento_pct").reset_index()
    else:
        mt = measure(df, metric, "mes")
        fig = go.Figure(go.Scatter(x=mt.index, y=mt.values, mode="lines+markers", line=dict(color=C.SERIES_BLUE, width=2)))
        fig.update_layout(title=f"{label} mensual"); _style(fig)
        summ = f"Crecimiento interanual de {_low(label)} ({periodo}{', por día' if metric in L.ADDITIVE else ''}){where}: {g:+.1f} %."
        data = [{"periodo": periodo, "crecimiento_pct": round(float(g), 2)}]
    summ += " Solo hay dos meses comparables con el año anterior (el histórico es de ~14 meses)."
    return {"ok": True, "summary": summ, "data": _rows(data) if isinstance(data, pd.DataFrame) else data, "figure": fig,
            "meta": {"tool": "analyze_data", "analysis": "tendencia"}}


def _eventos(metric: str, group_by: str | None, df: pd.DataFrame, where: str) -> dict:
    """Lift de cada tipo de día contra lo esperado en días normales (misma serie y mismo día de la semana)."""
    if metric not in L.ADDITIVE:
        return _err("El análisis de eventos usa monto, unidades o transacciones (no ticket).")
    d = df.assign(serie=df["tienda"] + "|" + df["categoria"])
    base = d[(d["evento"] == "día normal") & (~d["es_quincena"])]
    exp = base.groupby(["serie", "dia_semana"], observed=True)[metric].mean().rename("esperado")
    d = d.join(exp, on=["serie", "dia_semana"])
    d["grp"] = d[group_by] if group_by in SPLIT_DIMS else "Cadena"
    d["tipo"] = np.where(d["evento"] != "día normal", d["evento"], np.where(d["es_quincena"], "quincena", "día normal"))
    r = d[d["tipo"] != "día normal"].groupby(["tipo", "grp"], observed=True).apply(
        lambda g: (g[metric].sum() / g["esperado"].sum() - 1) * 100, include_groups=False).rename("lift_pct").reset_index().dropna()
    if r.empty:
        return _err("No hay días de evento en el periodo filtrado.")
    order = [t for t in ["Buen Fin", "Navidad (temporada)", "Semana Santa", "festivo", "quincena"] if t in set(r["tipo"])]
    fig = go.Figure()
    groups = sorted(r["grp"].unique()) if group_by in SPLIT_DIMS else ["Cadena"]
    cmap = _color_map(group_by, groups) if group_by in SPLIT_DIMS else {"Cadena": C.SERIES_BLUE}
    for g in groups:
        s = r[r["grp"] == g].set_index("tipo").reindex(order)
        fig.add_bar(x=order, y=s["lift_pct"], name=_nm(g), marker_color=cmap[g],
                    text=None if group_by in SPLIT_DIMS else [f"{v:+.0f} %" for v in s["lift_pct"]], textposition="outside", cliponaxis=False)
    fig.update_layout(title=f"Efecto de cada tipo de día sobre {_low(L.METRIC_LABEL[metric])} (vs día normal equivalente)", barmode="group")
    _style(fig, legend=group_by in SPLIT_DIMS); fig.update_yaxes(ticksuffix=" %")
    tot = r.groupby("tipo")["lift_pct"].mean() if group_by in SPLIT_DIMS else r.set_index("tipo")["lift_pct"]
    summ = (f"Efecto de los días especiales sobre {_low(L.METRIC_LABEL[metric])}{where}, contra lo esperado en un día normal de la misma serie y mismo día de la semana: " +
            "; ".join(f"{t} {tot[t]:+.0f} %" for t in order) + ". Cada evento anual ocurre una sola vez en los datos: es descriptivo, no causal.")
    return {"ok": True, "summary": summ, "data": _rows(r.sort_values(["tipo", "grp"]), 60), "figure": fig,
            "meta": {"tool": "analyze_data", "analysis": "eventos"}}


def _patron_semanal(metric: str, df: pd.DataFrame, where: str) -> dict:
    if metric not in L.ADDITIVE:
        return _err("El patrón semanal usa monto, unidades o transacciones.")
    d = df[df["evento"] == "día normal"]
    daily = d.groupby(["fecha", "dia_semana"], observed=True)[metric].sum().reset_index()
    idx = daily.groupby("dia_semana", observed=True)[metric].mean()
    idx = idx / idx.mean() * 100
    fig = go.Figure(go.Bar(x=idx.index.astype(str), y=idx.values, marker_color=C.SERIES_BLUE, text=[f"{v:.0f}" for v in idx.values],
                           textposition="outside", cliponaxis=False))
    fig.add_hline(y=100, line_dash="dot", line_color=C.MUTED)
    fig.update_layout(title=f"Índice semanal de {_low(L.METRIC_LABEL[metric])} (100 = promedio de la semana)")
    _style(fig); fig.update_yaxes(visible=False)
    pk, lo = idx.idxmax(), idx.idxmin()
    pay = d[~d["es_quincena"]]
    q = d[d["es_quincena"]]
    q_lift = (q.groupby("fecha")[metric].sum().mean() / pay.groupby("fecha")[metric].sum().mean() - 1) * 100 if len(q) and len(pay) else float("nan")
    summ = (f"Patrón semanal de {_low(L.METRIC_LABEL[metric])}{where} (excluye Buen Fin, Navidad, Semana Santa y festivos): el día más fuerte es {pk} "
            f"(índice {idx.max():.0f}) y el más débil {lo} (índice {idx.min():.0f}); {idx.max() / idx.min():.1f} veces de diferencia. "
            f"Fin de semana vs entre semana: {_pct((idx[['sábado', 'domingo']].mean() / idx[L.DIAS[:5]].mean() - 1) * 100)}. "
            f"Día de quincena vs otro día (sin ajustar por día de la semana): {_pct(q_lift)}.")
    return {"ok": True, "summary": summ, "data": _rows(idx.rename("indice").reset_index()), "figure": fig,
            "meta": {"tool": "analyze_data", "analysis": "patron_semanal"}}


def _anomalias(metric: str, group_by: str | None, top_n: int, df: pd.DataFrame, where: str, incluir_eventos: bool) -> dict:
    if metric not in L.ADDITIVE:
        return _err("La detección de anomalías usa monto, unidades o transacciones.")
    key = group_by if group_by in SPLIT_DIMS else None
    d = df.assign(grp=df[key] if key else "Cadena")
    daily = d.groupby(["grp", "fecha"], observed=True)[metric].sum().reset_index()
    ev = df.drop_duplicates("fecha").set_index("fecha")["evento"]
    daily["evento"] = daily["fecha"].map(ev)
    qn = df.drop_duplicates("fecha").set_index("fecha")["es_quincena"]
    daily["quincena"] = daily["fecha"].map(qn)
    evd = ev[ev != "día normal"].index
    daily["cerca_evento"] = daily["fecha"].map(lambda f: bool(len(evd)) and bool((abs((evd - f).days) <= 7).any()))
    out = []
    for g, s in daily.groupby("grp"):
        s = s.sort_values("fecha").reset_index(drop=True)
        s["dow"] = s["fecha"].dt.day_of_week
        s["esperado"] = s.groupby("dow")[metric].transform(lambda x: x.rolling(9, center=True, min_periods=4).median())
        s["desv"] = s[metric] / s["esperado"] - 1
        base = s[(s["evento"] == "día normal") & (~s["quincena"]) & (~s["cerca_evento"])]["desv"].dropna()
        mad = (base - base.median()).abs().median() * 1.4826
        s["z"] = (s["desv"] - base.median()) / (mad if mad > 0 else np.nan)
        out.append(s)
    a = pd.concat(out)
    cand = a[a["z"].abs() >= 3.5].copy()
    if not incluir_eventos:
        cand = cand[(cand["evento"] == "día normal") & (~cand["quincena"]) & (~cand["cerca_evento"])]
    cand = cand.reindex(cand["z"].abs().sort_values(ascending=False).index).head(top_n)
    if cand.empty:
        return {"ok": True, "summary": f"No se detectaron días atípicos{where} (|z| robusto ≥ 3.5, excluyendo días de evento, quincena y ±7 días de un evento).", "data": [],
                "figure": None, "meta": {"tool": "analyze_data", "analysis": "anomalias"}}
    cand["etiqueta"] = cand["grp"].astype(str).map(_nm) + " · " + cand["fecha"].dt.strftime("%d-%b-%y")
    fig = go.Figure(go.Bar(y=cand["etiqueta"][::-1], x=cand["desv"][::-1] * 100, orientation="h", marker_color=C.SERIES_BLUE,
                           text=[f"{v:+.0f} %" for v in cand["desv"][::-1] * 100], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Días atípicos: desviación contra lo esperado (mismo día de la semana, vecindad de 9 semanas)")
    _style(fig, height=max(260, 34 * len(cand) + 80)); fig.update_xaxes(ticksuffix=" %"); fig.update_yaxes(showgrid=False)
    summ = (f"Días atípicos de {_low(L.METRIC_LABEL[metric])}{where}: " + "; ".join(
        f"{r.grp if key else 'Cadena'} el {r.fecha:%d-%b-%Y} ({r.desv * 100:+.0f} % vs esperado, {_fmt(metric, getattr(r, metric))} observado)" for r in cand.itertuples()) +
        ". Criterio: z robusto ≥ 3.5 sobre la desviación contra la mediana del mismo día de la semana; se excluyen días de evento, de quincena y ±7 días de un evento.")
    return {"ok": True, "summary": summ, "data": _rows(cand[["grp", "fecha", metric, "esperado", "desv", "z", "evento"]], top_n), "figure": fig,
            "meta": {"tool": "analyze_data", "analysis": "anomalias"}}


def _comparar(metric: str, group_by: str | None, valores: list | None, df: pd.DataFrame, where: str) -> dict:
    if group_by not in SPLIT_DIMS or not valores or len(valores) != 2:
        return _err("Para comparar indica group_by (categoria, formato, region o nse) y exactamente 2 valores en 'valores'.")
    vs = []
    for v in valores:
        m = _match(v, L.universe(group_by))
        if m is None:
            return _err(f"Valor '{v}' no válido para '{group_by}'. Opciones: {', '.join(L.universe(group_by))}")
        vs.append(m)
    d = df[df[group_by].isin(vs)]
    n_days = d["fecha"].nunique()
    stores = d.groupby(group_by)["tienda"].nunique()
    rows = []
    for v in vs:
        g = d[d[group_by] == v]
        tot = measure(g, metric)
        rows.append({group_by: v, "total": float(tot), "tiendas": int(stores[v]),
                     "por_tienda_por_dia": float(tot / stores[v] / n_days) if metric in L.ADDITIVE else float(tot)})
    a, b = rows
    fig = go.Figure()
    mt = measure(d, metric, ["mes", group_by]).unstack(group_by)
    for v, color in zip(vs, (C.SERIES_BLUE, C.SERIES_ORANGE)):
        fig.add_scatter(x=mt.index, y=mt[v], mode="lines+markers", name=_nm(v), line=dict(color=color, width=2), marker=dict(size=5))
    fig.update_layout(title=f"{L.METRIC_LABEL[metric]} por mes: {vs[0]} vs {vs[1]}"); _style(fig, legend=True)
    if metric in L.ADDITIVE:
        r1 = a["total"] / b["total"]; r2 = a["por_tienda_por_dia"] / b["por_tienda_por_dia"]
        summ = (f"{vs[0]} vs {vs[1]} en {_low(L.METRIC_LABEL[metric])} ({_periodo(d)}){where}: {vs[0]} {_fmt(metric, a['total'])} ({a['tiendas']} tiendas) y "
                f"{vs[1]} {_fmt(metric, b['total'])} ({b['tiendas']} tiendas); el total de {vs[0]} es {r1:.2f} veces el de {vs[1]}, y por tienda y por día {r2:.2f} veces "
                f"({_fmt(metric, a['por_tienda_por_dia'])} vs {_fmt(metric, b['por_tienda_por_dia'])}).")
    else:
        summ = f"Ticket promedio ({_periodo(d)}){where}: {vs[0]} {_fmt(metric, a['total'])} vs {vs[1]} {_fmt(metric, b['total'])} ({_pct((a['total'] / b['total'] - 1) * 100)})."
    return {"ok": True, "summary": summ, "data": _rows(pd.DataFrame(rows)), "figure": fig, "meta": {"tool": "analyze_data", "analysis": "comparar"}}


def _promocion(df: pd.DataFrame, where: str) -> dict:
    g = df.groupby("promocion")
    m = pd.DataFrame({
        "Monto medio por serie-día": g["monto"].mean(), "Unidades medias": g["unidades"].mean(),
        "Transacciones pagadas medias": g["transacciones"].mean(),
        "Ticket promedio": g["monto_con_tx"].sum() / g["transacciones"].sum(),
        "Transacciones totales (artefacto)": g["tx_totales"].mean()})
    if set(m.index) != {"con promoción", "sin promoción"}:
        return _err("El filtro dejó solo días con o sin promoción; no hay contra qué comparar.")
    diff = (m.loc["con promoción"] / m.loc["sin promoción"] - 1) * 100
    colors = [C.SERIES_BLUE if "artefacto" not in k else C.NEUTRAL for k in diff.index]
    fig = go.Figure(go.Bar(x=diff.index, y=diff.values, marker_color=colors, text=[f"{v:+.1f} %" for v in diff.values], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Días con promoción vs sin promoción (diferencia de promedios)"); _style(fig); fig.update_yaxes(ticksuffix=" %"); fig.update_xaxes(tickangle=-15)
    summ = (f"Comparación simple de días con vs sin promoción{where}: " + "; ".join(f"{k} {v:+.1f} %" for k, v in diff.items()) +
            ". Las transacciones totales suben solo por un artefacto de registro en promoción (no hay efecto real en monto, unidades ni transacciones pagadas). "
            "Para el efecto causal con intervalos de confianza, ver la pestaña «Uplift promocional».")
    return {"ok": True, "summary": summ, "data": _rows(m.reset_index()), "figure": fig, "meta": {"tool": "analyze_data", "analysis": "promocion"}}


def analyze_data(analysis: str, metric: str = "monto", group_by: str | None = None, filters: dict | None = None,
                 valores: list | None = None, top_n: int = 8, incluir_eventos: bool = False) -> dict:
    """Diagnóstico: tendencia interanual, efecto de eventos, patrón semanal, anomalías, comparación y promoción."""
    if analysis not in ANALYSES:
        return _err(f"analysis debe ser uno de: {', '.join(ANALYSES)}")
    if (m := _check_metric(metric)):
        return _err(m)
    if group_by not in (None, *SPLIT_DIMS):
        return _err(f"group_by debe ser uno de: {', '.join(SPLIT_DIMS)} (o vacío)")
    try:
        df, notes = apply_filters(L.get_df(), filters)
    except ValueError as e:
        return _err(str(e))
    where = f" · filtros: {', '.join(notes)}" if notes else ""
    if analysis == "tendencia":
        return _tendencia(metric, group_by, df, where)
    if analysis == "eventos":
        return _eventos(metric, group_by, df, where)
    if analysis == "patron_semanal":
        return _patron_semanal(metric, df, where)
    if analysis == "anomalias":
        return _anomalias(metric, group_by, max(1, min(int(top_n), 15)), df, where, incluir_eventos)
    if analysis == "comparar":
        return _comparar(metric, group_by, valores, df, where)
    return _promocion(df, where)


# ───────────────────────── registro y despacho ─────────────────────────
TOOL_REGISTRY = {"query_data": (query_data, 1), "analyze_data": (analyze_data, 2)}
TOOL_LEVELS = {k: lvl for k, (_, lvl) in TOOL_REGISTRY.items()}


def dispatch(name: str, args: dict | None) -> dict:
    """Ejecuta una herramienta con validación; nunca lanza excepciones."""
    if name not in TOOL_REGISTRY:
        return _err(f"Herramienta desconocida '{name}'. Disponibles: {', '.join(TOOL_REGISTRY)}")
    fn = TOOL_REGISTRY[name][0]
    args = dict(args or {})
    bad = [k for k in args if k not in inspect.signature(fn).parameters]
    if bad:
        return _err(f"Parámetros no válidos para {name}: {', '.join(bad)}")
    try:
        return fn(**args)
    except Exception as e:                       # noqa: BLE001  (se devuelve como error recuperable para el LLM)
        return _err(f"Error inesperado en {name}: {type(e).__name__}: {e}")


def to_llm_payload(res: dict, max_rows: int = 20) -> str:
    """Versión compacta para el LLM: resumen + filas acotadas, sin figura."""
    import json
    if not res.get("ok"):
        return json.dumps({"ok": False, "error": res.get("error")}, ensure_ascii=False)
    return json.dumps({"ok": True, "summary": res["summary"], "data": (res.get("data") or [])[:max_rows]}, ensure_ascii=False, default=str)


_FILTERS = {
    "type": "object", "additionalProperties": False,
    "description": "Filtros opcionales. Cada dimensión acepta un texto o lista de textos (sin distinguir acentos/mayúsculas).",
    "properties": {
        "categoria": {"type": "array", "items": {"type": "string"}}, "formato": {"type": "array", "items": {"type": "string"}},
        "region": {"type": "array", "items": {"type": "string"}}, "nse": {"type": "array", "items": {"type": "string"}},
        "tienda": {"type": "array", "items": {"type": "string"}, "description": "Ej. STR_012 o 12"},
        "evento": {"type": "array", "items": {"type": "string"}, "description": "Buen Fin, Navidad (temporada), Semana Santa, festivo, día normal"},
        "promocion": {"type": "string", "enum": ["con", "sin"]},
        "fecha_desde": {"type": "string", "description": "YYYY-MM-DD"}, "fecha_hasta": {"type": "string", "description": "YYYY-MM-DD"}},
}
TOOL_SCHEMAS = [
    {"name": "query_data",
     "description": ("Nivel 1 - DESCRIPTIVO (¿qué pasó?). Totales, rankings y series de tiempo del histórico de transacciones (tienda × categoría × día). "
                     "Úsala para: top/bottom de categorías, tiendas, regiones, formatos o NSE; evolución mensual o semanal (group_by='mes'/'semana', con split_by "
                     "para desglosar); ventas por día de la semana, por tipo de día o por promoción; y totales con filtros. NO explica causas ni compara eventos "
                     "(usa analyze_data). El campo summary ya trae las cifras clave."),
     "input_schema": {"type": "object", "properties": {
         "metric": {"type": "string", "enum": METRICS, "description": "monto (default), unidades, transacciones (pagadas) o ticket"},
         "group_by": {"type": "string", "enum": GROUP_DIMS}, "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
         "order": {"type": "string", "enum": ["desc", "asc"]}, "filters": _FILTERS,
         "split_by": {"type": "string", "enum": SPLIT_DIMS, "description": "Solo con group_by mes/semana"}}, "required": []}},
    {"name": "analyze_data",
     "description": ("Nivel 2 - DIAGNÓSTICO (¿por qué pasó?). analysis: 'tendencia' (crecimiento interanual ene-feb 2024 vs 2023), 'eventos' (efecto de Buen Fin, "
                     "Navidad, Semana Santa, festivos y quincena vs un día normal equivalente), 'patron_semanal' (índice por día de la semana y efecto quincena), "
                     "'anomalias' (días atípicos), 'comparar' (dos valores de una misma dimensión; requiere group_by y valores=[A,B]) y 'promocion' (días con vs sin "
                     "promoción; para el efecto causal remite a la pestaña Uplift). No pronostica."),
     "input_schema": {"type": "object", "properties": {
         "analysis": {"type": "string", "enum": ANALYSES}, "metric": {"type": "string", "enum": METRICS},
         "group_by": {"type": "string", "enum": SPLIT_DIMS}, "filters": _FILTERS,
         "valores": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
         "top_n": {"type": "integer", "minimum": 1, "maximum": 15}, "incluir_eventos": {"type": "boolean"}}, "required": ["analysis"]}},
]
