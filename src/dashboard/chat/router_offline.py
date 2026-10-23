"""Modo guiado: interpreta la pregunta con reglas (palabras clave + entidades) y llama a las mismas
herramientas que usa Claude. Sin LLM, sin red, determinista. Pensado para que el chat funcione al clonar el repo."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import data_layer as L
from . import tools as T
from .answer import Answer

SUGGESTIONS = [
    "¿Qué categoría vende más?",
    "Evolución mensual de ventas por formato",
    "Top 5 tiendas por unidades",
    "¿Cuánto suben las ventas en Buen Fin?",
    "¿Qué día de la semana se vende más?",
    "Crecimiento interanual por categoría",
    "Compara Supercenter vs Bodega",
    "¿Hubo días atípicos en ventas?",
    "¿Las promociones aumentan las ventas?",
    "Ticket promedio por región en Electrónica",
]

METRIC_WORDS = [("ticket", ["ticket", "compra promedio", "gasto promedio"]),
                ("unidades", ["unidad", "pieza", "volumen", "articulos"]),
                ("transacciones", ["transaccion", "tickets emitidos", "operaciones", "visitas"]),
                ("monto", ["venta", "monto", "ingreso", "factura", "dinero", "pesos", "mxn"])]
GROUP_WORDS = [("dia_semana", ["dia de la semana", "dias de la semana", "por dia", "que dia", "dia mas", "dia se vende", "fin de semana"]),
               ("mes", ["mensual", "por mes", "cada mes", "meses", "evolucion", "tendencia mensual"]),
               ("semana", ["semanal", "por semana", "cada semana", "semanas"]),
               ("tienda", ["tienda", "sucursal", "store", "str_"]),
               ("categoria", ["categoria", "departamento", "linea"]),
               ("formato", ["formato", "supercenter", "bodega", "express", "superama"]),
               ("region", ["region", "zona"]),
               ("nse", ["nse", "nivel socioeconomico", "socioeconomico"]),
               ("promocion", ["con promocion", "sin promocion"])]
ALIASES = {"categoria": {"electronica": "Electronica", "electronicos": "Electronica", "abarrotes": "Abarrotes", "bebidas": "Bebidas", "hogar": "Hogar",
                         "ropa": "Ropa", "cuidado personal": "Cuidado_Personal", "cuidado_personal": "Cuidado_Personal"}}
EVENT_WORDS = {"buen fin": "Buen Fin", "navidad": "Navidad (temporada)", "semana santa": "Semana Santa", "festivo": "festivo", "feriado": "festivo"}
MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}


@dataclass
class Plan:
    tool: str
    args: dict
    assumptions: list[str] = field(default_factory=list)


def _has(q: str, words) -> bool:
    return any(w in q for w in words)


def _entities(q: str) -> tuple[dict, list[str]]:
    """Filtros detectados en el texto: {dimensión: [valores]} + valores sueltos por dimensión (para comparar)."""
    found: dict[str, list[str]] = {}
    for dim in ("categoria", "formato", "region", "nse"):
        hits = []
        for u in L.universe(dim):
            keys = {L.norm(u), L.norm(u).replace("_", " ")}
            if dim == "categoria":
                keys |= {k for k, v in ALIASES["categoria"].items() if v == u}
            if dim == "nse" and len(u) <= 3:
                if re.search(rf"\b(nse\s*)?{re.escape(L.norm(u))}\b", q) and ("nse" in q or "nivel" in q):
                    hits.append(u)
                continue
            if any(re.search(rf"\b{re.escape(k)}\b", q) for k in keys):
                hits.append(u)
        if hits:
            found[dim] = hits
    m = re.findall(r"(?:str_?|tienda\s*(?:no\.?\s*|#\s*)?)(\d{1,3})\b", q)
    if m:
        found["tienda"] = [f"STR_{int(m[0]):03d}"]
    ev = [v for k, v in EVENT_WORDS.items() if k in q]
    if ev:
        found["evento"] = ev
    yr = re.search(r"\b(202[34])\b", q)
    for name, n in MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            y = int(yr.group(1)) if yr else (2024 if n <= 2 else 2023)
            start = f"{y}-{n:02d}-01"
            end = (L.get_df()["fecha"].max() if n == 2 and y == 2024 else __import__("pandas").Timestamp(start) + __import__("pandas").offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            found["fecha_desde"], found["fecha_hasta"] = [start], [end]
            break
    return found, []


def interpret(question: str) -> Plan | None:
    q = L.norm(question)
    ent, _ = _entities(q)
    notes: list[str] = []
    metric = next((m for m, ws in METRIC_WORDS if _has(q, ws)), None)
    if metric is None:
        metric = "monto"
        notes.append("métrica = venta (MXN) por defecto")
    group = next((g for g, ws in GROUP_WORDS if _has(q, ws)), None)
    explicit_filters = {k: v for k, v in ent.items() if k not in ("evento",)}
    desc = ("desc", "asc")[_has(q, ["menos", "menor", "peor", "bottom", "ultimos", "mas bajo", "mas baja"])]
    top = re.search(r"\b(?:top|primeras?|mejores|peores|principales)\s*(\d{1,2})\b", q) or re.search(r"\b(\d{1,2})\s*(?:mejores|peores|principales)\b", q)
    top_n = int(top.group(1)) if top else None

    # prioridad de intención: anomalías > promoción > eventos > comparar > tendencia > patrón semanal > consulta
    if _has(q, ["anomal", "atipic", "raro", "outlier", "inusual", "pico", "caida brusca"]):
        return Plan("analyze_data", {"analysis": "anomalias", "metric": metric if metric != "ticket" else "monto",
                                     "group_by": group if group in T.SPLIT_DIMS else None, "filters": explicit_filters or None, "top_n": top_n or 8}, notes)
    if _has(q, ["promo", "descuento", "oferta"]) and not _has(q, ["uplift"]):
        f = dict(explicit_filters)
        return Plan("analyze_data", {"analysis": "promocion", "filters": f or None}, notes)
    if _has(q, ["buen fin", "navidad", "semana santa", "festivo", "evento", "temporada", "quincena", "fin de quincena", "dia de pago"]) \
            and _has(q, ["suben", "aument", "efecto", "impacto", "cuanto", "lift", "pega", "afecta", "compar", "vs", "contra", "evento", "ventas", "quincena", "caen", "bajan", "crece"]):
        return Plan("analyze_data", {"analysis": "eventos", "metric": metric if metric != "ticket" else "monto",
                                     "group_by": group if group in T.SPLIT_DIMS else None, "filters": explicit_filters or None}, notes)
    versus = re.search(r"\b(vs\.?|versus|contra|frente a|comparad[oa]s? con|compara)\b", q)
    if versus or _has(q, ["compara", "diferencia entre"]):
        for dim in ("formato", "categoria", "region", "nse"):
            if len(ent.get(dim, [])) == 2:
                f = {k: v for k, v in explicit_filters.items() if k != dim}
                return Plan("analyze_data", {"analysis": "comparar", "metric": metric, "group_by": dim, "valores": ent[dim], "filters": f or None}, notes)
        notes.append("para comparar necesito dos valores de la misma dimensión (p. ej. «Supercenter vs Bodega»)")
    if _has(q, ["interanual", "yoy", "ano anterior", "crecimiento", "creci", "vs 2023", "contra 2023", "crece"]):
        g = group if group in T.SPLIT_DIMS else None
        return Plan("analyze_data", {"analysis": "tendencia", "metric": metric, "group_by": g, "filters": explicit_filters or None}, notes)
    if group == "dia_semana" and _has(q, ["patron", "comporta", "quincena", "fin de semana"]) or _has(q, ["patron semanal", "estacionalidad semanal"]):
        return Plan("analyze_data", {"analysis": "patron_semanal", "metric": metric if metric != "ticket" else "monto", "filters": explicit_filters or None}, notes)
    if group is None and _has(q, ["ventas", "venta", "vende", "monto", "unidades", "ticket", "transacciones", "cuanto", "total"]):
        if "tienda" in q or "str_" in q or ent.get("tienda"):
            group = "tienda"
        elif _has(q, ["mejor", "peor", "mas", "menos", "top", "ranking"]):
            group = "categoria"
            notes.append("agrupé por categoría por defecto")
        else:
            group = "total"
    if group is None:
        return None
    if group in ent and group not in T.TIME_DIMS and f"por {group}" not in q and group != "dia_semana":
        group = "total"                                   # «ventas en la región Norte»: filtra, no agrupa
    split = None
    if group in T.TIME_DIMS:
        for dim in ("categoria", "formato", "region", "nse"):
            if f"por {dim}" in q or (dim == "categoria" and "por categoria" in q) or (dim == "nse" and "por nse" in q):
                split = dim if dim not in explicit_filters else None
    f = {k: v for k, v in explicit_filters.items() if k != group}
    if ent.get("evento"):
        f["evento"] = ent["evento"]
    if group == "tienda" and top_n is None:
        top_n = 5 if _has(q, ["top", "mejores", "peores", "mas", "menos"]) else 10
    return Plan("query_data", {"metric": metric, "group_by": group, "order": desc, "top_n": top_n, "filters": f or None, "split_by": split}, notes)


def followups(plan: Plan) -> list[str]:
    a = plan.args
    if plan.tool == "query_data":
        g = a.get("group_by")
        out = {"categoria": ["Evolución mensual de ventas por categoría", "¿Cuánto suben las ventas en Buen Fin?"],
               "tienda": ["Compara Supercenter vs Bodega", "¿Hubo días atípicos en ventas?"],
               "mes": ["Crecimiento interanual por categoría", "¿Qué día de la semana se vende más?"],
               "dia_semana": ["¿Cuánto suben las ventas en Buen Fin?", "Crecimiento interanual por categoría"]}.get(g)
        return out or ["¿Qué categoría vende más?", "¿Las promociones aumentan las ventas?"]
    an = a.get("analysis")
    return {"eventos": ["¿Qué día de la semana se vende más?", "¿Hubo días atípicos en ventas?"],
            "tendencia": ["Evolución mensual de ventas por categoría", "Compara Supercenter vs Bodega"],
            "anomalias": ["¿Cuánto suben las ventas en Buen Fin?", "Evolución mensual de ventas por formato"],
            "comparar": ["Top 5 tiendas por unidades", "Crecimiento interanual por categoría"],
            "patron_semanal": ["¿Cuánto suben las ventas en Buen Fin?", "Evolución mensual de ventas por formato"],
            "promocion": ["Crecimiento interanual por categoría", "¿Qué categoría vende más?"]}.get(an, SUGGESTIONS[:2])


def answer(question: str) -> Answer:
    t0 = time.time()
    plan = interpret(question)
    if plan is None:
        txt = ("No pude relacionar tu pregunta con un análisis disponible. En este modo (sin LLM) entiendo preguntas sobre **ventas, unidades, ticket o transacciones** "
               "agrupadas por categoría, tienda, región, formato, NSE, mes o día de la semana; y análisis de **eventos, tendencia interanual, anomalías, comparaciones y promoción**. "
               "Prueba con alguno de los ejemplos de abajo.")
        return Answer(txt, "offline", followups=SUGGESTIONS[:4], seconds=time.time() - t0)
    res = T.dispatch(plan.tool, {k: v for k, v in plan.args.items() if v is not None})
    call = {"name": plan.tool, "args": {k: v for k, v in plan.args.items() if v is not None}, "ok": res.get("ok"), "summary": res.get("summary") or res.get("error")}
    if not res.get("ok"):
        return Answer(f"No pude resolverlo: {res.get('error')}", "offline", [call], followups=SUGGESTIONS[:3], error=res.get("error"), seconds=time.time() - t0)
    interp = f"\n\n*Interpreté: {plan.tool} · " + ", ".join(f"{k}={v}" for k, v in call["args"].items()) + (f" · {'; '.join(plan.assumptions)}" if plan.assumptions else "") + "*"
    return Answer(res["summary"] + interp, "offline", [call], [res["figure"]] if res.get("figure") is not None else [], [res.get("data") or []],
                  followups(plan), seconds=time.time() - t0)
