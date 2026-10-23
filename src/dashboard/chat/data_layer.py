"""Capa de datos del chat: transacciones históricas observadas, con columnas en español y definiciones de métricas.

Solo lee el panel maestro (``data/raw`` → ``load_panel``); nunca modifica datos. Las herramientas del chat
consultan este DataFrame y el LLM (si lo hay) solo recibe resultados agregados.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
NSE_ORDER = ["C", "C+", "B", "A/B"]
FORMATOS = ["Supercenter", "Bodega", "Express"]
EVENTOS = ["Buen Fin", "Navidad (temporada)", "Semana Santa", "festivo", "día normal"]

# dimensión (nombre para el usuario/LLM) → columna del DataFrame
DIMS = {"categoria": "categoria", "formato": "formato", "region": "region", "nse": "nse", "tienda": "tienda",
        "mes": "mes", "semana": "semana", "dia_semana": "dia_semana", "evento": "evento", "promocion": "promocion"}
ADDITIVE = {"monto", "unidades", "transacciones"}
METRIC_LABEL = {"monto": "Venta (MXN)", "unidades": "Unidades vendidas", "transacciones": "Transacciones pagadas",
                "ticket": "Ticket promedio (MXN)"}


def norm(s) -> str:
    """Minúsculas sin acentos, para comparar nombres escritos de formas distintas."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("_", " ").split())


@lru_cache(maxsize=None)
def get_df() -> pd.DataFrame:
    from src.data.preprocess import load_panel
    p = load_panel()
    p = p[~p["is_missing_row"]].copy()
    out = pd.DataFrame({
        "fecha": p["date"], "tienda": p["store_id"], "categoria": p["category"].astype(str),
        "formato": p["store_format"].astype(str), "region": p["region"].astype(str), "nse": p["socioeconomic_level"].astype(str),
        "monto": p["amount_total"], "unidades": p["units_sold"], "tx_totales": p["total_transactions"],
        "promocion": np.where(p["has_promotion"] == 1, "con promoción", "sin promoción"),
    })
    paid = p["cash_transactions"] + p["card_transactions"]                 # NaN si falta efectivo
    out["transacciones"] = paid
    out["monto_con_tx"] = p["amount_total"].where(paid.notna())            # base del ticket (mismas filas que las transacciones)
    out["mes"] = p["date"].dt.strftime("%Y-%m")
    out["semana"] = p["date"].dt.to_period("W-SUN").dt.start_time
    out["dia_semana"] = pd.Categorical([DIAS[i] for i in p["date"].dt.day_of_week], categories=DIAS, ordered=True)
    out["es_quincena"] = p["is_payday"].astype(bool).values
    out["evento"] = np.select(
        [p["is_buen_fin"], p["is_navidad_season"], p["is_semana_santa"], p["is_official_holiday"]],
        EVENTOS[:4], default="día normal")
    return out.reset_index(drop=True)


def universe(col: str) -> list[str]:
    """Valores posibles de una dimensión, en orden natural (para colores estables y mensajes de ayuda)."""
    df = get_df()
    if col == "nse":
        return NSE_ORDER
    if col == "formato":
        return FORMATOS
    if col == "dia_semana":
        return DIAS
    if col == "evento":
        return EVENTOS
    return sorted(df[col].astype(str).unique())


def schema_summary() -> str:
    """Resumen agregado del dataset para el prompt del LLM (sin filas individuales)."""
    df = get_df()
    return (
        f"Transacciones diarias por tienda × categoría: {len(df):,} registros observados, "
        f"{df['tienda'].nunique()} tiendas, {df['categoria'].nunique()} categorías, del {df['fecha'].min():%Y-%m-%d} al {df['fecha'].max():%Y-%m-%d} "
        f"(~14 meses; una sola ocurrencia de Buen Fin, Navidad y Semana Santa).\n"
        f"- categorias: {', '.join(universe('categoria'))}\n- formatos de tienda: {', '.join(FORMATOS)}\n"
        f"- regiones: {', '.join(universe('region'))}\n- nivel socioeconómico (nse): {', '.join(NSE_ORDER)}\n"
        f"- tipos de día (evento): {', '.join(EVENTOS)}\n"
        "Métricas: 'monto' = venta en MXN (sin nulos); 'unidades' = unidades vendidas (≈3 % de filas sin dato); "
        "'transacciones' = transacciones pagadas (efectivo + tarjeta, solo filas con efectivo observado); "
        "'ticket' = monto ÷ transacciones pagadas. NO existe métrica de transacciones totales confiable: ese conteo se infla ≈22 % "
        "solo en días con promoción."
    )
