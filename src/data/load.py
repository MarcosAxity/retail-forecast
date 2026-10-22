"""Carga y validación de los archivos crudos."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

CATEGORIES = ["Abarrotes", "Bebidas", "Cuidado_Personal", "Hogar", "Electronica", "Ropa"]
SOCIO_ORDER = ["C", "C+", "B", "A/B"]


def load_transactions(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or RAW_DIR / "transactions.csv", parse_dates=["date"])
    _validate_transactions(df)
    return df


def load_stores(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or RAW_DIR / "stores.csv")
    if df["store_id"].duplicated().any():
        raise ValueError("stores.csv: store_id duplicado")
    df["socioeconomic_level"] = pd.Categorical(
        df["socioeconomic_level"], categories=SOCIO_ORDER, ordered=True
    )
    for c in ("has_pharmacy", "has_fuel_station"):
        df[c] = df[c].astype(bool)
    return df


def load_calendar(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or RAW_DIR / "calendar.csv", parse_dates=["date"])
    if df["date"].duplicated().any():
        raise ValueError("calendar.csv: fecha duplicada")
    for c in ("is_holiday", "is_payday", "is_weekend", "is_navidad_season",
              "is_buen_fin", "is_semana_santa"):
        df[c] = df[c].astype(bool)
    return df


def _validate_transactions(df: pd.DataFrame) -> None:
    key = ["date", "store_id", "category"]
    if df.duplicated(key).any():
        raise ValueError("transactions.csv: llaves (date, store_id, category) duplicadas")
    unknown = set(df["category"]) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Categorías inesperadas: {unknown}")
    if df["amount_total"].isna().any() or (df["amount_total"] < 0).any():
        raise ValueError("amount_total con nulos o negativos")
    if not df["has_promotion"].isin([0, 1]).all():
        raise ValueError("has_promotion debe ser 0/1")
