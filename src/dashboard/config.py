"""Configuración central del dashboard: rutas, paleta y modo de chat.

Todo lo que el dashboard muestra sale de ``artifacts/`` y ``outputs/`` (se regenera con el
pipeline); aquí solo se declara dónde está cada cosa.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# --- Paleta (chrome: navy/blue/teal; series: paleta categórica validada) -----------------
NAVY, BLUE_DARK, TEAL = "#0B1E3D", "#1A4A8A", "#00C2A8"
INK, MUTED, GRID, SURFACE = "#0F172A", "#64748B", "#E2E8F0", "#FFFFFF"
SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#B4BCC6"          # barras no destacadas
WARN = "#B45309"

def _load_dotenv() -> None:
    """Carga ``.env`` (sin dependencias) sin pisar variables ya definidas en el entorno."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# --- Chat ---------------------------------------------------------------------------------
CHAT_MODES = ("auto", "claude", "offline", "off")


def _cli_chat_arg() -> str | None:
    """Lee ``--chat`` de los argumentos que streamlit pasa tras ``--``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--chat", choices=CHAT_MODES)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.chat


def has_api_key() -> bool:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return key.startswith("sk-ant-") and "xxxx" not in key


def api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "")


def claude_model() -> str:
    return os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")


def claude_requested_without_key() -> bool:
    """True si se pidió explícitamente Claude pero no hay clave válida (la pestaña avisa y degrada al modo guiado)."""
    return (_cli_chat_arg() or os.getenv("CHAT_MODE", "auto").lower()) == "claude" and not has_api_key()


def resolve_chat_mode() -> str:
    """Devuelve el modo efectivo: ``claude``, ``offline`` u ``off``.

    Prioridad: argumento ``--chat`` > variable ``CHAT_MODE`` > ``auto``.
    ``auto`` usa Claude si hay ANTHROPIC_API_KEY válida y, si no, el modo guiado sin LLM.
    """
    requested = _cli_chat_arg() or os.getenv("CHAT_MODE", "auto").lower()
    if requested not in CHAT_MODES:
        requested = "auto"
    if requested == "auto":
        return "claude" if has_api_key() else "offline"
    if requested == "claude" and not has_api_key():
        return "offline"   # se pidió Claude pero no hay llave: degradar con aviso en la pestaña
    return requested
