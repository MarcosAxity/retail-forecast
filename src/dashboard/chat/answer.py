"""Estructura común de respuesta del chat (modo guiado y modo Claude)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Answer:
    text: str
    mode: str                                   # "offline" | "claude"
    tool_calls: list[dict] = field(default_factory=list)   # [{name, args, ok, summary}]
    figures: list = field(default_factory=list)            # figuras Plotly
    tables: list = field(default_factory=list)             # filas (list[dict]) por herramienta
    followups: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None
