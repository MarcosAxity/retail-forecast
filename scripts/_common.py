"""Utilidades comunes de los scripts del pipeline.

Cada script (`scripts/NN_nombre.py`) se ejecuta desde la raíz del repo:

    uv run python scripts/NN_nombre.py

Al llamar a `stage("NN_nombre")` el script:
- crea `outputs/NN_nombre/`,
- copia todo lo que imprime a `outputs/NN_nombre/log.txt`,
- guarda cada tabla mostrada con `show()`/`display()` como `tabla_XX.csv`
  y cada figura (`savefig()`) como `fig_XX.png`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUTPUTS = ROOT / "outputs"
_state: dict = {"dir": None, "tables": 0, "figs": 0}


class _Tee:
    """Escribe en la consola y en el log del script."""

    def __init__(self, stream, path: Path):
        self.stream, self.file = stream, open(path, "w", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def stage(name: str) -> Path:
    """Inicia la etapa `name`: crea su carpeta de salida y empieza a registrar la bitácora."""
    d = OUTPUTS / name
    d.mkdir(parents=True, exist_ok=True)
    for old in list(d.glob("tabla_*.csv")) + list(d.glob("fig_*.png")):
        try:
            old.unlink()                              # evita mezclar corridas distintas
        except OSError:
            pass                                      # (carpeta sin permiso de borrado: se sobrescribe)
    _state.update(dir=d, tables=0, figs=0)
    sys.stdout = _Tee(sys.__stdout__, d / "log.txt")
    print(f"[{name}] inicio {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return d


def out_dir() -> Path:
    return _state["dir"] or OUTPUTS


def show(obj, title: str | None = None):
    """Imprime un objeto (y si es tabla, la guarda como CSV numerado)."""
    obj = getattr(obj, "data", obj)                   # Styler -> DataFrame
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        _state["tables"] += 1
        fname = f"tabla_{_state['tables']:02d}.csv"
        if _state["dir"] is not None:
            (obj.to_frame() if isinstance(obj, pd.Series) else obj).to_csv(_state["dir"] / fname)
        print(f"\n[{fname}]" + (f" {title}" if title else ""))
        print(obj.to_string())
    elif obj is not None:
        print(obj)


display = show


def savefig(title: str | None = None):
    """Guarda la figura activa como PNG numerado y la cierra."""
    _state["figs"] += 1
    fname = f"fig_{_state['figs']:02d}.png"
    if _state["dir"] is not None:
        plt.gcf().savefig(_state["dir"] / fname, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"\n[{fname}]" + (f" {title}" if title else ""))
