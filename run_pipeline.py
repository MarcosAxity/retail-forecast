"""Orquestador del pipeline de punta a punta.

Ejecuta, en orden, los scripts de `scripts/` (cada uno guarda sus salidas en `outputs/<paso>/`).

    uv run python run_pipeline.py                 # todo el pipeline
    uv run python run_pipeline.py --list          # muestra los pasos, entradas y salidas sin ejecutar
    uv run python run_pipeline.py --only 11       # solo el paso 11
    uv run python run_pipeline.py --from 07 --to 10   # un tramo
    uv run python run_pipeline.py --skip 10       # todo menos el paso 10 (el más lento)

Al terminar escribe `outputs/pipeline_run.json` con el estado y la duración de cada paso.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (id, script, propósito, entradas clave, salidas clave)
STEPS = [
    ("00", "00_preparar_datos", "Panel maestro, features (h=14) y baselines", "data/raw/*.csv", "data/processed/*.parquet · outputs/reports/baselines_h14.csv"),
    ("01", "01_eda_calidad_y_dinamica", "EDA 1/3: estructura, calidad, segmentos, dinámica, promociones", "data/raw/*.csv", "outputs/01_*/"),
    ("02", "02_eda_nulos_consistencias_calendario", "EDA 2/3: nulos, consistencias, calendario, extremos", "data/raw/*.csv", "outputs/02_*/"),
    ("03", "03_eda_tiendas_promocion_ruido", "EDA 3/3: atributos de tienda, asignación de promociones, ruido", "data/raw/*.csv", "outputs/03_*/"),
    ("04", "04_auditoria_replenishment", "Auditoría de replenishment_signal (fuga de información)", "data/raw/*.csv", "outputs/04_*/"),
    ("05", "05_definicion_target", "Definición del target (monto vs. unidades vs. transacciones)", "data/raw/*.csv", "outputs/05_*/"),
    ("06", "06_features_y_baselines", "Features anti-fuga, folds de validación y baselines", "data/raw/*.csv", "outputs/06_*/"),
    ("07", "07_modelado_lightgbm", "LightGBM: backtest, ablaciones, hiperparámetros, SHAP", "data/processed/features_h14.parquet", "outputs/07_*/ · configs/model_params.json"),
    ("08", "08_backtest_oficial_y_modelo_final", "Backtest oficial (8 folds) y modelo final", "configs/model_params.json", "outputs/reports/backtest_*_h14.* · outputs/models/lgbm_h14.joblib"),
    ("09", "09_modelado_eventos", "Capa de reinicio de régimen y target multiplicativo", "outputs/reports/backtest_predictions_h14.parquet", "outputs/09_*/"),
    ("10", "10_comparacion_modelos", "Competencia de modelos y argumentación de la métrica", "outputs/reports/backtest_predictions_h14.parquet", "outputs/10_*/"),
    ("11", "11_uplift_promocional", "Fase 4: uplift promocional (Capa 2)", "data/processed/panel.parquet", "outputs/11_*/ · resumen.json"),
    ("12", "12_exportar_artifacts", "Artifacts del modelo: modelo, especificación, métricas, entorno, manifiesto", "outputs/models · outputs/reports · configs", "artifacts/"),
    ("13", "13_pronostico_proximos_14_dias", "Pronóstico de los 14 días siguientes al último dato (insumo del dashboard)", "artifacts/model/lgbm_h14.joblib · data/raw/*.csv", "outputs/13_*/pronostico_14d.csv"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="lista los pasos y sale")
    ap.add_argument("--only", nargs="+", metavar="ID", help="ids a ejecutar (p. ej. 07 11)")
    ap.add_argument("--from", dest="start", metavar="ID", help="primer paso")
    ap.add_argument("--to", dest="end", metavar="ID", help="último paso")
    ap.add_argument("--skip", nargs="+", default=[], metavar="ID", help="ids a omitir")
    a = ap.parse_args()

    ids = [s[0] for s in STEPS]
    sel = [s for s in STEPS if (not a.only or s[0] in a.only) and s[0] not in a.skip
           and (not a.start or s[0] >= a.start) and (not a.end or s[0] <= a.end)]
    if a.list:
        for i, name, what, inp, out in STEPS:
            print(f"{i}  scripts/{name}.py\n     {what}\n     entradas: {inp}\n     salidas : {out}")
        return 0

    print(f"Pipeline: {len(sel)} pasos → {', '.join(s[0] for s in sel)}\n")
    log, t_all = [], time.time()
    for i, name, what, *_ in sel:
        print(f"[{i}] {name} — {what}", flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py")], cwd=ROOT).returncode
        log.append({"id": i, "script": f"scripts/{name}.py", "estado": "ok" if rc == 0 else f"error ({rc})",
                    "segundos": round(time.time() - t0, 1)})
        print(f"[{i}] {log[-1]['estado']} en {log[-1]['segundos']} s\n", flush=True)
        if rc != 0:
            break
    out = ROOT / "outputs" / "pipeline_run.json"
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev_steps = {s["id"]: s for s in prev.get("pasos", [])}
    prev_steps.update({s["id"]: s for s in log})
    out.write_text(json.dumps({"ultima_corrida": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
                               "pasos": [prev_steps[i] for i in ids if i in prev_steps]}, indent=2, ensure_ascii=False))
    print(f"Resumen → {out.relative_to(ROOT)} | total {time.time() - t_all:.0f} s")
    return 0 if all(s["estado"] == "ok" for s in log) else 1


if __name__ == "__main__":
    raise SystemExit(main())
