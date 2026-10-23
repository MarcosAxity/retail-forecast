"""12 · Artifacts del desarrollo del modelo → carpeta `artifacts/`.

Reúne, con un manifiesto de integridad, todo lo necesario para auditar y reutilizar el modelo:

  artifacts/model/lgbm_h14.joblib         modelo final entrenado (LightGBM, objetivo L2 sobre log del monto)
  artifacts/model/model_spec.json         especificación: features, hiperparámetros, capa de régimen, protocolo
  artifacts/model/feature_importance.csv  importancia por ganancia de cada feature
  artifacts/metrics/*.csv, uplift.json    métricas de backtest, baselines y resumen del uplift
  artifacts/environment/                  versiones de librerías, Python y huella de los datos de entrada
  artifacts/manifest.json                 sha256, tamaño y paso del pipeline que produjo cada archivo

Ejecutar desde la raíz del repo:  uv run python scripts/12_exportar_artifacts.py
Entradas : outputs/models/lgbm_h14.joblib · outputs/reports/*.csv · configs/model_params.json · outputs/11_*/resumen.json
Salidas  : artifacts/ · outputs/12_exportar_artifacts/log.txt
"""
import hashlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

from _common import ROOT, stage

from src.models import regime as R
from src.models import train as T

stage("12_exportar_artifacts")
ART = ROOT / "artifacts"
for sub in ("model", "metrics", "environment"):
    (ART / sub).mkdir(parents=True, exist_ok=True)
MANIFEST: dict = {}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register(path, step: str, note: str):
    MANIFEST[str(path.relative_to(ART))] = {"sha256": sha256(path), "bytes": path.stat().st_size, "paso": step, "descripcion": note}


# --- 1. Modelo -------------------------------------------------------------------------------
bundle = joblib.load(ROOT / "outputs" / "models" / "lgbm_h14.joblib")
model, cols = bundle["model"], bundle["columns"]
shutil.copy2(ROOT / "outputs" / "models" / "lgbm_h14.joblib", ART / "model" / "lgbm_h14.joblib")
register(ART / "model" / "lgbm_h14.joblib", "08", "Modelo final LightGBM + metadatos (columnas, parámetros, fecha de corte de entrenamiento)")

spec = {
    "modelo": "LightGBM global (una sola ecuación para las 480 series tienda × categoría)",
    "horizonte_dias": bundle["horizon"],
    "target": "amount_total (MXN); se modela log(amount_total) con objetivo L2",
    "metrica_principal": "WAPE en tienda × categoría × semana (con sesgo al lado)",
    "features": cols,
    "n_features": len(cols),
    "hiperparametros": bundle["params"],
    "entrenado_hasta": bundle["train_end"],
    "capa_de_regimen": {
        "descripcion": "reinicio de régimen tras Buen Fin y Navidad (src/models/regime.py); el factor es 1.0 fuera de esas ventanas",
        "ventana_buen_fin_dias": list(R.BF_WINDOW),
        "ventana_navidad_referencia_dias": list(R.XMAS_REF_WINDOW),
        "ventana_navidad_destino_dias": list(R.XMAS_TARGET_WINDOW),
        "n_referencias_semanales": R.N_REFS, "k_rezagos_limpios": R.K_CLEAN, "minimo_validos": R.MIN_OK,
    },
    "protocolo_validacion": {"folds_desarrollo": T.DEV_FOLDS, "folds_confirmacion": T.CONFIRM_FOLDS,
                             "regla_de_adopcion_pts_wape": T.MIN_GAIN, "backtest_continuo": "48 semanas, 14 folds (scripts 09 y 10)"},
    "exclusiones": "replenishment_signal excluida por fuga de información (script 04); total_transactions no se pronostica (script 05)",
    "semilla": bundle["params"].get("random_state"),
}
(ART / "model" / "model_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
register(ART / "model" / "model_spec.json", "12", "Especificación completa del modelo y de la capa de régimen")

imp = pd.DataFrame({"feature": cols, "ganancia": model.booster_.feature_importance("gain"), "splits": model.booster_.feature_importance("split")})
imp["% ganancia"] = imp.ganancia / imp.ganancia.sum() * 100
imp.sort_values("ganancia", ascending=False).to_csv(ART / "model" / "feature_importance.csv", index=False)
register(ART / "model" / "feature_importance.csv", "12", "Importancia por ganancia (complementa SHAP del paso 07)")

# humo: el modelo guardado se recarga y predice
F = pd.read_parquet(ROOT / "data" / "processed" / "features_h14.parquet")
x = F[F["eligible"]].tail(1000)
pred = np.exp(model.predict(x[cols]))
assert np.isfinite(pred).all() and (pred > 0).all(), "predicciones no válidas"
print(f"Prueba de humo: modelo recargado, {len(pred)} predicciones finitas y positivas (mediana {np.median(pred):,.0f} MXN).")

# --- 2. Métricas -----------------------------------------------------------------------------
for src, dst, step, note in [
    (ROOT / "outputs" / "reports" / "backtest_metrics_h14.csv", "backtest_metrics_h14.csv", "08", "WAPE/MAPE/sesgo por fold y modelo (backtest oficial)"),
    (ROOT / "outputs" / "reports" / "baselines_h14.csv", "baselines_h14.csv", "00", "Baselines por fold y nivel de agregación"),
    (ROOT / "outputs" / "11_uplift_promocional" / "resumen.json", "uplift_resumen.json", "11", "Resumen del análisis de uplift promocional"),
]:
    if src.exists():
        shutil.copy2(src, ART / "metrics" / dst)
        register(ART / "metrics" / dst, step, note)
    else:
        print(f"AVISO: falta {src.relative_to(ROOT)} (¿se ejecutó el paso {step}?)")
for step_dir, names in {"10_comparacion_modelos": ["comparacion_regulares.csv", "comparacion_eventos.csv", "significancia_vs_competidores.csv"]}.items():
    for nm in names:
        src = ROOT / "outputs" / step_dir / nm
        if src.exists():
            dst = ART / "metrics" / nm
            shutil.copy2(src, dst); register(dst, step_dir[:2], f"Resultado de la competencia de modelos (outputs/{step_dir}/{nm})")
        else:
            print(f"AVISO: falta {src.relative_to(ROOT)} (¿se ejecutó el paso {step_dir[:2]}?)")

# --- 3. Entorno y datos ----------------------------------------------------------------------
pk = ["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "statsmodels", "scipy", "shap", "pyarrow", "joblib", "matplotlib"]
env = [f"python=={platform.python_version()}", f"plataforma={platform.platform()}"] + [f"{p}=={md.version(p)}" for p in pk]
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
except OSError:
    commit = ""
env.append(f"git_commit_base={commit or 'n/d'}")
(ART / "environment" / "versions.txt").write_text("\n".join(env) + "\n")
register(ART / "environment" / "versions.txt", "12", "Versiones usadas (el entorno exacto está bloqueado en uv.lock)")
fp = {}
for name in ("transactions", "stores", "calendar"):
    f = ROOT / "data" / "raw" / f"{name}.csv"
    fp[f"{name}.csv"] = {"filas": int(sum(1 for _ in open(f, encoding="utf-8")) - 1), "sha256": sha256(f)}
(ART / "environment" / "data_fingerprint.json").write_text(json.dumps(fp, indent=2))
register(ART / "environment" / "data_fingerprint.json", "12", "Huella de los datos de entrada")

(ART / "manifest.json").write_text(json.dumps({"generado": time.strftime("%Y-%m-%d %H:%M:%S"), "archivos": MANIFEST}, indent=2, ensure_ascii=False))
print(f"\nArtifacts exportados en artifacts/ ({len(MANIFEST)} archivos + manifest.json):")
for k, v in MANIFEST.items():
    print(f"  {k:70s} {v['bytes']:>10,d} B  paso {v['paso']}")
