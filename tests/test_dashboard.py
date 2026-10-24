"""Pruebas del dashboard: lectura de artefactos, cifras de portada y arranque de la app."""
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard import config as C
from src.dashboard import data_access as D

ROOT = Path(__file__).resolve().parents[1]


def test_headline_metrics_coherentes():
    h = D.headline_metrics()
    assert 6.0 < h["wape_reg"] < 9.0
    assert h["wape_reg"] < h["wape_sma4"]            # el modelo supera al baseline en regulares
    assert h["piso_serie_semana"] < h["wape_reg"]    # nadie puede estar por debajo del piso de ruido
    assert h["wape_evento_capa"] > h["wape_reg"]     # eventos siguen siendo la limitación


def test_importancia_suma_100():
    assert D.importancia_ganancia()["% ganancia"].sum() == pytest.approx(100, abs=0.5)


def test_artefacto_faltante_explica_el_paso(monkeypatch):
    monkeypatch.setattr(C, "ROOT", ROOT / "no_existe")
    with pytest.raises(D.ArtifactMissing, match="run_pipeline.py --only"):
        D._path("artifacts/model/model_spec.json", "12")


@pytest.mark.parametrize("env,key,esperado", [
    ("auto", "", "offline"), ("auto", "sk-ant-abc123", "claude"),
    ("claude", "", "offline"), ("off", "sk-ant-abc123", "off"), ("offline", "sk-ant-abc123", "offline"),
])
def test_modo_de_chat(monkeypatch, env, key, esperado):
    monkeypatch.setattr("sys.argv", ["streamlit"])
    monkeypatch.setenv("CHAT_MODE", env)
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    assert C.resolve_chat_mode() == esperado


def test_app_arranca_sin_excepciones():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.radio(key="imp_kind").set_value("SHAP (media del |valor|)").run()
    at.radio(key="bt_metric").set_value("Sesgo (%)").run()
    assert not at.exception


def test_simulador_selecciones():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    for key, val in (("sim_cat", "Electronica"), ("sim_store", "STR_005"), ("sim_cat", "Todas las categorías")):
        at.selectbox(key=key).set_value(val).run()
        assert not at.exception, (key, val)
    at.radio(key="sim_metric").set_value("Unidades").run()
    at.radio(key="sim_base").set_value("Promedio por tienda").run()
    assert not at.exception


def test_csv_simulador_explica_y_luego_resultado():
    import io
    from src.dashboard import tab_simulador as T
    fc, hist = D.pronostico(), D.historia()
    cat, sid = "Electronica", "STR_005"
    f, h = fc[(fc["category"] == cat) & (fc["store_id"] == sid)], hist[(hist["category"] == cat) & (hist["store_id"] == sid)]
    txt = T._csv_transparente(f, h, cat, sid, 1).decode("utf-8-sig")
    lineas = txt.splitlines()
    n_exp = sum(1 for ln in lineas if ln.startswith("#"))
    assert n_exp > 20 and all(ln.startswith("#") for ln in lineas[:n_exp])      # primero la explicación, luego los datos
    assert "COMO SE CALCULO" in txt and "WAPE" in txt and "Seleccion: categoria = Electronica" in txt
    df = pd.read_csv(io.StringIO(txt), comment="#")
    assert len(df) == len(f) == 14
    assert abs(df["monto_mxn"].sum() - f["pred_amount"].sum()) < 14                # redondeo a 0.1
    assert (df["unidades_estimadas"] - df["monto_mxn"] / df["precio_unitario_mxn"]).abs().max() < 1


def test_uplift_tablas_y_pestana():
    res = D.uplift_resultados()
    assert {"amount_total (monto)", "units_sold (unidades)"} <= set(res["resultado"])
    assert (res["IC95 inf %"] <= res["efecto %"]).all() and (res["efecto %"] <= res["IC95 sup %"]).all()
    sub = D.uplift_subgrupos()
    assert len(sub) == D.uplift_resumen()["subgrupos"]["evaluados"]
    assert (sub["p ajustado (BH)"] >= sub["p"] - 1e-9).all()      # la corrección nunca baja el p-valor
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    for dim in ("category", "evento", "nivel_venta"):
        at.selectbox(key="up_dim").set_value(dim).run()
        assert not at.exception, dim
    at.slider(key="up_margen").set_value(40).run()
    at.number_input(key="up_costo").set_value(0).run()
    assert not at.exception


def test_chat_tab_offline_flow(monkeypatch):
    """El chat guiado responde a una sugerencia sin API ni red."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CHAT_MODE", "offline")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    sug = [b for b in at.button if b.key and b.key.startswith("chat_sug_")]
    assert sug, "no se muestran sugerencias"
    sug[0].click().run()
    assert not at.exception
    assert any("Abarrotes" in m.markdown[0].value for m in at.chat_message if m.markdown and m.markdown[0].value)
