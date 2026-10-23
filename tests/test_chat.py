"""Pruebas del chat: herramientas contra pandas, enrutador guiado y orquestador Claude con cliente simulado."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.dashboard.chat import data_layer as L
from src.dashboard.chat import orchestrator_claude as O
from src.dashboard.chat import router_offline as R
from src.dashboard.chat import tools as T


@pytest.fixture(scope="module")
def df():
    return L.get_df()


def test_total_matches_pandas(df):
    r = T.dispatch("query_data", {"metric": "monto", "group_by": "total"})
    assert r["ok"] and r["data"][0]["valor"] == pytest.approx(df["monto"].sum())


def test_ranking_by_category_matches_pandas(df):
    r = T.dispatch("query_data", {"metric": "unidades", "group_by": "categoria"})
    exp = df.groupby("categoria")["unidades"].sum().sort_values(ascending=False)
    assert [d["categoria"] for d in r["data"]] == list(exp.index)
    assert r["data"][0]["valor"] == pytest.approx(exp.iloc[0], rel=1e-3)


def test_filters_accent_insensitive(df):
    r = T.dispatch("query_data", {"group_by": "total", "filters": {"categoria": ["electrónica"], "region": "norte"}})
    exp = df[(df["categoria"] == "Electronica") & (df["region"] == "Norte")]["monto"].sum()
    assert r["ok"] and r["data"][0]["valor"] == pytest.approx(exp)


def test_ticket_uses_paid_transactions_only(df):
    r = T.dispatch("query_data", {"metric": "ticket", "group_by": "total"})
    d = df.dropna(subset=["transacciones"])
    assert r["data"][0]["valor"] == pytest.approx(d["monto"].sum() / d["transacciones"].sum(), rel=1e-6)


@pytest.mark.parametrize("name,args", [
    ("query_data", {"group_by": "inexistente"}),
    ("query_data", {"filters": {"region": "Atlantida"}}),
    ("query_data", {"filters": {"tienda": "999"}}),
    ("analyze_data", {"analysis": "comparar", "group_by": "formato", "valores": ["solo_uno"]}),
    ("analyze_data", {"analysis": "nada"}),
    ("herramienta_falsa", {}),
    ("query_data", {"parametro_raro": 1}),
])
def test_invalid_inputs_return_recoverable_errors(name, args):
    r = T.dispatch(name, args)
    assert r["ok"] is False and r["error"]


def test_analyses_run_and_return_figure():
    for a in ["tendencia", "eventos", "patron_semanal", "anomalias", "promocion"]:
        r = T.dispatch("analyze_data", {"analysis": a})
        assert r["ok"], (a, r)
        assert r["summary"]
    assert T.dispatch("analyze_data", {"analysis": "comparar", "group_by": "formato", "valores": ["Supercenter", "Bodega"]})["figure"] is not None


def test_event_lift_signs():
    r = T.dispatch("analyze_data", {"analysis": "eventos"})
    lift = {d["tipo"]: d["lift_pct"] for d in r["data"]}
    assert lift["Buen Fin"] > 100 and lift["Navidad (temporada)"] > 0 and lift["Semana Santa"] < 0


def test_yoy_matches_manual(df):
    r = T.dispatch("analyze_data", {"analysis": "tendencia"})
    cur = df[df["mes"].isin(["2024-01", "2024-02"])]
    prev = df[df["mes"].isin(["2023-01", "2023-02"])]
    g = (cur["monto"].sum() / cur["fecha"].nunique()) / (prev["monto"].sum() / prev["fecha"].nunique()) - 1
    assert r["data"][0]["crecimiento_pct"] == pytest.approx(g * 100, abs=0.01)


def test_llm_payload_has_no_figure():
    import json
    r = T.dispatch("query_data", {"group_by": "categoria"})
    payload = json.loads(T.to_llm_payload(r))
    assert payload["ok"] and "figure" not in payload and len(payload["data"]) <= 20


# ───────── enrutador guiado ─────────
@pytest.mark.parametrize("q,tool,key,val", [
    ("¿Qué categoría vende más?", "query_data", "group_by", "categoria"),
    ("Top 5 tiendas por unidades", "query_data", "metric", "unidades"),
    ("Evolución mensual de ventas por formato", "query_data", "split_by", "formato"),
    ("¿Cuánto suben las ventas en Buen Fin?", "analyze_data", "analysis", "eventos"),
    ("Crecimiento interanual por categoría", "analyze_data", "analysis", "tendencia"),
    ("Compara Supercenter vs Bodega", "analyze_data", "analysis", "comparar"),
    ("¿Hubo días atípicos?", "analyze_data", "analysis", "anomalias"),
    ("¿Las promociones aumentan las ventas?", "analyze_data", "analysis", "promocion"),
    ("ventas de electrónica en la región Norte", "query_data", "group_by", "total"),
])
def test_router_intents(q, tool, key, val):
    p = R.interpret(q)
    assert p is not None and p.tool == tool and p.args[key] == val


def test_router_unknown_question_gives_suggestions():
    a = R.answer("cuál es el sentido de la vida")
    assert a.followups and not a.figures


def test_all_suggestions_answer_ok():
    for q in R.SUGGESTIONS:
        a = R.answer(q)
        assert a.error is None and a.figures, q


# ───────── orquestador Claude con cliente simulado ─────────
def _block(**kw):
    kw.setdefault("model_dump", lambda kw=kw: {k: v for k, v in kw.items() if k != "model_dump"})
    return SimpleNamespace(**kw)


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def _resp(content, stop):
    return SimpleNamespace(content=content, stop_reason=stop, usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0))


def test_orchestrator_runs_tool_then_answers():
    tool = _block(type="tool_use", id="t1", name="query_data", input={"group_by": "categoria"})
    client = FakeClient([_resp([tool], "tool_use"), _resp([_block(type="text", text="Abarrotes lidera.")], "end_turn")])
    a = O.run("¿qué categoría vende más?", client=client)
    assert a.text == "Abarrotes lidera." and a.tool_calls[0]["ok"] and len(a.figures) == 1 and a.usage["iterations"] == 2
    sent = client.calls[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and "summary" in sent["content"]


def test_orchestrator_recovers_from_tool_error():
    bad = _block(type="tool_use", id="t1", name="query_data", input={"filters": {"region": "Atlantida"}})
    client = FakeClient([_resp([bad], "tool_use"), _resp([_block(type="text", text="Esa región no existe.")], "end_turn")])
    a = O.run("ventas en Atlántida", client=client)
    assert a.tool_calls[0]["ok"] is False and client.calls[1]["messages"][-1]["content"][0].get("is_error")


def test_orchestrator_iteration_limit():
    loop = [_resp([_block(type="tool_use", id=f"t{i}", name="query_data", input={"group_by": "total"})], "tool_use") for i in range(O.MAX_ITER)]
    a = O.run("x", client=FakeClient(loop))
    assert a.error == "max_iterations"


def test_orchestrator_api_failure_is_graceful():
    class Boom:
        messages = SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("red caída")))
    a = O.run("hola", client=Boom())
    assert a.error and "guiado" in a.text
