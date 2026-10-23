"""Pestaña 4 · Chat sobre las transacciones históricas (modo Claude o modo guiado sin LLM)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config as C
from .. import ui
from . import orchestrator_claude, router_offline
from .answer import Answer

KEY = "chat_history"


def _badge(mode: str) -> None:
    if mode == "claude":
        ui.callout(f"<b>Modo Claude · lenguaje libre</b> (modelo <i>{C.claude_model()}</i>). Claude decide qué consulta ejecutar; <b>todas las cifras vienen de pandas</b>, no del modelo.")
    else:
        ui.callout("<b>Modo guiado · sin LLM, sin API.</b> Entiendo preguntas con reglas y ejecuto las mismas herramientas de análisis. "
                   "Con <i>ANTHROPIC_API_KEY</i> el chat acepta lenguaje libre (ver README).")


def _render_answer(a: Answer, idx: int, detail: bool) -> None:
    st.markdown(a.text)
    for j, fig in enumerate(a.figures):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"chat_fig_{idx}_{j}")
    if detail and a.tool_calls:
        with st.expander("Detalle técnico", expanded=False):
            for j, c in enumerate(a.tool_calls):
                st.markdown(f"**{c['name']}** {'✓' if c['ok'] else '✗'}")
                st.code(str(c["args"]), language="python")
                if j < len(a.tables) and a.tables[j]:
                    st.dataframe(pd.DataFrame(a.tables[j]), width="stretch", hide_index=True)
            if a.usage:
                u = a.usage
                st.caption(f"Tokens: {u.get('input', 0)} entrada · {u.get('output', 0)} salida · {u.get('cache_read', 0)} desde caché · {u.get('iterations', 0)} iteraciones · {a.seconds:.1f} s")


def _ask(question: str, mode: str) -> Answer:
    hist = [{"role": m["role"], "content": m["text"]} for m in st.session_state.get(KEY, []) if m["role"] in ("user", "assistant")]
    if mode == "claude":
        a = orchestrator_claude.run(question, hist)
        if a.error and a.error != "max_iterations" and not a.tool_calls:
            fb = router_offline.answer(question)               # degradación elegante si Claude falla
            fb.text = f"⚠️ {a.text}\n\nMientras tanto, respondo en modo guiado:\n\n{fb.text}"
            return fb
        return a
    return router_offline.answer(question)


def render(mode: str) -> None:
    ui.section("Pregunta a los datos históricos", "Ventas, unidades, ticket y transacciones por categoría, tienda, región, formato, NSE y tiempo.")
    if C.claude_requested_without_key():
        st.warning("Pediste `--chat claude` pero no hay una `ANTHROPIC_API_KEY` válida; uso el modo guiado. Configura la clave en `.env` (ver `.env.example`).")
    _badge(mode)

    hist: list[dict] = st.session_state.setdefault(KEY, [])
    top = st.columns([6, 1.3, 1.6])
    with top[1]:
        if st.button("Nueva conversación", key="chat_reset"):
            st.session_state[KEY] = []
            st.rerun()
    with top[2]:
        detail = st.toggle("Detalle técnico", value=False, key="chat_detail")

    body = st.container()                                   # historial + respuesta, siempre por encima del cuadro de texto
    typed = st.chat_input("Escribe tu pregunta sobre las ventas históricas…", key="chat_input")
    question = st.session_state.pop("chat_pending", None) or typed

    with body:
        if not hist and not question:
            st.markdown("**Prueba con una de estas preguntas:**")
            cols = st.columns(2)
            for i, q in enumerate(router_offline.SUGGESTIONS):
                if cols[i % 2].button(q, key=f"chat_sug_{i}", width="stretch"):
                    st.session_state["chat_pending"] = q
                    st.rerun()
        for i, m in enumerate(hist):
            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    _render_answer(m["answer"], i, detail)
                else:
                    st.markdown(m["text"])
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Analizando…"):
                    a = _ask(question, mode)
                _render_answer(a, len(hist) + 1, detail)
                if a.followups:
                    st.caption("Siguientes preguntas: " + " · ".join(f"«{f}»" for f in a.followups[:3]))
            hist.append({"role": "user", "text": question})
            hist.append({"role": "assistant", "text": a.text, "answer": a})
