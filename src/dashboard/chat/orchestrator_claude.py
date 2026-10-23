"""Modo Claude: lenguaje libre con tool use (patrón del proyecto demo-ia). Claude decide qué herramienta llamar;
todas las cifras salen de las herramientas (pandas), nunca del modelo."""
from __future__ import annotations

import time

from .. import config as C
from . import data_layer as L
from . import tools as T
from .answer import Answer

MAX_ITER = 6
MAX_HISTORY_TURNS = 8

SYSTEM = """Eres el analista de datos de una cadena retail en México. Respondes en español, de forma breve y directa, preguntas sobre el HISTÓRICO de transacciones (ene-2023 a feb-2024).
Tienes herramientas de consulta (query_data, analyze_data). Reglas:
- Toda cifra que cites debe provenir de una herramienta en ESTA conversación. Nunca estimes ni inventes números.
- Llama a las herramientas que necesites (puedes encadenar varias). Si una herramienta devuelve error, corrige los parámetros y reintenta.
- Si piden un pronóstico, remite a la pestaña «Simulador»; si piden el efecto causal de las promociones, remite a «Uplift promocional» (aquí solo hay comparaciones descriptivas).
- Los eventos (Buen Fin, Navidad…) ocurren una sola vez en los datos: describe su efecto sin afirmar causalidad.
- Sé honesto con los límites: solo ~14 meses de historia, sin SKU (solo categoría), unidades derivadas del precio.
- Cierra con 1-2 frases de interpretación para negocio; no repitas todas las cifras que ya muestra la gráfica.

Esquema de los datos:
{schema}"""


def _client():
    try:
        import anthropic
    except ImportError as e:                      # pragma: no cover - depende del entorno
        raise RuntimeError("Falta el paquete 'anthropic'. Instálalo con: uv sync --extra chat") from e
    return anthropic.Anthropic(api_key=C.api_key())


def run(question: str, history: list[dict] | None = None, client=None) -> Answer:
    """``history``: mensajes previos [{role, content(str)}] (solo texto) para dar contexto de conversación."""
    t0 = time.time()
    try:
        client = client or _client()
    except Exception as e:                        # noqa: BLE001
        return Answer(f"No pude iniciar el modo Claude: {e}", "claude", error=str(e), seconds=time.time() - t0)

    messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-2 * MAX_HISTORY_TURNS:] if m.get("content")]
    messages.append({"role": "user", "content": question})
    system = [{"type": "text", "text": SYSTEM.format(schema=L.schema_summary()), "cache_control": {"type": "ephemeral"}}]
    tools = [dict(t) for t in T.TOOL_SCHEMAS]
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    ans = Answer("", "claude")
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "iterations": 0}
    try:
        for it in range(MAX_ITER):
            resp = client.messages.create(model=C.claude_model(), max_tokens=1500, system=system, tools=tools, messages=messages)
            u = resp.usage
            usage["input"] += getattr(u, "input_tokens", 0) or 0
            usage["output"] += getattr(u, "output_tokens", 0) or 0
            usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            usage["iterations"] = it + 1
            uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                ans.text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
                break
            messages.append({"role": "assistant", "content": [b.model_dump() if hasattr(b, "model_dump") else b for b in resp.content]})
            results = []
            for b in uses:
                res = T.dispatch(b.name, dict(b.input or {}))
                ans.tool_calls.append({"name": b.name, "args": dict(b.input or {}), "ok": res.get("ok"), "summary": res.get("summary") or res.get("error")})
                if res.get("ok"):
                    if res.get("figure") is not None:
                        ans.figures.append(res["figure"])
                    ans.tables.append(res.get("data") or [])
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": T.to_llm_payload(res), **({} if res.get("ok") else {"is_error": True})})
            messages.append({"role": "user", "content": results})
        else:
            ans.text = "Llegué al límite de pasos de análisis sin cerrar la respuesta. Intenta una pregunta más concreta."
            ans.error = "max_iterations"
        if not ans.text:
            ans.text = "No obtuve una respuesta del modelo."
    except Exception as e:                        # noqa: BLE001  (errores de red, clave inválida, límites…)
        name = type(e).__name__
        ans.error = f"{name}: {e}"
        ans.text = ("La clave de API no fue aceptada. Revisa ANTHROPIC_API_KEY o usa `--chat offline`." if "Authentication" in name else
                    f"Falló la llamada a Claude ({name}). Puedes reintentar o usar el modo guiado (`--chat offline`).")
    ans.usage, ans.seconds = usage, time.time() - t0
    return ans
