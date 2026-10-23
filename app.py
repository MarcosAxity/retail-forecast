"""Dashboard de la solución · Streamlit.

Ejecutar:  uv run streamlit run app.py
Chat:      uv run streamlit run app.py -- --chat {auto,claude,offline,off}
"""
import streamlit as st

from src.dashboard import config as C
from src.dashboard import tab_modelo, tab_simulador, tab_uplift, ui
from src.dashboard.chat import tab_chat

st.set_page_config(page_title="Retail Forecast · Dashboard", page_icon="📈", layout="wide")
ui.inject_css()
ui.header("Retail Forecast", "Pronóstico de ventas por tienda × categoría · Uplift promocional", "Horizonte 14 días")

chat_mode = C.resolve_chat_mode()
labels = ["Modelo", "Simulador", "Uplift promocional"] + (["Chat"] if chat_mode != "off" else [])
tabs = st.tabs(labels)

with tabs[0]:
    tab_modelo.render()
with tabs[1]:
    tab_simulador.render()
with tabs[2]:
    tab_uplift.render()
if chat_mode != "off":
    with tabs[3]:
        tab_chat.render(chat_mode)
