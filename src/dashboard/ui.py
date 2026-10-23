"""Componentes visuales compartidos: estilos, encabezado, tarjetas y plantilla de gráficas."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from . import config as C

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Mono:wght@500&display=swap');
html, body, .stApp, .stMarkdown, .stRadio label, .stSelectbox label, .stTabs [data-baseweb="tab"], [data-testid="stExpander"] summary p {{ font-family: 'Inter', sans-serif; }}
span[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.block-container {{ max-width: 1280px; padding-top: 1.2rem; padding-bottom: 3rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.app-header {{ background: linear-gradient(135deg, {C.NAVY} 0%, {C.BLUE_DARK} 100%); border-radius: 16px;
  padding: 1.6rem 2rem; margin-bottom: 1.2rem; display:flex; justify-content:space-between; align-items:center; gap:1rem; }}
.app-header h1 {{ color:#fff; font-size:1.55rem; font-weight:700; letter-spacing:-.02em; margin:0; padding:0; }}
.app-header .sub {{ color:rgba(255,255,255,.65); font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; margin-top:.3rem; }}
.badge {{ background:{C.TEAL}; color:{C.NAVY}; font-size:.7rem; font-weight:700; padding:.3rem .75rem; border-radius:999px;
  letter-spacing:.06em; text-transform:uppercase; white-space:nowrap; }}
.kpi {{ background:#fff; border:1px solid {C.GRID}; border-radius:10px; padding:1.05rem 1.25rem; height:100%;
  box-shadow:0 1px 3px rgba(0,0,0,.05); }}
.kpi .label {{ font-size:.7rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:.07em; margin-bottom:.35rem; }}
.kpi .value {{ font-size:1.9rem; font-weight:700; color:{C.NAVY}; font-family:'DM Mono', monospace; line-height:1.05; }}
.kpi .note {{ font-size:.78rem; color:{C.MUTED}; margin-top:.4rem; line-height:1.35; }}
.kpi.accent {{ background:linear-gradient(135deg, {C.NAVY} 0%, {C.BLUE_DARK} 100%); border:none; }}
.kpi.accent .label {{ color:rgba(255,255,255,.65); }} .kpi.accent .value {{ color:#fff; }} .kpi.accent .note {{ color:{C.TEAL}; }}
.callout {{ border-left:4px solid {C.BLUE_DARK}; background:#EFF6FF; border-radius:6px; padding:.8rem 1rem; font-size:.9rem;
  color:{C.INK}; margin:.5rem 0 1rem; line-height:1.5; }}
.callout.warn {{ border-left-color:{C.WARN}; background:#FFFBEB; }}
.section-title {{ font-size:1.15rem; font-weight:700; color:{C.NAVY}; margin:1.6rem 0 .1rem; letter-spacing:-.01em; }}
.section-sub {{ font-size:.85rem; color:{C.MUTED}; margin-bottom:.6rem; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; padding:.6rem 1.1rem; }}
</style>
"""


PRETTY = {"Electronica": "Electrónica", "Cuidado_Personal": "Cuidado personal"}


def pretty(name) -> str:
    """Nombre legible de categorías y niveles (acentos, sin guiones bajos)."""
    s = str(name)
    return PRETTY.get(s, s.replace("_", " "))


def money(x: float, decimals: int = 0) -> str:
    """Moneda con el signo delante del símbolo: -$1,263 (no $-1,263)."""
    return f"{'-' if x < 0 else ''}${abs(x):,.{decimals}f}"


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str, badge: str) -> None:
    st.markdown(
        f'<div class="app-header"><div><h1>{html.escape(title)}</h1><div class="sub">{html.escape(subtitle)}</div></div>'
        f'<span class="badge">{html.escape(badge)}</span></div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, note: str = "", accent: bool = False) -> None:
    cls = "kpi accent" if accent else "kpi"
    st.markdown(
        f'<div class="{cls}"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div>'
        f'<div class="note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{html.escape(sub)}</div>', unsafe_allow_html=True)


def callout(text: str, warn: bool = False) -> None:
    """``text`` admite <b> y <i>; el contenido debe venir de código propio, no de usuarios."""
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=C.INK), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, linecolor=C.GRID, tickcolor=C.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=C.GRID, linecolor="rgba(0,0,0,0)", zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
