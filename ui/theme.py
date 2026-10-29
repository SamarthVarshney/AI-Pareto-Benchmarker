"""Minimalist dark-mode engineering CSS theme, breadcrumb header, and card components."""

from __future__ import annotations

from typing import Any

import streamlit as st


COLOR_TOKENS: dict[str, str] = {
    "bg_primary": "#09090B",
    "bg_elevated": "#121215",
    "bg_interactive": "#18181B",
    "border_subtle": "#27272A",
    "border_active": "#52525B",
    "text_primary": "#FAFAFA",
    "text_secondary": "#A1A1AA",
    "text_muted": "#71717A",
    "accent_emerald": "#10B981",
    "accent_blue": "#3B82F6",
    "accent_amber": "#F59E0B",
    "accent_red": "#EF4444",
}

STAGE_BREADCRUMBS: list[tuple[int, str]] = [
    (0, "00 AUTHENTICATION"),
    (1, "01 MODEL & DATA"),
    (2, "02 BENCHMARKING"),
    (3, "03 PARETO ENGINE"),
    (4, "04 EXPORT"),
]


def inject_minimalist_css() -> None:
    """Inject Linear/Vercel-inspired monochrome dark engineering CSS into Streamlit."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #09090B;
        --bg-elevated: #121215;
        --bg-interactive: #18181B;
        --border-subtle: #27272A;
        --border-active: #52525B;
        --text-primary: #FAFAFA;
        --text-secondary: #A1A1AA;
        --text-muted: #71717A;
        --accent-emerald: #10B981;
        --accent-blue: #3B82F6;
        --accent-amber: #F59E0B;
        --accent-red: #EF4444;
    }

    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header[data-testid="stHeader"] {background: transparent !important; height: 2rem !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}

    /* Base App Surface & Typography */
    .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    /* Sidebar Surface */
    section[data-testid="stSidebar"] {
        background-color: #0D0D10 !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* Headings */
    h1, h2, h3, h4, .pareto-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.025em !important;
        font-weight: 600 !important;
    }

    /* Monospace Telemetry Utility */
    .mono-telemetry, code, pre {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* Minimalist Surface Card */
    .pareto-card {
        background-color: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 20px;
        box-shadow: none !important;
        margin-bottom: 16px;
    }

    .pareto-card-interactive {
        background-color: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 20px;
        box-shadow: none !important;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }

    .pareto-card-interactive:hover {
        border-color: var(--border-active);
        background-color: var(--bg-interactive);
    }

    /* Typographic Badges */
    .badge-optimal {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        letter-spacing: 0.04em;
    }

    .badge-baseline {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: rgba(59, 130, 246, 0.12);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.35);
        letter-spacing: 0.04em;
    }

    .badge-warning {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.35);
        letter-spacing: 0.04em;
    }

    .badge-dominated {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: rgba(113, 113, 122, 0.12);
        color: #A1A1AA;
        border: 1px solid #27272A;
        letter-spacing: 0.04em;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background-color: var(--bg-interactive) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.1rem !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent-emerald) !important;
        background-color: rgba(16, 185, 129, 0.10) !important;
        color: #10B981 !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #10B981 !important;
        color: #09090B !important;
        border: 1px solid #10B981 !important;
        font-weight: 600 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #059669 !important;
        color: #FAFAFA !important;
        border-color: #059669 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        color: var(--text-secondary) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent-emerald) !important;
    }

    /* Inputs & Selectboxes */
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: var(--bg-primary) !important;
        border-color: var(--border-subtle) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }

    /* Checkboxes: Override Streamlit default red (#FF4B4B) with Emerald (#10B981) */
    span[data-baseweb="checkbox"] {
        border-radius: 4px !important;
    }
    label[data-baseweb="checkbox"] > span:first-child {
        background-color: #10B981 !important;
        border-color: #10B981 !important;
    }
    label[data-baseweb="checkbox"] input:not(:checked) + span {
        background-color: #18181B !important;
        border: 1px solid #3F3F46 !important;
    }

    /* Sliders: Override Streamlit default red (#FF4B4B) with Emerald (#10B981) */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #10B981 !important;
        border-color: #10B981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.20) !important;
    }
    div[data-testid="stThumbValue"], div[data-baseweb="slider"] div[data-testid="stTickBarMin"], div[data-baseweb="slider"] div[data-testid="stTickBarMax"] {
        color: #10B981 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="slider"] > div > div > div:first-child {
        background-color: #10B981 !important;
    }

    /* Bordered Streamlit containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #121215 !important;
        border: 1px solid #27272A !important;
        border-radius: 6px !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_breadcrumb_nav(active_view: int, auth_user: dict[str, Any] | None = None) -> None:
    """Render minimalist top navigation breadcrumb bar showing active state-machine stage."""
    parts: list[str] = []
    for stage_idx, label in STAGE_BREADCRUMBS:
        if stage_idx == active_view:
            parts.append(
                f'<span style="color:#10B981;font-weight:600;border-bottom:2px solid #10B981;padding-bottom:4px;">'
                f"{label}</span>"
            )
        elif stage_idx < active_view:
            parts.append(f'<span style="color:#A1A1AA;">{label}</span>')
        else:
            parts.append(f'<span style="color:#52525B;">{label}</span>')

    sep = '<span style="color:#27272A;margin:0 12px;">/</span>'
    breadcrumb_html = sep.join(parts)

    user_badge_html = ""
    if auth_user:
        uid = auth_user.get("user_id", "engineer")
        role = auth_user.get("role", "Engineer")
        clearance = auth_user.get("clearance", "[VERIFIED]")
        user_badge_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.74rem;color:#A1A1AA;">'
            f'<span style="color:#10B981;margin-right:8px;">{clearance}</span>'
            f'<span style="color:#FAFAFA;">{uid}</span> '
            f'<span style="color:#71717A;">({role})</span>'
            f"</div>"
        )
    else:
        user_badge_html = (
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.74rem;color:#71717A;">'
            "SESSION STATUS: [UNAUTHENTICATED]"
            "</div>"
        )

    bar_html = f"""
    <div style="
        background-color:#121215;
        border:1px solid #27272A;
        border-radius:6px;
        padding:12px 20px;
        margin-bottom:24px;
        display:flex;
        flex-wrap:wrap;
        justify-content:space-between;
        align-items:center;
        gap:12px;
    ">
        <div style="display:flex;align-items:center;gap:16px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;font-weight:700;color:#FAFAFA;letter-spacing:0.05em;">
                PARETO // BENCHMARK v1.0
            </span>
            <span style="color:#27272A;">|</span>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;letter-spacing:0.03em;">
                {breadcrumb_html}
            </div>
        </div>
        {user_badge_html}
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    primary_value: str,
    delta_text: str,
    accent_color: str = "#10B981",
) -> None:
    """Render a minimalist monospace KPI card with tabular-nums formatting."""
    html = f"""
    <div class="pareto-card" style="padding:18px 20px;margin-bottom:12px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.70rem;color:#71717A;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">
            {label}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.55rem;font-weight:700;color:#FAFAFA;font-variant-numeric:tabular-nums;margin-bottom:6px;">
            {primary_value}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:{accent_color};font-variant-numeric:tabular-nums;">
            {delta_text}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
