"""Scaling Law Verification Dashboard (2019–2024).

Interactive 6-tab Streamlit + Plotly dashboard evaluating whether AI scaling
laws hold across real-world models in Epoch AI's
`benchmarks_with_model_accessibility.csv` dataset (131 models, 24 raw columns).
"""
from __future__ import annotations

import html
import math
import sys
from pathlib import Path
from typing import Any, Callable

_APP_DIR = str(Path(__file__).parent.resolve())
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import numpy as np
import pandas as pd
import streamlit as st

import astryx_tokens as T
from charts import (
    RAW_COLUMN_GROUPS,
    RAW_COLUMN_META,
    _fmt_scale_value,
    build_dataset_column_sunburst,
    build_dataset_coverage_bar,
    build_direct_scaling_chart,
    build_multi_actual_vs_predicted_chart,
    build_open_closed_bar_chart,
    build_open_closed_coverage_bar,
    build_open_closed_pie_chart,
    build_qq_plot,
    build_residual_plot,
    build_time_performance_chart,
    build_time_residual_plot,
    build_time_scale_chart,
    compute_shared_perf_y_range,
    compute_shared_time_range,
)
from scaling_stats import (
    ACCESS_OPTIONS,
    BENCHMARK_OPTIONS,
    SCALE_META,
    SOURCE_REPO_URL,
    evaluate_line_table,
    fit_multi_ols,
    fit_simple_ols,
    format_ols_stat_line,
    format_p,
    format_p_short,
    load_scaling_data,
)

st.set_page_config(
    page_title="스케일링 법칙은 실제 AI 모델에서도 성립했는가?",
    layout="wide",
    initial_sidebar_state="auto",
)

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

st.html(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Montserrat:wght@600;700;800&display=swap');
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    /* Unified Flagship Astryx Design Tokens */
    :root {{
        --font-family-heading: {T.FONT_FAMILY_HEADING};
        --font-family-body: {T.FONT_FAMILY_BODY};
        --font-family-code: {T.FONT_FAMILY_CODE};

        --spacing-1: 0.25rem;
        --spacing-2: 0.5rem;
        --spacing-3: 0.75rem;
        --spacing-4: 1rem;
        --spacing-5: 1.25rem;
        --spacing-6: 1.5rem;

        --radius-inner: {T.RADIUS_INNER};
        --radius-element: {T.RADIUS_ELEMENT};
        --radius-container: {T.RADIUS_CONTAINER};
        --radius-page: {T.RADIUS_PAGE};
        --radius-full: {T.RADIUS_FULL};

        --color-bg-body: {T.COLOR_BG_BODY};
        --color-bg-sidebar: {T.COLOR_BG_SIDEBAR};
        --color-bg-card: {T.COLOR_BG_CARD};
        --color-bg-muted: {T.COLOR_BG_MUTED};
        --color-hero-start: {T.COLOR_BG_HERO_START};
        --color-hero-end: {T.COLOR_BG_HERO_END};
        --color-hero-tile: {T.COLOR_BG_HERO_TILE};

        --color-text-primary: {T.COLOR_TEXT_PRIMARY};
        --color-text-secondary: {T.COLOR_TEXT_SECONDARY};
        --color-text-muted: {T.COLOR_TEXT_MUTED};
        --color-on-dark: {T.COLOR_ON_DARK};
        --color-on-dark-muted: {T.COLOR_ON_DARK_MUTED};

        --color-accent: {T.COLOR_ACCENT};
        --color-hero-blue: {T.COLOR_HERO_BLUE};
        --color-hero-orange: {T.COLOR_HERO_ORANGE};
        --color-hero-green: {T.COLOR_HERO_GREEN};
        --color-border: {T.COLOR_BORDER};
        --color-border-emphasized: {T.COLOR_BORDER_EMPHASIZED};
        --color-border-hero: {T.COLOR_BORDER_HERO};

        --data-open: {T.DATA_OPEN};
        --data-closed: {T.DATA_CLOSED};

        --shadow-low: {T.SHADOW_LOW};
        --shadow-hero: {T.SHADOW_HERO};
        --duration-fast: {T.DURATION_FAST};
        --ease-standard: {T.EASE_STANDARD};
    }}

    .stApp,
    [data-testid="stSidebar"],
    .flagship-hero,
    .summary-side-card,
    .compare-strip,
    .y-axis-note,
    .panel-title,
    .sidebar-group-label,
    .footer-links {{
        font-family: var(--font-family-body);
    }}

    /* Cool-Slate Research Sidebar Rail */
    [data-testid="stSidebar"] {{
        border-right: 1px solid var(--color-border-emphasized);
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.25rem;
    }}
    .sidebar-group-label {{
        font-family: var(--font-family-heading);
        font-size: 0.70rem;
        font-weight: 700;
        color: var(--color-text-secondary);
        letter-spacing: 0.08em;
        margin: var(--spacing-2) 0 var(--spacing-1) 0;
        text-transform: uppercase;
    }}

    .block-container {{
        padding-top: 3.6rem;
        padding-bottom: 2.2rem;
        max-width: 1320px;
    }}

    /* ── Deep Midnight-Slate Executive Hero Banner (Title + Integrated 4 KPIs) ─ */
    .flagship-hero {{
        background: linear-gradient(135deg, var(--color-hero-start) 0%, var(--color-hero-end) 100%);
        color: var(--color-on-dark);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: var(--radius-page);
        padding: 1.12rem 1.45rem 1.15rem 1.45rem;
        box-shadow: var(--shadow-hero);
        margin: 0 0 0.90rem 0;
    }}
    .hero-top-bar {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem 1rem;
        margin-bottom: 0.50rem;
    }}
    .hero-eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: var(--font-family-code);
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        color: var(--color-hero-blue);
    }}
    .hero-eyebrow-dot {{
        width: 7px;
        height: 7px;
        border-radius: var(--radius-full);
        background: var(--color-hero-blue);
        display: inline-block;
        flex-shrink: 0;
    }}
    .hero-pills {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.38rem;
    }}
    .hero-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.32rem;
        padding: 0.18rem 0.62rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid var(--color-border-hero);
        border-radius: var(--radius-full);
        font-family: var(--font-family-code);
        font-size: 0.72rem;
        color: var(--color-on-dark-muted);
        font-variant-numeric: tabular-nums;
    }}
    .hero-pill strong {{
        color: var(--color-on-dark);
        font-weight: 600;
    }}
    .hero-dot-open {{
        width: 7px;
        height: 7px;
        border-radius: var(--radius-full);
        background: var(--color-hero-blue);
        display: inline-block;
    }}
    .hero-dot-closed {{
        width: 7px;
        height: 7px;
        border-radius: 2px;
        background: var(--color-hero-orange);
        display: inline-block;
    }}
    .flagship-hero h1.hero-title {{
        font-family: var(--font-family-heading) !important;
        font-size: clamp(1.26rem, 1.85vw, 1.66rem) !important;
        font-weight: 800 !important;
        line-height: 1.30 !important;
        letter-spacing: -0.024em;
        word-break: keep-all;
        color: var(--color-on-dark) !important;
        padding: 0 !important;
        margin: 0 0 0.90rem 0 !important;
    }}
    .hero-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.72rem;
    }}
    @media (max-width: 960px) {{
        .hero-kpi-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}
    .hero-kpi-tile {{
        background: var(--color-hero-tile);
        border: 1px solid var(--color-border-hero);
        border-top: 2px solid var(--tile-accent, var(--color-hero-blue));
        border-radius: var(--radius-container);
        padding: 0.62rem 0.90rem 0.60rem 0.90rem;
        min-width: 0;
        transition: background-color var(--duration-fast) var(--ease-standard);
    }}
    .hero-kpi-tile:hover {{
        background: rgba(255, 255, 255, 0.095);
    }}
    .hero-kpi-label {{
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--color-on-dark-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 0.22rem;
    }}
    .hero-kpi-value {{
        font-family: var(--font-family-heading);
        font-size: 1.62rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--color-on-dark);
        line-height: 1.18;
        font-variant-numeric: tabular-nums;
        margin: 0;
    }}

    /* ── 6-Tab Presentation Navigation Rail (`st.tabs`) ─────────────────────── */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background: var(--color-bg-muted);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-container);
        padding: 4px;
        gap: 3px;
        margin-bottom: 0.85rem;
    }}
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {{
        display: none;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] {{
        flex: 1 1 auto;
        justify-content: center;
        border-radius: var(--radius-element);
        padding: 0.45rem 0.42rem;
        background: transparent;
        color: var(--color-text-secondary);
        font-family: var(--font-family-heading);
        font-weight: 600;
        font-size: 0.79rem;
        transition: background-color var(--duration-fast) var(--ease-standard),
                    color var(--duration-fast) var(--ease-standard),
                    box-shadow var(--duration-fast) var(--ease-standard);
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {{
        font-size: 0.79rem;
        font-weight: 600;
        margin: 0;
        white-space: nowrap;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
        color: var(--color-text-primary);
        background: rgba(5, 54, 89, 0.055);
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
        background: var(--color-hero-end);
        color: var(--color-on-dark);
        box-shadow: 0 2px 6px rgba(11, 21, 36, 0.20);
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {{
        color: var(--color-on-dark);
        font-weight: 700;
    }}
    @media (max-width: 720px) {{
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 4px;
        }}
        div[data-testid="stTabs"] button[data-baseweb="tab"] {{
            width: 100%;
            padding: 0.44rem 0.50rem;
        }}
    }}

    /* ── Elevated White Astryx Analytical Cards on #F1F4F7 Canvas ──────────── */
    div[class*="st-key-card_"] {{
        background: var(--color-bg-card);
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-container) !important;
        box-shadow: var(--shadow-low);
        padding: var(--spacing-4) var(--spacing-5) !important;
    }}
    div[data-testid="stExpander"] > details {{
        background: var(--color-bg-card);
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-container) !important;
        box-shadow: var(--shadow-low);
        overflow: hidden;
    }}
    div[data-testid="stExpander"] > details > summary {{
        background: var(--color-bg-card);
        color: var(--color-text-primary);
        font-weight: 600;
    }}
    div[data-testid="stTabs"] div[data-baseweb="select"] > div {{
        background-color: var(--color-bg-body);
        border-color: var(--color-border-emphasized);
    }}

    .y-axis-note {{
        display: inline-block;
        font-size: 0.78rem;
        color: var(--color-text-secondary);
        background: var(--color-bg-muted);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-element);
        padding: 0.30rem 0.72rem;
        margin: 0 0 0.75rem 0;
        line-height: 1.4;
    }}
    .panel-title {{
        font-family: var(--font-family-heading);
        font-size: 0.99rem;
        font-weight: 700;
        letter-spacing: -0.014em;
        color: var(--color-text-primary);
        margin: 0 0 0.22rem 0;
        line-height: 1.38;
    }}
    .panel-stat {{
        font-size: 0.79rem;
        font-family: var(--font-family-code);
        font-variant-numeric: tabular-nums;
        color: var(--color-text-secondary);
        border-bottom: 1px solid var(--color-border);
        padding: 0.14rem 0 0.48rem 0;
        margin: 0 0 var(--spacing-2) 0;
        line-height: 1.42;
    }}
    .sunburst-legend {{
        display: flex;
        flex-direction: column;
        gap: 0.30rem;
        padding: 0.48rem 0.68rem;
        margin: 0.25rem 0 0.35rem 0;
        background: var(--color-bg-body);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-element);
        font-size: 0.75rem;
        color: var(--color-text-secondary);
    }}
    .sunburst-legend-row {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.35rem 0.75rem;
    }}
    .sunburst-legend-tag {{
        font-family: var(--font-family-code);
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--color-text-muted);
        min-width: 78px;
        flex-shrink: 0;
    }}
    .sunburst-legend-item {{
        display: inline-flex;
        align-items: center;
        gap: 0.32rem;
        color: var(--color-text-primary);
        font-weight: 500;
        white-space: nowrap;
    }}
    .sunburst-swatch {{
        width: 10px;
        height: 10px;
        border-radius: 3px;
        display: inline-block;
        flex-shrink: 0;
    }}
    div[class*="st-key-card_"] div[data-testid="stExpander"] > details {{
        background: var(--color-bg-body);
        box-shadow: none;
        border-radius: var(--radius-element) !important;
        margin-top: 0.35rem;
    }}
    div[class*="st-key-card_"] div[data-testid="stExpander"] > details > summary {{
        background: var(--color-bg-body);
        font-size: 0.80rem;
        padding: 0.42rem 0.72rem;
    }}
    .col-guide-intro {{
        font-size: 0.77rem;
        color: var(--color-text-secondary);
        margin: 0 0 0.50rem 0;
        line-height: 1.48;
    }}
    .col-guide-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.76rem;
        line-height: 1.45;
    }}
    .col-guide-table th {{
        text-align: left;
        font-family: var(--font-family-heading);
        font-weight: 700;
        font-size: 0.72rem;
        color: var(--color-text-secondary);
        border-bottom: 1px solid var(--color-border-emphasized);
        padding: 0.36rem 0.48rem;
        white-space: nowrap;
    }}
    .col-guide-table td {{
        padding: 0.40rem 0.48rem;
        border-bottom: 1px solid var(--color-border);
        color: var(--color-text-primary);
        vertical-align: top;
    }}
    .col-guide-table tr:last-child td {{
        border-bottom: none;
    }}
    .col-guide-role {{
        display: inline-flex;
        align-items: center;
        gap: 0.34rem;
        font-weight: 600;
        white-space: nowrap;
    }}
    .col-guide-cnt {{
        font-family: var(--font-family-code);
        font-weight: 600;
        text-align: center;
        white-space: nowrap;
        color: var(--color-text-secondary);
    }}
    .col-guide-cols {{
        font-family: var(--font-family-code);
        font-size: 0.71rem;
        color: var(--color-text-secondary);
        word-break: break-word;
    }}

    /* ── Asymmetric Plot Summary Side Card (Pure Data Metrics) ─────────────── */
    .summary-side-card {{
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-container);
        box-shadow: var(--shadow-low);
        padding: 0.95rem 1.0rem;
    }}
    .side-kicker {{
        font-family: var(--font-family-code);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--color-text-muted);
        white-space: nowrap;
        margin-bottom: 0.60rem;
    }}
    .ratio-track {{
        display: flex;
        width: 100%;
        height: 10px;
        border-radius: var(--radius-full);
        overflow: hidden;
        background: var(--color-bg-muted);
        margin-bottom: 0.32rem;
    }}
    .ratio-caption {{
        display: flex;
        justify-content: space-between;
        font-family: var(--font-family-code);
        font-size: 0.71rem;
        color: var(--color-text-secondary);
        font-variant-numeric: tabular-nums;
        margin-bottom: 0.50rem;
    }}
    .ratio-open {{
        background: var(--data-open);
        height: 100%;
    }}
    .ratio-closed {{
        background: var(--data-closed);
        height: 100%;
    }}
    .side-stat-row {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 0.42rem 0;
        border-top: 1px solid var(--color-border);
        font-size: 0.79rem;
        gap: 0.35rem;
    }}
    .side-stat-k {{
        color: var(--color-text-secondary);
        font-weight: 500;
        word-break: keep-all;
        white-space: nowrap;
    }}
    .side-stat-v {{
        font-family: var(--font-family-code);
        font-weight: 700;
        color: var(--color-text-primary);
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        text-align: right;
    }}

    /* ── Tab 3 Side-by-Side Model A vs Model B Metric Comparison Strip ─────── */
    .compare-strip {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.25rem 0 0.75rem 0;
    }}
    @media (max-width: 720px) {{
        .compare-strip {{
            grid-template-columns: 1fr;
        }}
    }}
    .compare-box {{
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-left: 4px solid var(--box-accent, var(--data-open));
        border-radius: var(--radius-container);
        padding: 0.75rem 1.0rem;
        box-shadow: var(--shadow-low);
    }}
    .compare-head {{
        font-family: var(--font-family-heading);
        font-size: 0.84rem;
        font-weight: 700;
        color: var(--color-text-primary);
        margin-bottom: 0.35rem;
    }}
    .compare-metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem 1.15rem;
        font-family: var(--font-family-code);
        font-size: 0.80rem;
        color: var(--color-text-secondary);
        font-variant-numeric: tabular-nums;
    }}
    .compare-metrics strong {{
        color: var(--color-text-primary);
    }}
    .vif-warn-inline {{
        background: {T.COLOR_WARN_BG};
        color: {T.COLOR_WARN_TEXT};
        padding: 0.08rem 0.42rem;
        border-radius: var(--radius-inner);
        font-weight: 700;
    }}
    .vif-ok-inline {{
        background: {T.COLOR_OK_BG};
        color: {T.COLOR_OK_TEXT};
        padding: 0.08rem 0.42rem;
        border-radius: var(--radius-inner);
        font-weight: 700;
    }}

    /* ── Astryx Neutral Concentric Segmented Radio Control (10px outer / 6px inner) ─ */
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: inline-flex;
        flex-wrap: wrap;
        width: fit-content;
        max-width: 100%;
        min-height: auto;
        background: var(--color-bg-muted);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-element);
        padding: var(--spacing-1);
        gap: 3px;
        margin: var(--spacing-1) 0 var(--spacing-2) 0;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] {{
        display: inline-flex;
        align-items: center;
        background: transparent;
        border: none;
        border-radius: var(--radius-inner);
        padding: 0.26rem 0.60rem;
        margin: 0;
        cursor: pointer;
        transition: background-color var(--duration-fast) var(--ease-standard),
                    box-shadow var(--duration-fast) var(--ease-standard),
                    color var(--duration-fast) var(--ease-standard);
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child:has(+ input[type="radio"]) {{
        display: none;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div:last-child,
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p {{
        color: var(--color-text-secondary);
        font-weight: 500;
        margin: 0;
        font-size: 0.80rem;
        line-height: 1.35;
        white-space: nowrap;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"]:hover:not(:has(input:checked)) {{
        background: rgba(5, 54, 89, 0.045);
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
        background: var(--color-bg-card);
        box-shadow: 0 1px 2px rgba(5, 54, 89, 0.11);
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) > div:last-child,
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) div[data-testid="stMarkdownContainer"] p {{
        color: var(--color-text-primary);
        font-weight: 700;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:focus-visible) {{
        outline: 2px solid var(--color-accent);
        outline-offset: 2px;
    }}
    @media (max-width: 640px) {{
        .flagship-hero {{
            padding: 1.05rem 1.05rem;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] {{
            flex-wrap: wrap;
            width: 100%;
        }}
        div[class*="st-key-card_"] {{
            padding: var(--spacing-3) var(--spacing-3) !important;
        }}
    }}

    .footer-links {{
        font-size: 0.82rem;
        color: var(--color-text-secondary);
        text-align: left;
        padding-top: var(--spacing-1);
    }}
    .footer-links a {{
        color: var(--color-text-primary);
        text-decoration: underline dotted var(--color-text-muted);
        text-underline-offset: 3px;
    }}
    .footer-links a:hover {{
        color: var(--color-accent);
    }}
    </style>
    """
)


def _fmt_vif(v: Any) -> str:
    if pd.isna(v):
        return "-"
    val = float(v)
    if val >= 1e4:
        return f"{val:.2e} (!)"
    if val >= 5.0:
        return f"{val:.2f} (!)"
    return f"{val:.2f}"


SIMPLE_FMT: dict[str, Any] = {
    "b1 (10x)": "{:+.4f}",
    "SE": "{:.4f}",
    "t": "{:.2f}",
    "z": "{:.2f}",
    "R²": "{:.3f}",
    "RSE": "{:.3f}",
}

MULTI_FMT: dict[str, Any] = {
    "coef": "{:.4f}",
    "SE": "{:.4f}",
    "t": "{:.2f}",
    "z": "{:.2f}",
    "VIF": _fmt_vif,
}

HC3_FMT: dict[str, Any] = {
    "coef": "{:.4f}",
    "SE": "{:.4f}",
    "t / z": "{:.2f}",
}


def _style_derived_table(
    df: pd.DataFrame,
    *,
    fmt_map: dict[str, Any] | None = None,
    warn_col: str | None = None,
    warn_predicate: Callable[[Any], bool] | None = None,
    ok_predicate: Callable[[Any], bool] | None = None,
):
    styler = df.style
    if warn_col and warn_col in df.columns and (warn_predicate or ok_predicate):
        def _cell_style(v: Any) -> str:
            if warn_predicate and warn_predicate(v):
                return f"background-color: {T.COLOR_WARN_BG}; color: {T.COLOR_WARN_TEXT}; font-weight: 700;"
            if ok_predicate and ok_predicate(v):
                return f"background-color: {T.COLOR_OK_BG}; color: {T.COLOR_OK_TEXT}; font-weight: 700;"
            return ""

        styler = styler.map(_cell_style, subset=[warn_col])
    if fmt_map:
        active_fmt = {col: fn for col, fn in fmt_map.items() if col in df.columns}
        if active_fmt:
            styler = styler.format(active_fmt, na_rep="-")
    return styler


def _render_hero_banner(
    yr_span_txt: str,
    bench_choice: str,
    open_n: int,
    closed_n: int,
    kpi_items: list[dict[str, str]],
) -> None:
    yr_esc = html.escape(yr_span_txt, quote=True)
    b_esc = html.escape(bench_choice, quote=True)
    tiles = []
    for item in kpi_items:
        lbl = html.escape(str(item["label"]), quote=True)
        val = html.escape(str(item["value"]), quote=True)
        acc = html.escape(str(item.get("accent", T.COLOR_HERO_BLUE)), quote=True)
        tiles.append(
            f"<div class='hero-kpi-tile' style='--tile-accent:{acc};'>"
            f"<div class='hero-kpi-label'>{lbl}</div>"
            f"<div class='hero-kpi-value'>{val}</div>"
            f"</div>"
        )
    st.markdown(
        f"<section class='flagship-hero'>"
        f"<div class='hero-top-bar'>"
        f"<div class='hero-eyebrow'><span class='hero-eyebrow-dot'></span>"
        f"<span>EPOCH AI · benchmarks_with_model_accessibility.csv</span></div>"
        f"<div class='hero-pills'>"
        f"<span class='hero-pill'>기간 <strong>{yr_esc}</strong></span>"
        f"<span class='hero-pill'><strong>{b_esc}</strong> 0~1</span>"
        f"<span class='hero-pill'><span class='hero-dot-open'></span><strong>{open_n}</strong>"
        f"&nbsp;/&nbsp;<span class='hero-dot-closed'></span><strong>{closed_n}</strong></span>"
        f"</div>"
        f"</div>"
        f"<h1 class='hero-title'>스케일링 법칙은 실제 AI 모델에서도 성립했는가? "
        f"<span style='white-space:nowrap;'>({yr_esc})</span></h1>"
        f"<div class='hero-kpi-grid'>{''.join(tiles)}</div>"
        f"</section>",
        unsafe_allow_html=True,
    )


def _render_summary_card(
    kicker: str,
    left_pct: float,
    right_pct: float,
    left_cap: str,
    right_cap: str,
    rows: list[tuple[str, str]],
    badge_html: str | None = None,
) -> None:
    """Render a consistent Astryx right-side Distribution Summary card next to a plot."""
    k_esc = html.escape(kicker, quote=True)
    lc_esc = html.escape(left_cap, quote=True)
    rc_esc = html.escape(right_cap, quote=True)
    row_html = []
    for k_txt, v_txt in rows:
        row_html.append(
            f"<div class='side-stat-row'>"
            f"<span class='side-stat-k'>{html.escape(k_txt, quote=True)}</span>"
            f"<span class='side-stat-v'>{html.escape(v_txt, quote=True)}</span>"
            f"</div>"
        )
    if badge_html:
        row_html.append(badge_html)
    st.markdown(
        f"<div class='summary-side-card'>"
        f"<div class='side-kicker'>{k_esc}</div>"
        f"<div class='ratio-track'>"
        f"<div class='ratio-open' style='width:{left_pct:.1f}%;'></div>"
        f"<div class='ratio-closed' style='width:{right_pct:.1f}%;'></div>"
        f"</div>"
        f"<div class='ratio-caption'><span>{lc_esc}</span><span>{rc_esc}</span></div>"
        f"{''.join(row_html)}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _format_simple_table(items: list[tuple[str, dict]], use_hc3: bool = False) -> pd.DataFrame:
    stat_col = "z" if use_hc3 else "t"
    rows = []
    for label, reg in items:
        if reg.get("ok"):
            rows.append(
                {
                    "규모 변수 (개별 OLS)": label,
                    "n": int(reg["n"]),
                    "b1 (10x)": round(reg["b1"], 4),
                    "SE": round(reg["se"], 4),
                    stat_col: round(reg["t"], 2) if pd.notna(reg["t"]) else None,
                    "p": format_p_short(reg["p"]),
                    "95% CI": f"{reg['ci_low']:+.3f} ~ {reg['ci_high']:+.3f}",
                    "R²": round(reg["r2"], 3),
                    "RSE": round(reg["rse"], 3),
                }
            )
        else:
            rows.append(
                {
                    "규모 변수 (개별 OLS)": label,
                    "n": int(reg.get("n", 0)),
                    "b1 (10x)": None,
                    "SE": None,
                    stat_col: None,
                    "p": "-",
                    "95% CI": str(reg.get("reason", "-")),
                    "R²": None,
                    "RSE": None,
                }
            )
    return pd.DataFrame(rows)


def _format_multi_table(res: dict) -> pd.DataFrame:
    stat_col = "z" if res.get("use_hc3") else "t"
    rows = []
    for t in res["terms"]:
        vif_val = t["vif"]
        rows.append(
            {
                "Variable": t["label"],
                "coef": round(t["coef"], 4),
                "SE": round(t["se"], 4),
                stat_col: round(t["t"], 2) if pd.notna(t["t"]) else None,
                "p": format_p_short(t["p"]),
                "95% CI": f"{t['ci_low']:+.3f} ~ {t['ci_high']:+.3f}",
                "VIF": round(vif_val, 2) if pd.notna(vif_val) else None,
            }
        )
    return pd.DataFrame(rows)


def _format_hc3_table(reg: dict) -> pd.DataFrame:
    rows = []
    if "se_std" in reg:
        var_name = reg.get("var_name", reg.get("x_col", "x"))
        rows.append(
            {
                "Variable": var_name,
                "Method": "OLS",
                "coef": round(reg["b1"], 4),
                "SE": round(reg["se_std"], 4),
                "t / z": round(reg["t_std"], 2) if pd.notna(reg["t_std"]) else None,
                "p": format_p_short(reg["p_std"]),
                "95% CI": f"{reg['ci_std'][0]:+.3f} ~ {reg['ci_std'][1]:+.3f}",
            }
        )
        rows.append(
            {
                "Variable": var_name,
                "Method": "HC3",
                "coef": round(reg["b1"], 4),
                "SE": round(reg["se_hc3"], 4),
                "t / z": round(reg["t_hc3"], 2) if pd.notna(reg["t_hc3"]) else None,
                "p": format_p_short(reg["p_hc3"]),
                "95% CI": f"{reg['ci_hc3'][0]:+.3f} ~ {reg['ci_hc3'][1]:+.3f}",
            }
        )
    elif "terms" in reg:
        for t in reg["terms"]:
            rows.append(
                {
                    "Variable": t["label"],
                    "Method": "OLS",
                    "coef": round(t["coef"], 4),
                    "SE": round(t["se_std"], 4),
                    "t / z": round(t["t_std"], 2) if pd.notna(t["t_std"]) else None,
                    "p": format_p_short(t["p_std"]),
                    "95% CI": f"{t['ci_std'][0]:+.3f} ~ {t['ci_std'][1]:+.3f}",
                }
            )
            rows.append(
                {
                    "Variable": t["label"],
                    "Method": "HC3",
                    "coef": round(t["coef"], 4),
                    "SE": round(t["se_hc3"], 4),
                    "t / z": round(t["t_hc3"], 2) if pd.notna(t["t_hc3"]) else None,
                    "p": format_p_short(t["p_hc3"]),
                    "95% CI": f"{t['ci_hc3'][0]:+.3f} ~ {t['ci_hc3'][1]:+.3f}",
                }
            )
    return pd.DataFrame(rows)


def _format_multi_header(res: dict) -> str:
    se_tag = " · HC3" if res.get("use_hc3") else ""
    return (
        f"F={res['f_stat']:.2f} · {format_p(res['f_pval'])} · "
        f"R²={res['r2']:.3f} · Adj. R²={res['adj_r2']:.3f} · "
        f"RSE={res['rse']:.3f} · n={res['n']}{se_tag}"
    )


def main() -> None:
    try:
        data, baselines, raw_cols = load_scaling_data()
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as exc:
        st.error(f"Data load error: {exc}")
        st.stop()

    # ── Sidebar Controls (Cool-Slate Research Rail, 2 Visual Groups) ─────────
    st.sidebar.header("필터 및 회귀 설정")
    st.sidebar.markdown("<div class='sidebar-group-label'>데이터 필터</div>", unsafe_allow_html=True)
    bench_choice = st.sidebar.selectbox(
        "Benchmark",
        options=BENCHMARK_OPTIONS,
    )
    random_line = baselines.get(bench_choice, 0.25)

    min_yr = int(data["_year"].min())
    max_yr = int(data["_year"].max())
    yr_range = st.sidebar.slider(
        "Date",
        min_value=min_yr,
        max_value=max_yr,
        value=(min_yr, max_yr),
    )

    selected_access = st.sidebar.multiselect(
        "Open/Closed",
        options=ACCESS_OPTIONS,
        default=ACCESS_OPTIONS,
    )

    st.sidebar.divider()
    st.sidebar.markdown("<div class='sidebar-group-label'>표시 및 회귀 설정</div>", unsafe_allow_html=True)
    show_summary = st.sidebar.checkbox("Plot Summary 표시", value=True)
    show_reg = st.sidebar.checkbox("OLS 회귀선 + 95% 신뢰구간", value=True)
    show_pi = st.sidebar.checkbox("95% 예측구간", value=False, disabled=not show_reg)
    use_hc3 = st.sidebar.checkbox("HC3 강건 표준오차", value=False, disabled=not show_reg)

    # Single filtered dataframe `view` used by EVERY KPI, chart, regression, table & CSV
    view = data.loc[
        data["_year"].between(yr_range[0], yr_range[1])
        & data["_access"].isin(selected_access)
    ].copy()

    if view.empty:
        st.warning("선택한 조건에 해당하는 모델이 없습니다.")
        st.stop()

    yr_span_txt = f"{yr_range[0]}" if yr_range[0] == yr_range[1] else f"{yr_range[0]}–{yr_range[1]}"
    open_n = int((view["_access"] == "Open").sum())
    closed_n = int((view["_access"] == "Closed").sum())

    # ── Fit Weeks 6-7 OLS / Multiple Regression Models on `view` ─────────────
    reg_time_perf = fit_simple_ols(view, "year_frac", bench_choice, use_hc3=use_hc3)
    reg_time_flop = fit_simple_ols(view, "year_frac", "log10_flop", use_hc3=use_hc3)
    reg_time_param = fit_simple_ols(view, "year_frac", "log10_params", use_hc3=use_hc3)
    reg_time_data = fit_simple_ols(view, "year_frac", "log10_dataset", use_hc3=use_hc3)

    reg_dir_flop = fit_simple_ols(view, "log10_flop", bench_choice, use_hc3=use_hc3)
    reg_dir_param = fit_simple_ols(view, "log10_params", bench_choice, use_hc3=use_hc3)
    reg_dir_total = fit_simple_ols(view, "log10_total_params", bench_choice, use_hc3=use_hc3)
    reg_dir_data = fit_simple_ols(view, "log10_dataset", bench_choice, use_hc3=use_hc3)

    reg_multi_2 = fit_multi_ols(view, ["log10_params", "log10_dataset"], bench_choice, use_hc3=use_hc3)
    reg_multi_3 = fit_multi_ols(
        view, ["log10_flop", "log10_params", "log10_dataset"], bench_choice, use_hc3=use_hc3
    )

    # ── Build Minimalist Executive Hero Banner + 4 Clean KPI Tiles ───────────
    perf_valid = pd.to_numeric(view[bench_choice], errors="coerce").dropna()
    best_val = f"{perf_valid.max():.3f}" if not perf_valid.empty else "-"
    med_val = f"{perf_valid.median():.3f}" if not perf_valid.empty else "-"
    r2_val = (
        f"{reg_dir_flop['r2']:.3f}"
        if (show_reg and reg_dir_flop.get("ok"))
        else "-"
    )
    kpi_items: list[dict[str, str]] = [
        {
            "label": "모델 수",
            "value": f"{len(view)}개",
            "accent": T.COLOR_ON_DARK_MUTED,
        },
        {
            "label": f"{bench_choice} 최고",
            "value": best_val,
            "accent": T.COLOR_HERO_BLUE,
        },
        {
            "label": f"{bench_choice} 중앙값",
            "value": med_val,
            "accent": T.COLOR_HERO_GREEN,
        },
        {
            "label": "연산량 설명력 (R²)",
            "value": r2_val,
            "accent": T.COLOR_HERO_ORANGE if r2_val != "-" else T.DATA_NEUTRAL,
        },
    ]

    _render_hero_banner(
        yr_span_txt=yr_span_txt,
        bench_choice=bench_choice,
        open_n=open_n,
        closed_n=closed_n,
        kpi_items=kpi_items,
    )

    scale_label_to_key = {
        SCALE_META[k]["raw_col"]: k for k in ["log10_flop", "log10_params", "log10_dataset"]
    }
    scale_options = list(scale_label_to_key.keys())

    # ── 6-Tab Focused Presentation Structure (Raw Data -> Time -> Scaling -> Multi -> Diag) ─
    tab_raw, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "01 · 원자료",
            "02 · 시간 추이",
            "03 · Open / Closed",
            "04 · 스케일링 법칙",
            "05 · 다중 회귀",
            "06 · 회귀 진단",
        ]
    )

    # ── TAB 1 (NEW): Raw Dataset Column Sunburst, Coverage & Raw CSV Table ───
    with tab_raw:
        raw_note = (
            "※ 24개 원자료 컬럼을 5개 역할로 분류했습니다. "
            "휠 조각을 클릭하면 해당 컬럼의 Open / Closed / 결측 비율로 확대됩니다."
        )
        st.markdown(
            f"<div class='y-axis-note'>{html.escape(raw_note, quote=True)}</div>",
            unsafe_allow_html=True,
        )
        raw_view_opts = [
            "컬럼 계층 휠 (Sunburst)",
            "유효 관측률 (Bar)",
            "Open / Closed 공개율 (Bar)",
        ]
        if st.session_state.get("sec0_raw_toggle") not in raw_view_opts:
            st.session_state["sec0_raw_toggle"] = raw_view_opts[0]

        if show_summary:
            col_t0_chart, col_t0_side = st.columns([2.7, 1.3], gap="medium")
        else:
            col_t0_chart, col_t0_side = st.container(), None

        with col_t0_chart:
            with st.container(border=True, key="card_sec0_raw"):
                cur_t0 = st.session_state.get("sec0_raw_toggle", raw_view_opts[0])
                if cur_t0 == raw_view_opts[0]:
                    t0_title = f"원자료 컬럼 구조 · {len(raw_cols)}개 컬럼 (n={len(view)})"
                elif cur_t0 == raw_view_opts[1]:
                    t0_title = f"컬럼별 유효 관측률 · {len(raw_cols)}개 컬럼 (n={len(view)})"
                else:
                    t0_title = f"Open / Closed 주요 컬럼 공개율 (n={len(view)})"
                st.markdown(
                    f"<div class='panel-title'>{html.escape(t0_title, quote=True)}</div>",
                    unsafe_allow_html=True,
                )
                raw_view_choice = st.radio(
                    "Raw Data View Toggle",
                    options=raw_view_opts,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="sec0_raw_toggle",
                )
                if raw_view_choice == raw_view_opts[0]:
                    role_items_html = "".join(
                        f"<span class='sunburst-legend-item'>"
                        f"<span class='sunburst-swatch' style='background:{g['color']};'></span>"
                        f"{html.escape(g['label'])}</span>"
                        for g in RAW_COLUMN_GROUPS
                    )
                    drill_items = [
                        ("Open 유효", T.DATA_OPEN),
                        ("Closed 유효", T.DATA_CLOSED),
                        ("결측 (NaN)", "#CBD5E1"),
                    ]
                    drill_items_html = "".join(
                        f"<span class='sunburst-legend-item'>"
                        f"<span class='sunburst-swatch' style='background:{clr};'></span>"
                        f"{html.escape(lbl)}</span>"
                        for lbl, clr in drill_items
                    )
                    st.markdown(
                        f"<div class='sunburst-legend'>"
                        f"<div class='sunburst-legend-row'>"
                        f"<span class='sunburst-legend-tag'>역할 (안쪽 링)</span>{role_items_html}"
                        f"</div>"
                        f"<div class='sunburst-legend-row'>"
                        f"<span class='sunburst-legend-tag'>클릭 확대</span>{drill_items_html}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    fig_raw = build_dataset_column_sunburst(view, raw_cols)
                elif raw_view_choice == raw_view_opts[1]:
                    fig_raw = build_dataset_coverage_bar(view, raw_cols)
                else:
                    fig_raw = build_open_closed_coverage_bar(view)
                st.plotly_chart(fig_raw, width="stretch", theme=None, config=PLOTLY_CONFIG)
                if raw_view_choice == raw_view_opts[0]:
                    with st.expander("컬럼 유형 분류 기준 보기 (5개 역할 · 24개 컬럼)", expanded=False):
                        total_v = max(len(view), 1)
                        guide_rows_html = []
                        for g in RAW_COLUMN_GROUPS:
                            gcols = [
                                c
                                for c in raw_cols
                                if RAW_COLUMN_META.get(c, {}).get("group", "notes") == g["key"]
                            ]
                            gcols_sorted = sorted(
                                gcols,
                                key=lambda c: (
                                    -int(
                                        (
                                            view[c].notna()
                                            & view[c].astype(str).str.strip().ne("")
                                        ).sum()
                                    )
                                    if c in view.columns
                                    else 0,
                                    c,
                                ),
                            )
                            col_tags = []
                            for c in gcols_sorted:
                                if c in view.columns:
                                    nv = int((view[c].notna() & view[c].astype(str).str.strip().ne("")).sum())
                                    pct_c = 100.0 * nv / total_v
                                    col_tags.append(f"{c} ({pct_c:.0f}%)")
                                else:
                                    col_tags.append(c)
                            guide_rows_html.append(
                                f"<tr>"
                                f"<td><span class='col-guide-role'>"
                                f"<span class='sunburst-swatch' style='background:{g['color']};'></span>"
                                f"{html.escape(g['label'])}</span></td>"
                                f"<td class='col-guide-cnt'>{len(gcols_sorted)}</td>"
                                f"<td>{html.escape(g['criteria'])}</td>"
                                f"<td class='col-guide-cols'>{html.escape(', '.join(col_tags))}</td>"
                                f"</tr>"
                            )
                        st.markdown(
                            "<div class='col-guide-intro'>"
                            "원본 24개 컬럼을 <strong>① 회귀분석에서의 통계적 역할</strong>(메타 · 규모 X · 벤치마크 Y · 텍스트 주석)으로 분류하고, "
                            "벤치마크 8종은 <strong>② 유효 관측률 50% 기준</strong>으로 채택(MMLU·BBH)과 미채택(GPQA·Elo·SEAL·HELM)으로 구분했습니다."
                            "</div>"
                            "<table class='col-guide-table'>"
                            "<thead><tr><th>컬럼 유형</th><th>개수</th><th>분류 기준</th><th>포함 컬럼 (현재 유효율)</th></tr></thead>"
                            f"<tbody>{''.join(guide_rows_html)}</tbody>"
                            "</table>",
                            unsafe_allow_html=True,
                        )

        if col_t0_side is not None:
            with col_t0_side:
                denom_v = max(len(view), 1)
                if raw_view_choice == raw_view_opts[2]:
                    open_df = view[view["_access"] == "Open"]
                    closed_df = view[view["_access"] == "Closed"]
                    open_pct0 = 100.0 * len(open_df) / denom_v
                    closed_pct0 = 100.0 * len(closed_df) / denom_v

                    def _pair_rate(col_name: str) -> str:
                        o_r = (
                            f"{100.0 * open_df[col_name].notna().mean():.0f}%"
                            if len(open_df) > 0 and col_name in open_df.columns
                            else "-"
                        )
                        c_r = (
                            f"{100.0 * closed_df[col_name].notna().mean():.0f}%"
                            if len(closed_df) > 0 and col_name in closed_df.columns
                            else "-"
                        )
                        return f"{o_r} / {c_r}"

                    _render_summary_card(
                        kicker="PLOT SUMMARY · 공개율 (O / C)",
                        left_pct=open_pct0,
                        right_pct=closed_pct0,
                        left_cap=f"Open {len(open_df)} ({open_pct0:.0f}%)",
                        right_cap=f"Closed {len(closed_df)} ({closed_pct0:.0f}%)",
                        rows=[
                            ("Training compute", _pair_rate("Training compute (FLOP)")),
                            ("Active Parameters", _pair_rate("Active Parameters")),
                            ("Model size", _pair_rate("Model size (parameters)")),
                            ("Dataset size", _pair_rate("Dataset size")),
                            ("MMLU", _pair_rate("MMLU")),
                            ("BBH", _pair_rate("BBH")),
                        ],
                    )
                else:
                    scale_covs = [
                        100.0 * view[c].notna().mean()
                        for c in [
                            "Training compute (FLOP)",
                            "Active Parameters",
                            "Model size (parameters)",
                            "Dataset size",
                        ]
                        if c in view.columns
                    ]
                    min_s = min(scale_covs) if scale_covs else 0.0
                    max_s = max(scale_covs) if scale_covs else 0.0
                    mmlu_pct0 = 100.0 * view["MMLU"].notna().mean() if "MMLU" in view.columns else 0.0
                    bbh_pct0 = 100.0 * view["BBH"].notna().mean() if "BBH" in view.columns else 0.0
                    n_m_multi = int(
                        view[["log10_flop", "log10_params", "log10_dataset", "MMLU"]].dropna().shape[0]
                    )
                    n_b_multi = int(
                        view[["log10_flop", "log10_params", "log10_dataset", "BBH"]].dropna().shape[0]
                    )
                    used_cols_n = 9
                    other_cols_n = max(len(raw_cols) - used_cols_n, 0)
                    used_pct0 = 100.0 * used_cols_n / max(len(raw_cols), 1)
                    other_pct0 = 100.0 - used_pct0
                    _render_summary_card(
                        kicker="PLOT SUMMARY · 원자료",
                        left_pct=used_pct0,
                        right_pct=other_pct0,
                        left_cap=f"분석 활용 {used_cols_n}개 ({used_pct0:.0f}%)",
                        right_cap=f"기타·주석 {other_cols_n}개 ({other_pct0:.0f}%)",
                        rows=[
                            ("원본 / 현재 모델", f"132행 → {len(view)}종"),
                            ("분리 기준선", "Random chance (1행)"),
                            ("전체 컬럼 수", f"{len(raw_cols)}개 (5개 역할)"),
                            ("규모 변수 (X)", f"4개 ({min_s:.0f}~{max_s:.0f}%)"),
                            ("MMLU / BBH 유효", f"{mmlu_pct0:.0f}% / {bbh_pct0:.0f}%"),
                            ("미채택 벤치마크", "6개 (50% 미만)"),
                            ("다중회귀 교집합", f"MMLU {n_m_multi} · BBH {n_b_multi}"),
                        ],
                    )

        with st.container(border=True, key="card_sec0_table"):
            st.markdown(
                f"<div class='panel-title'>원자료 테이블 · benchmarks_with_model_accessibility.csv "
                f"({len(view)} rows × {len(raw_cols)} cols)</div>",
                unsafe_allow_html=True,
            )
            raw_export = view[raw_cols].copy()
            st.dataframe(raw_export, width="stretch", height=320, hide_index=True)
            st.download_button(
                label="Download CSV (UTF-8)",
                data=raw_export.to_csv(index=False).encode("utf-8-sig"),
                file_name="benchmarks_with_model_accessibility_filtered.csv",
                mime="text/csv",
                key="raw_csv_download",
            )

    # ── TAB 2: Unified Time-Trend Toggle (Benchmark + 3 Scales) ──────────────
    with tab1:
        y_note = (
            "※ 버튼으로 벤치마크(0~1)와 규모 지표(log10)를 전환합니다. "
            "지표별 Y축 단위와 범위가 다르므로 기울기를 직접 비교하지 않습니다."
        )
        st.markdown(
            f"<div class='y-axis-note'>{html.escape(y_note, quote=True)}</div>",
            unsafe_allow_html=True,
        )
        shared_x = compute_shared_time_range(view)
        scale_reg_time_map = {
            "log10_flop": reg_time_flop,
            "log10_params": reg_time_param,
            "log10_dataset": reg_time_data,
        }
        t1_options = [bench_choice] + scale_options
        if st.session_state.get("sec1_time_toggle") not in t1_options:
            st.session_state["sec1_time_toggle"] = bench_choice

        if show_summary:
            col_t1_chart, col_t1_side = st.columns([2.7, 1.3], gap="medium")
        else:
            col_t1_chart, col_t1_side = st.container(), None

        with col_t1_chart:
            with st.container(border=True, key="card_sec1_time"):
                cur_t1 = st.session_state.get("sec1_time_toggle", bench_choice)
                if cur_t1 == bench_choice:
                    t1_title = f"출시 시점별 {bench_choice} 추이 (n={len(perf_valid)})"
                else:
                    cur_t1_meta = SCALE_META[scale_label_to_key[cur_t1]]
                    cur_t1_n = int((view[cur_t1_meta["num_col"]] > 0).sum())
                    t1_title = f"출시 시점별 {cur_t1_meta['raw_col']} 추이 (n={cur_t1_n})"
                st.markdown(
                    f"<div class='panel-title'>{html.escape(t1_title, quote=True)}</div>",
                    unsafe_allow_html=True,
                )
                sec1_choice = st.radio(
                    "Section 1 Time Metric Toggle",
                    options=t1_options,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="sec1_time_toggle",
                )

                if sec1_choice == bench_choice:
                    if show_reg:
                        stat_line = format_ols_stat_line(reg_time_perf, mode="time_perf")
                        st.markdown(
                            f"<div class='panel-stat'>{html.escape(stat_line, quote=True)}</div>",
                            unsafe_allow_html=True,
                        )
                    if not perf_valid.empty:
                        fig_t1 = build_time_performance_chart(
                            df=view,
                            bench_col=bench_choice,
                            reg=reg_time_perf,
                            show_reg=show_reg,
                            show_pi=show_pi,
                            x_range=shared_x,
                            random_baseline=random_line,
                        )
                        st.plotly_chart(fig_t1, width="stretch", theme=None, config=PLOTLY_CONFIG)
                    else:
                        st.info(f"{bench_choice}: n=0")
                else:
                    s_col = scale_label_to_key[sec1_choice]
                    s_meta = SCALE_META[s_col]
                    reg_obj = scale_reg_time_map[s_col]
                    valid_n = int((view[s_meta["num_col"]] > 0).sum())
                    if show_reg:
                        s_line = format_ols_stat_line(reg_obj, mode="time_log10")
                        st.markdown(
                            f"<div class='panel-stat'>{html.escape(s_line, quote=True)}</div>",
                            unsafe_allow_html=True,
                        )
                    if valid_n > 0:
                        fig_t1 = build_time_scale_chart(
                            df=view,
                            scale_col=s_col,
                            reg=reg_obj,
                            show_reg=show_reg,
                            show_pi=show_pi,
                            x_range=shared_x,
                        )
                        st.plotly_chart(fig_t1, width="stretch", theme=None, config=PLOTLY_CONFIG)
                    else:
                        st.info(f"{s_meta['raw_col']}: n=0")

        if col_t1_side is not None:
            with col_t1_side:
                if sec1_choice == bench_choice:
                    sub_t1p = view.dropna(subset=["year_frac", bench_choice]).copy()
                    n_t1p = len(sub_t1p)
                    o_t1p = sub_t1p.loc[sub_t1p["_access"] == "Open", bench_choice].dropna()
                    c_t1p = sub_t1p.loc[sub_t1p["_access"] == "Closed", bench_choice].dropna()
                    o_pct_t1p = (100.0 * len(o_t1p) / n_t1p) if n_t1p else 0.0
                    c_pct_t1p = (100.0 * len(c_t1p) / n_t1p) if n_t1p else 0.0
                    vals_t1p = pd.to_numeric(sub_t1p[bench_choice], errors="coerce").dropna()
                    if not vals_t1p.empty:
                        q1_v, med_v, q3_v = np.quantile(vals_t1p, [0.25, 0.50, 0.75])
                        above_n = int((vals_t1p > random_line).sum())
                        rows_t1p = [
                            ("유효 표본", f"n={n_t1p}"),
                            ("최고 / 최저", f"{vals_t1p.max():.3f} / {vals_t1p.min():.3f}"),
                            ("중앙값", f"{med_v:.3f}"),
                            ("사분위 (IQR)", f"{q1_v:.2f} ~ {q3_v:.2f}"),
                            ("Open 중앙값", f"{o_t1p.median():.3f} (n={len(o_t1p)})" if not o_t1p.empty else "-"),
                            ("Closed 중앙값", f"{c_t1p.median():.3f} (n={len(c_t1p)})" if not c_t1p.empty else "-"),
                            (f">{random_line:.2f} 초과", f"{above_n}/{n_t1p} ({100.0 * above_n / n_t1p:.0f}%)"),
                        ]
                    else:
                        rows_t1p = [("유효 표본", "n=0")]
                    if show_reg and reg_time_perf.get("ok"):
                        rows_t1p.append(("연간 기울기", f"{reg_time_perf['b1']:+.3f} / yr"))
                        rows_t1p.append(("설명력 (R²)", f"{reg_time_perf['r2']:.3f}"))
                    _render_summary_card(
                        kicker=f"PLOT SUMMARY · {bench_choice}",
                        left_pct=o_pct_t1p,
                        right_pct=c_pct_t1p,
                        left_cap=f"Open {len(o_t1p)} ({o_pct_t1p:.0f}%)",
                        right_cap=f"Closed {len(c_t1p)} ({c_pct_t1p:.0f}%)",
                        rows=rows_t1p,
                    )
                else:
                    s_col = scale_label_to_key[sec1_choice]
                    s_meta = SCALE_META[s_col]
                    reg_obj = scale_reg_time_map[s_col]
                    num_s1 = s_meta["num_col"]
                    sub_t1s = view.dropna(subset=["year_frac", num_s1]).copy()
                    sub_t1s = sub_t1s[sub_t1s[num_s1] > 0]
                    n_t1s = len(sub_t1s)
                    o_t1s = sub_t1s.loc[sub_t1s["_access"] == "Open", num_s1]
                    c_t1s = sub_t1s.loc[sub_t1s["_access"] == "Closed", num_s1]
                    o_pct_t1s = (100.0 * len(o_t1s) / n_t1s) if n_t1s else 0.0
                    c_pct_t1s = (100.0 * len(c_t1s) / n_t1s) if n_t1s else 0.0
                    if n_t1s > 0:
                        s_vals = sub_t1s[num_s1]
                        log_span = math.log10(float(s_vals.max())) - math.log10(float(s_vals.min()))
                        rows_t1s = [
                            ("유효 표본", f"n={n_t1s}"),
                            ("최대값", _fmt_scale_value(float(s_vals.max()), s_col)),
                            ("중앙값", _fmt_scale_value(float(s_vals.median()), s_col)),
                            ("최소값", _fmt_scale_value(float(s_vals.min()), s_col)),
                            ("범위 (log10)", f"{log_span:.1f} 자릿수"),
                            ("Open 중앙값", _fmt_scale_value(float(o_t1s.median()), s_col) if not o_t1s.empty else "-"),
                            ("Closed 중앙값", _fmt_scale_value(float(c_t1s.median()), s_col) if not c_t1s.empty else "-"),
                        ]
                    else:
                        rows_t1s = [("유효 표본", "n=0")]
                    if show_reg and reg_obj.get("ok"):
                        rows_t1s.append(("연간 성장", f"{10.0 ** reg_obj['b1']:.2f}x / yr"))
                        rows_t1s.append(("설명력 (R²)", f"{reg_obj['r2']:.3f}"))
                    _render_summary_card(
                        kicker="PLOT SUMMARY · SCALE",
                        left_pct=o_pct_t1s,
                        right_pct=c_pct_t1s,
                        left_cap=f"Open {len(o_t1s)} ({o_pct_t1s:.0f}%)",
                        right_cap=f"Closed {len(c_t1s)} ({c_pct_t1s:.0f}%)",
                        rows=rows_t1s,
                    )

    # ── TAB 2: Open / Closed Distribution (Asymmetric Editorial Split) ───────
    with tab2:
        if show_summary:
            col_chart, col_side = st.columns([2.7, 1.3], gap="medium")
        else:
            col_chart, col_side = st.container(), None
        with col_chart:
            with st.container(border=True, key="card_sec2_oc"):
                oc_title = f"연도별 Open / Closed 분포 ({yr_span_txt} · 총 n={len(view)})"
                st.markdown(
                    f"<div class='panel-title'>{html.escape(oc_title, quote=True)}</div>",
                    unsafe_allow_html=True,
                )
                oc_mode = st.radio(
                    "Open/Closed View Toggle",
                    options=["연도별 모델 수 (Bar)", "전체 비율 (Pie)"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="open_closed_toggle",
                )
                if oc_mode == "연도별 모델 수 (Bar)":
                    fig_oc = build_open_closed_bar_chart(view, yr_range[0], yr_range[1])
                else:
                    fig_oc = build_open_closed_pie_chart(view)
                st.plotly_chart(fig_oc, width="stretch", theme=None, config=PLOTLY_CONFIG)

        if col_side is not None:
            with col_side:
                total_n = max(len(view), 1)
                open_pct = 100.0 * open_n / total_n
                closed_pct = 100.0 * closed_n / total_n
                yr_counts = view.groupby("_year").size()
                peak_yr = int(yr_counts.idxmax()) if not yr_counts.empty else yr_range[0]
                peak_cnt = int(yr_counts.max()) if not yr_counts.empty else 0

                open_yr = view.loc[view["_access"] == "Open"].groupby("_year").size()
                closed_yr = view.loc[view["_access"] == "Closed"].groupby("_year").size()
                open_peak_txt = f"{int(open_yr.idxmax())} ({int(open_yr.max())}개)" if not open_yr.empty else "-"
                closed_peak_txt = f"{int(closed_yr.idxmax())} ({int(closed_yr.max())}개)" if not closed_yr.empty else "-"

                _render_summary_card(
                    kicker="PLOT SUMMARY · OPEN / CLOSED",
                    left_pct=open_pct,
                    right_pct=closed_pct,
                    left_cap=f"Open {open_n} ({open_pct:.0f}%)",
                    right_cap=f"Closed {closed_n} ({closed_pct:.0f}%)",
                    rows=[
                        ("전체 모델 수", f"n={len(view)}"),
                        ("분석 기간", yr_span_txt),
                        ("최다 출시 연도", f"{peak_yr} ({peak_cnt}개)"),
                        ("Open 최다 연도", open_peak_txt),
                        ("Closed 최다 연도", closed_peak_txt),
                    ],
                )

    # ── TAB 4: Simple OLS (Single-Variable Scaling Regressions ONLY) ─────────
    with tab3:
        shared_direct_y = compute_shared_perf_y_range(
            view,
            bench_choice,
            [reg_dir_flop, reg_dir_param, reg_dir_total, reg_dir_data],
            show_reg=show_reg,
            show_pi=show_pi,
        )
        scaling_label_to_key = {
            SCALE_META[k]["raw_col"]: k
            for k in ["log10_flop", "log10_params", "log10_total_params", "log10_dataset"]
        }
        scaling_options = list(scaling_label_to_key.keys())
        scale_reg_dir_map = {
            "log10_flop": reg_dir_flop,
            "log10_params": reg_dir_param,
            "log10_total_params": reg_dir_total,
            "log10_dataset": reg_dir_data,
        }

        if show_summary:
            col_t3_chart, col_t3_side = st.columns([2.7, 1.3], gap="medium")
        else:
            col_t3_chart, col_t3_side = st.container(), None
        with col_t3_chart:
            with st.container(border=True, key="card_sec3_direct"):
                cur_s3 = st.session_state.get("sec2_scale_toggle", scaling_options[0])
                if cur_s3 not in scaling_options:
                    cur_s3 = scaling_options[0]
                cur_s3_meta = SCALE_META[scaling_label_to_key[cur_s3]]
                cur_s3_num = cur_s3_meta["num_col"]
                cur_s3_n = int(view[[cur_s3_num, bench_choice]].dropna().loc[lambda d: d[cur_s3_num] > 0].shape[0])
                s3_title = f"{cur_s3_meta['raw_col']} vs {bench_choice} (n={cur_s3_n})"
                st.markdown(
                    f"<div class='panel-title'>{html.escape(s3_title, quote=True)}</div>",
                    unsafe_allow_html=True,
                )
                sec2_scale_label = st.radio(
                    "Section 2 Scale Toggle",
                    options=scaling_options,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="sec2_scale_toggle",
                )
                d_col = scaling_label_to_key[sec2_scale_label]
                d_meta = SCALE_META[d_col]
                d_reg = scale_reg_dir_map[d_col]
                num_col = d_meta["num_col"]
                pair_n = int(view[[num_col, bench_choice]].dropna().loc[lambda d: d[num_col] > 0].shape[0])

                if show_reg:
                    d_line = format_ols_stat_line(d_reg, mode="direct_scale")
                    st.markdown(
                        f"<div class='panel-stat'>{html.escape(d_line, quote=True)}</div>",
                        unsafe_allow_html=True,
                    )

                if pair_n > 0:
                    fig_d = build_direct_scaling_chart(
                        df=view,
                        scale_col=d_col,
                        bench_col=bench_choice,
                        reg=d_reg,
                        show_reg=show_reg,
                        show_pi=show_pi,
                        y_range=shared_direct_y,
                        random_baseline=random_line,
                        show_y_title=True,
                    )
                    st.plotly_chart(fig_d, width="stretch", theme=None, config=PLOTLY_CONFIG)
                else:
                    st.info(f"{d_meta['raw_col']} vs {bench_choice}: n=0")

        if col_t3_side is not None:
            with col_t3_side:
                sub_t3 = view.dropna(subset=[num_col, bench_choice]).copy()
                sub_t3 = sub_t3[sub_t3[num_col] > 0]
                n_t3 = len(sub_t3)
                o_t3 = sub_t3.loc[sub_t3["_access"] == "Open", bench_choice]
                c_t3 = sub_t3.loc[sub_t3["_access"] == "Closed", bench_choice]
                o_pct_t3 = (100.0 * len(o_t3) / n_t3) if n_t3 else 0.0
                c_pct_t3 = (100.0 * len(c_t3) / n_t3) if n_t3 else 0.0
                if n_t3 > 0:
                    rows_t3 = [
                        ("유효 쌍 표본", f"n={n_t3}"),
                        ("규모 중앙값", _fmt_scale_value(float(sub_t3[num_col].median()), d_col)),
                        (f"{bench_choice} 중앙값", f"{sub_t3[bench_choice].median():.3f}"),
                        ("Open 중앙값", f"{o_t3.median():.3f} (n={len(o_t3)})" if not o_t3.empty else "-"),
                        ("Closed 중앙값", f"{c_t3.median():.3f} (n={len(c_t3)})" if not c_t3.empty else "-"),
                    ]
                else:
                    rows_t3 = [("유효 쌍 표본", "n=0")]
                if show_reg and d_reg.get("ok"):
                    rows_t3.append(("기울기 (10x)", f"{d_reg['b1']:+.3f}"))
                    rows_t3.append(("p-value", format_p_short(d_reg["p"])))
                    rows_t3.append(("R² / RSE", f"{d_reg['r2']:.3f} / {d_reg['rse']:.3f}"))
                _render_summary_card(
                    kicker="PLOT SUMMARY · SCALING",
                    left_pct=o_pct_t3,
                    right_pct=c_pct_t3,
                    left_cap=f"Open {len(o_t3)} ({o_pct_t3:.0f}%)",
                    right_cap=f"Closed {len(c_t3)} ({c_pct_t3:.0f}%)",
                    rows=rows_t3,
                )

        if show_reg:
            with st.container(border=True, key="card_sec3_simple"):
                se_tag = " · HC3" if use_hc3 else ""
                sp_title = f"규모 축별 스케일링 회귀 비교 (개별 OLS · Y = {bench_choice}{se_tag})"
                st.markdown(
                    f"<div class='panel-title'>{html.escape(sp_title, quote=True)}</div>",
                    unsafe_allow_html=True,
                )
                simple_df = _format_simple_table(
                    [
                        ("log10(Training compute (FLOP))", reg_dir_flop),
                        ("log10(Active Parameters)", reg_dir_param),
                        ("log10(Model size (parameters))", reg_dir_total),
                        ("log10(Dataset size)", reg_dir_data),
                    ],
                    use_hc3=use_hc3,
                )
                simple_styled = _style_derived_table(simple_df, fmt_map=SIMPLE_FMT)
                st.dataframe(simple_styled, width="stretch", height="content", hide_index=True)

    # ── TAB 4: Dedicated Multiple OLS & VIF Comparison ───────────────────────
    with tab4:
        if show_reg:
            multi_opts = ["Model A (2변수)", "Model B (3변수)"]
            multi_map = {
                "Model A (2변수)": (
                    reg_multi_2,
                    f"Model A (2변수): {bench_choice} ~ log10(Active Parameters) + log10(Dataset size)",
                    2,
                ),
                "Model B (3변수)": (
                    reg_multi_3,
                    f"Model B (3변수): {bench_choice} ~ log10(Training compute (FLOP)) + log10(Active Parameters) + log10(Dataset size)",
                    3,
                ),
            }

            if show_summary:
                col_t4m_chart, col_t4m_side = st.columns([2.7, 1.3], gap="medium")
            else:
                col_t4m_chart, col_t4m_side = st.container(), None

            with col_t4m_chart:
                with st.container(border=True, key="card_sec4_multi"):
                    cur_m = st.session_state.get("sec4_multi_toggle", multi_opts[0])
                    if cur_m not in multi_opts:
                        cur_m = multi_opts[0]
                    cur_m_title = multi_map[cur_m][1]
                    st.markdown(
                        f"<div class='panel-title'>{html.escape(cur_m_title, quote=True)}</div>",
                        unsafe_allow_html=True,
                    )
                    m_choice = st.radio(
                        "Multiple Regression Model Toggle",
                        options=multi_opts,
                        horizontal=True,
                        label_visibility="collapsed",
                        key="sec4_multi_toggle",
                    )
                    m_reg, m_title, m_k = multi_map[m_choice]
                    if m_reg.get("ok"):
                        m_hdr = _format_multi_header(m_reg)
                        st.markdown(
                            f"<div class='panel-stat'>{html.escape(m_hdr, quote=True)}</div>",
                            unsafe_allow_html=True,
                        )
                        fig_m = build_multi_actual_vs_predicted_chart(m_reg, bench_choice)
                        st.plotly_chart(fig_m, width="stretch", theme=None, config=PLOTLY_CONFIG)
                    else:
                        st.info(m_reg.get("reason", "표본 부족"))

            if col_t4m_side is not None:
                with col_t4m_side:
                    if m_reg.get("ok"):
                        acc_arr = np.asarray(m_reg["access_ordered"], dtype=object)
                        n_m = len(acc_arr)
                        o_m = int(np.sum(acc_arr == "Open"))
                        c_m = int(np.sum(acc_arr == "Closed"))
                        o_pct_m = (100.0 * o_m / n_m) if n_m else 0.0
                        c_pct_m = (100.0 * c_m / n_m) if n_m else 0.0
                        mv_m = max(m_reg["vifs"].values()) if m_reg["vifs"] else float("nan")
                        vif_txt_m = _fmt_vif(mv_m)
                        vif_cls_m = "vif-warn-inline" if (pd.notna(mv_m) and mv_m >= 5.0) else "vif-ok-inline"
                        vif_badge_m = (
                            f"<div class='side-stat-row'>"
                            f"<span class='side-stat-k'>최대 VIF</span>"
                            f"<span class='side-stat-v'><span class='{vif_cls_m}'>{html.escape(vif_txt_m)}</span></span>"
                            f"</div>"
                        )
                        _render_summary_card(
                            kicker="PLOT SUMMARY · MULTI OLS",
                            left_pct=o_pct_m,
                            right_pct=c_pct_m,
                            left_cap=f"Open {o_m} ({o_pct_m:.0f}%)",
                            right_cap=f"Closed {c_m} ({c_pct_m:.0f}%)",
                            rows=[
                                ("분석 표본", f"n={n_m}"),
                                ("설명변수 수", f"{m_k}개"),
                                ("설명력 (R²)", f"{m_reg['r2']:.3f}"),
                                ("수정 R²", f"{m_reg['adj_r2']:.3f}"),
                                ("잔차 오차 (RSE)", f"{m_reg['rse']:.3f}"),
                                ("F-통계량", f"{m_reg['f_stat']:.2f}"),
                                ("F p-value", format_p_short(m_reg["f_pval"])),
                            ],
                            badge_html=vif_badge_m,
                        )
                    else:
                        _render_summary_card(
                            kicker="PLOT SUMMARY · MULTI OLS",
                            left_pct=0.0,
                            right_pct=0.0,
                            left_cap="Open 0",
                            right_cap="Closed 0",
                            rows=[("상태", str(m_reg.get("reason", "표본 부족")))],
                        )

            def _max_vif_html(res: dict) -> str:
                if not res.get("ok") or not res.get("vifs"):
                    return "-"
                mv = max(res["vifs"].values())
                if pd.isna(mv):
                    return "-"
                txt = _fmt_vif(mv)
                cls = "vif-warn-inline" if mv >= 5.0 else "vif-ok-inline"
                return f"<span class='{cls}'>{html.escape(txt)}</span>"

            ma_adj = f"{reg_multi_2['adj_r2']:.3f}" if reg_multi_2.get("ok") else "-"
            ma_rse = f"{reg_multi_2['rse']:.3f}" if reg_multi_2.get("ok") else "-"
            mb_adj = f"{reg_multi_3['adj_r2']:.3f}" if reg_multi_3.get("ok") else "-"
            mb_rse = f"{reg_multi_3['rse']:.3f}" if reg_multi_3.get("ok") else "-"
            st.markdown(
                f"<div class='compare-strip'>"
                f"<div class='compare-box' style='--box-accent:{T.DATA_OPEN};'>"
                f"<div class='compare-head'>Model A (2변수) · Active Parameters + Dataset size</div>"
                f"<div class='compare-metrics'>"
                f"<span>Adj. R² <strong>{ma_adj}</strong></span>"
                f"<span>RSE <strong>{ma_rse}</strong></span>"
                f"<span>Max VIF {_max_vif_html(reg_multi_2)}</span>"
                f"</div></div>"
                f"<div class='compare-box' style='--box-accent:{T.DATA_CLOSED};'>"
                f"<div class='compare-head'>Model B (3변수) · + Training compute (FLOP)</div>"
                f"<div class='compare-metrics'>"
                f"<span>Adj. R² <strong>{mb_adj}</strong></span>"
                f"<span>RSE <strong>{mb_rse}</strong></span>"
                f"<span>Max VIF {_max_vif_html(reg_multi_3)}</span>"
                f"</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.container(border=True, key="card_sec4_multi_table"):
                st.markdown(
                    f"<div class='panel-title'>{html.escape(m_choice, quote=True)} 회귀 계수 및 VIF</div>",
                    unsafe_allow_html=True,
                )
                if m_reg.get("ok"):
                    m_styled = _style_derived_table(
                        _format_multi_table(m_reg),
                        fmt_map=MULTI_FMT,
                        warn_col="VIF",
                        warn_predicate=lambda v: pd.notna(v) and float(v) >= 5.0,
                        ok_predicate=lambda v: pd.notna(v) and float(v) < 5.0,
                    )
                    st.dataframe(m_styled, width="stretch", height="content", hide_index=True)
                else:
                    st.info(m_reg.get("reason", "표본 부족"))
        else:
            st.info("사이드바에서 'OLS 회귀선 + 95% 신뢰구간'을 켜면 다중 회귀 분석(Model A · Model B)이 표시됩니다.")

    # ── TAB 5: LINE Diagnostics & Untouched Raw CSV ──────────────────────────
    with tab5:
        if show_reg:
            diag_models = {
                f"log10(Training compute (FLOP)) vs {bench_choice}": (reg_dir_flop, bench_choice),
                f"log10(Active Parameters) vs {bench_choice}": (reg_dir_param, bench_choice),
                f"log10(Model size (parameters)) vs {bench_choice}": (reg_dir_total, bench_choice),
                f"log10(Dataset size) vs {bench_choice}": (reg_dir_data, bench_choice),
                f"Date vs {bench_choice}": (reg_time_perf, bench_choice),
                f"Model A: log10(Active Parameters) + log10(Dataset size) vs {bench_choice}": (
                    reg_multi_2,
                    bench_choice,
                ),
                f"Model B: log10(Training compute (FLOP)) + log10(Active Parameters) + log10(Dataset size) vs {bench_choice}": (
                    reg_multi_3,
                    bench_choice,
                ),
            }
            if show_summary:
                col_t5_chart, col_t5_side = st.columns([2.7, 1.3], gap="medium")
            else:
                col_t5_chart, col_t5_side = st.container(), None
            with col_t5_chart:
                with st.container(border=True, key="card_sec5_diag"):
                    diag_key = st.selectbox("Model", options=list(diag_models.keys()))
                    d_reg, d_ylabel = diag_models[diag_key]
                    if not d_reg.get("ok"):
                        st.info(d_reg.get("reason", "n < 4"))
                    else:
                        d_line = d_reg["line"]
                        line_rows = evaluate_line_table(d_line)
                        st.markdown("<div class='panel-title'>진단 플롯</div>", unsafe_allow_html=True)
                        diag_plot_choice = st.radio(
                            "LINE Plot Toggle",
                            options=["Residual vs Fitted", "Normal Q-Q", "Residuals over Date"],
                            horizontal=True,
                            label_visibility="collapsed",
                            key="sec3_diag_toggle",
                        )
                        if diag_plot_choice == "Residual vs Fitted":
                            fig_diag = build_residual_plot(d_line, d_ylabel)
                        elif diag_plot_choice == "Normal Q-Q":
                            fig_diag = build_qq_plot(d_line)
                        else:
                            fig_diag = build_time_residual_plot(d_line, d_ylabel)
                        st.plotly_chart(fig_diag, width="stretch", theme=None, config=PLOTLY_CONFIG)

            if col_t5_side is not None:
                with col_t5_side:
                    if d_reg.get("ok"):
                        resid_arr = np.asarray(d_line["resid"], dtype=float)
                        n_res = len(resid_arr)
                        pos_n = int(np.sum(resid_arr >= 0)) if n_res else 0
                        neg_n = int(np.sum(resid_arr < 0)) if n_res else 0
                        pos_pct = (100.0 * pos_n / n_res) if n_res else 0.0
                        neg_pct = (100.0 * neg_n / n_res) if n_res else 0.0
                        sd_res = float(np.std(resid_arr, ddof=1)) if n_res > 1 else 0.0
                        max_abs = float(np.max(np.abs(resid_arr))) if n_res else 0.0
                        out2_n = int(np.sum(np.abs(resid_arr) > 2.0 * sd_res)) if sd_res > 0 else 0
                        ok_cnt = sum(1 for r in line_rows if r.get("Status") == "OK")
                        ok_cls = "vif-ok-inline" if ok_cnt == 4 else "vif-warn-inline"
                        line_badge = (
                            f"<div class='side-stat-row'>"
                            f"<span class='side-stat-k'>LINE 가정 통과</span>"
                            f"<span class='side-stat-v'><span class='{ok_cls}'>{ok_cnt} / 4</span></span>"
                            f"</div>"
                        )
                        diag_rows = [
                            ("분석 표본", f"n={n_res}"),
                            ("잔차 표준편차", f"{sd_res:.3f}"),
                            ("최대 절대 잔차", f"{max_abs:.3f}"),
                            ("|e| > 2SD", f"{out2_n}개"),
                        ]
                        if diag_plot_choice == "Residual vs Fitted":
                            diag_rows.append(("선형성 (RESET p)", format_p_short(d_line.get("reset_p"))))
                            diag_rows.append(("등분산 (BP p)", format_p_short(d_line.get("bp_p"))))
                        elif diag_plot_choice == "Normal Q-Q":
                            diag_rows.append(("정규성 (Shapiro p)", format_p_short(d_line.get("shapiro_p"))))
                        else:
                            dw_val = d_line.get("dw", float("nan"))
                            diag_rows.append(("독립성 (DW)", f"{dw_val:.2f}" if pd.notna(dw_val) else "-"))

                        _render_summary_card(
                            kicker="PLOT SUMMARY · RESIDUAL",
                            left_pct=pos_pct,
                            right_pct=neg_pct,
                            left_cap=f"양(+) {pos_n} ({pos_pct:.0f}%)",
                            right_cap=f"음(-) {neg_n} ({neg_pct:.0f}%)",
                            rows=diag_rows,
                            badge_html=line_badge,
                        )
                    else:
                        _render_summary_card(
                            kicker="PLOT SUMMARY · RESIDUAL",
                            left_pct=0.0,
                            right_pct=0.0,
                            left_cap="양(+) 0",
                            right_cap="음(-) 0",
                            rows=[("상태", str(d_reg.get("reason", "표본 부족")))],
                        )

            if d_reg.get("ok"):
                with st.container(border=True, key="card_sec5_tables"):
                    st.markdown("<div class='panel-title'>LINE 4대 가정 검정</div>", unsafe_allow_html=True)
                    line_styled = _style_derived_table(
                        pd.DataFrame(line_rows),
                        fmt_map=None,
                        warn_col="Status",
                        warn_predicate=lambda v: v == "Check",
                        ok_predicate=lambda v: v == "OK",
                    )
                    st.dataframe(line_styled, width="stretch", height="content", hide_index=True)

                    st.markdown("<div class='panel-title'>OLS vs HC3</div>", unsafe_allow_html=True)
                    hc3_styled = _style_derived_table(
                        _format_hc3_table(d_reg),
                        fmt_map=HC3_FMT,
                    )
                    st.dataframe(hc3_styled, width="stretch", height="content", hide_index=True)
        else:
            st.info("사이드바에서 'OLS 회귀선 + 95% 신뢰구간'을 켜면 회귀 가정 진단(LINE · OLS vs HC3)이 표시됩니다.")

        with st.container(border=True, key="card_sec5_limits"):
            st.markdown("<div class='panel-title'>핵심 결론 및 해석의 한계</div>", unsafe_allow_html=True)

            open_sub = view[view["_access"] == "Open"]
            closed_sub = view[view["_access"] == "Closed"]
            o_ap_txt = (
                f"{100.0 * open_sub['Active Parameters'].notna().mean():.1f}%"
                if len(open_sub) > 0
                else "-"
            )
            c_ap_txt = (
                f"{100.0 * closed_sub['Active Parameters'].notna().mean():.1f}%"
                if len(closed_sub) > 0
                else "-"
            )
            if reg_dir_flop.get("ok"):
                bp_p_txt = format_p(reg_dir_flop["line"].get("bp_p"))
                hc3_z_txt = (
                    f"{reg_dir_flop['t_hc3']:.2f}"
                    if pd.notna(reg_dir_flop.get("t_hc3"))
                    else "-"
                )
                hc3_p_txt = format_p(reg_dir_flop.get("p_hc3"))
                flop_summary = (
                    f"Training compute (FLOP) → {bench_choice}: "
                    f"R²={reg_dir_flop['r2']:.3f}, 10배당 {reg_dir_flop['b1']:+.3f} "
                    f"(n={reg_dir_flop['n']}, HC3 z={hc3_z_txt}, {hc3_p_txt})"
                )
            else:
                bp_p_txt = "p = -"
                flop_summary = f"Training compute (FLOP) → {bench_choice}: 현재 필터 표본 부족"

            if reg_multi_2.get("ok"):
                ma_vif_val = max(reg_multi_2["vifs"].values()) if reg_multi_2.get("vifs") else float("nan")
                ma_summary = (
                    f"Model A (2변수, n={reg_multi_2['n']}): "
                    f"Adj. R²={reg_multi_2['adj_r2']:.3f}, Max VIF={_fmt_vif(ma_vif_val)}"
                )
            else:
                ma_summary = "Model A (2변수): 현재 필터 표본 부족"

            if reg_multi_3.get("ok"):
                mb_vif_val = max(reg_multi_3["vifs"].values()) if reg_multi_3.get("vifs") else float("nan")
                mb_vif_txt = _fmt_vif(mb_vif_val)
            else:
                mb_vif_txt = "표본 부족"

            limit_rows = [
                (
                    "실증 결론",
                    T.DATA_OPEN,
                    f"{flop_summary} · {ma_summary}",
                ),
                (
                    "관측 자료 한계",
                    T.DATA_REG_LINE,
                    "Kaplan(2020)의 통제 실험(cross-entropy loss)과 달리 기업·아키텍처·사후학습이 혼재된 관측 자료의 "
                    "다운스트림 정답률(0~1) 실증이며, 회귀선은 관측 범위 내부만 표시합니다(미래 외삽 금지).",
                ),
                (
                    "비공개 결측 편향",
                    T.DATA_CLOSED,
                    f"최신 Closed 모델의 규모 정보 비공개로 결측이 집중됩니다 "
                    f"(Active Parameters 공개율 Open {o_ap_txt} vs Closed {c_ap_txt}, 결측 임의 대체 없음).",
                ),
                (
                    "다중공선성",
                    "#B84D00",
                    f"C ≈ 6ND 구조 때문에 Model B(3변수)에 FLOP을 동시 투입하면 Max VIF={mb_vif_txt}로 "
                    "개별 계수가 불안정해지므로, 2변수 Model A와 단일 FLOP 모형을 분리해 해석합니다.",
                ),
                (
                    "포화 · 이분산",
                    "#475569",
                    f"벤치마크 점수가 0.25~1.00 구간에 갇혀 비선형 포화(RESET)와 등분산 경고(FLOP BP {bp_p_txt})가 "
                    "나타나므로, LINE 진단과 HC3 강건 표준오차를 함께 확인합니다.",
                ),
            ]
            limit_html = "".join(
                f"<tr>"
                f"<td><span class='col-guide-role'>"
                f"<span class='sunburst-swatch' style='background:{clr};'></span>"
                f"{html.escape(tag)}</span></td>"
                f"<td>{html.escape(desc)}</td>"
                f"</tr>"
                for tag, clr, desc in limit_rows
            )
            st.markdown(
                "<table class='col-guide-table'>"
                "<thead><tr><th>구분</th><th>데이터 근거 및 해석 주의사항</th></tr></thead>"
                f"<tbody>{limit_html}</tbody>"
                "</table>",
                unsafe_allow_html=True,
            )

    # ── Bottom-Only Data Source Link & Aggregation Standard ──────────────────
    st.divider()
    repo_url = html.escape(SOURCE_REPO_URL, quote=True)
    st.markdown(
        f"<div class='footer-links'>"
        f"<div>Source: "
        f"<a href='{repo_url}' target='_blank'>Epoch AI · open-model-trends / benchmarks_with_model_accessibility.csv (GitHub)</a></div>"
        f"<div style='margin-top:0.2rem;font-size:0.77rem;color:var(--color-text-muted);'>"
        f"집계 기준: 1행 = 1 AI 모델 (원본 132행 중 'Random chance' 기준선 1행 분리 · 실제 모델 131종 · 결측 미대체 · 이상값 미삭제)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
