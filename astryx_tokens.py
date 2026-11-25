"""Design Tokens & Colorblind-Safe Data Palette for the Scaling Law Dashboard.

Defines light-mode layered neutral surfaces (#F1F4F7 canvas, #FFFFFF elevated cards,
#E6EBF1 sidebar rail), WCAG AA+ text/status pairs, and colorblind-safe categorical
chart colors (#0171E3 Open, #EB6E00 Closed, #02165E OLS regression line).
"""
from __future__ import annotations

# ── Typography (Astryx Stone Montserrat/JetBrains Mono + Neutral Figtree + KR Pretendard) ─
FONT_FAMILY_HEADING = (
    "'Montserrat', 'Pretendard', 'Figtree', -apple-system, BlinkMacSystemFont, "
    "'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
)
FONT_FAMILY_BODY = (
    "'Figtree', 'Pretendard', -apple-system, BlinkMacSystemFont, "
    "'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
)
FONT_FAMILY_CODE = (
    "'JetBrains Mono', ui-monospace, 'SF Mono', Monaco, Consolas, monospace"
)

# ── Cool-Neutral Research Surfaces (Astryx Core + Swiss Sidebar Contrast) ────
COLOR_BG_BODY = "#F1F4F7"          # Astryx --color-background-body (high-clarity canvas)
COLOR_BG_SIDEBAR = "#E6EBF1"       # Refined cool-slate rail separating filters from cards
COLOR_BG_SURFACE = "#FFFFFF"       # Astryx --color-background-surface
COLOR_BG_CARD = "#FFFFFF"          # Astryx --color-background-card
COLOR_BG_MUTED = "#E4EAF0"         # Track fill for tabs & segmented controls

# ── Deep Midnight-Slate Hero Banner (Cool Research Match for #F1F4F7) ────────
COLOR_BG_HERO_START = "#0B1524"    # Deep Astryx ink navy
COLOR_BG_HERO_END = "#16263D"      # Refined midnight research slate
COLOR_BG_HERO_TILE = "rgba(255, 255, 255, 0.065)"  # Subtle dark-glass KPI panel
COLOR_BORDER_HERO = "rgba(255, 255, 255, 0.12)"

# ── Ink Typography Hierarchy ─────────────────────────────────────────────────
COLOR_TEXT_PRIMARY = "#0A1317"     # Astryx --color-text-primary
COLOR_TEXT_SECONDARY = "#445564"   # Astryx --color-text-secondary
COLOR_TEXT_MUTED = "#5D6C7B"       # Astryx --color-data-gray-4
COLOR_ON_DARK = "#FFFFFF"          # Astryx --color-on-dark
COLOR_ON_DARK_MUTED = "#B9C6D4"    # High-contrast cool mist on dark hero
COLOR_ON_LIGHT = "#0A1317"         # Astryx --color-on-light

# ── Accents & Hairline Borders ───────────────────────────────────────────────
COLOR_ACCENT = "#0064E0"           # Astryx Core --color-accent (focus ring & hover)
COLOR_ACCENT_NEUTRAL = "#1B2838"   # Neutral dark-slate chip & slider control
COLOR_HERO_BLUE = "#6EAEFF"        # Astryx blue.light[70] accent on dark hero
COLOR_HERO_ORANGE = "#FFB37E"      # Astryx orange.light[80] accent on dark hero
COLOR_HERO_GREEN = "#90CA90"       # Astryx green.light[75] accent on dark hero
COLOR_BORDER = "rgba(5, 54, 89, 0.11)"   # Astryx --color-border
COLOR_BORDER_EMPHASIZED = "#C8D2DC"      # Crisp hairline divider

# ── Astryx Semantic Status Pairs (9:1+ Contrast) ─────────────────────────────
COLOR_WARN_BG = "#FFE6CF"          # Astryx --color-data-orange-1
COLOR_WARN_TEXT = "#6B2203"        # Astryx --color-text-orange (9.4:1 contrast)
COLOR_OK_BG = "#E8FAEE"            # Soft tint of --color-data-shamrock-1
COLOR_OK_TEXT = "#09441F"          # Astryx --color-text-green (10.6:1 contrast)

# ── Official Astryx `tokens-data` Colorblind-Safe Chart Palette ──────────────
DATA_OPEN = "#0171E3"              # --color-data-categorical-blue (4.59:1 on #FFFFFF)
DATA_OPEN_FILL = "rgba(1, 113, 227, 0.56)"
DATA_OPEN_BAND = "rgba(1, 113, 227, 0.15)"

DATA_CLOSED = "#EB6E00"            # --color-data-categorical-orange
DATA_CLOSED_DARK = "#D66100"       # --color-data-orange-4 (sharper border)
DATA_CLOSED_FILL = "rgba(235, 110, 0, 0.56)"
DATA_CLOSED_BAND = "rgba(235, 110, 0, 0.11)"
DATA_CLOSED_BAND_LINE = "rgba(214, 97, 0, 0.34)"

DATA_REG_LINE = "#02165E"          # --color-data-blue-5 (16.2:1 midnight OLS line)
DATA_NEUTRAL = "#8494A3"           # --color-data-neutral
DATA_GRAY_5 = "#25363F"            # --color-data-gray-5
DATA_GRAY_4 = "#5D6C7B"            # --color-data-gray-4
DATA_GRAY_3 = "#AFB9C4"            # --color-data-gray-3
DATA_GRID_Y = "#DCE2E9"            # Crisp horizontal gridline
DATA_GRID_X = "#E6EBF0"            # Lighter vertical gridline

# ── Astryx Neutral Concentric Shape, Elevation & Motion ──────────────────────
RADIUS_INNER = "0.375rem"          # --radius-inner (6px)
RADIUS_ELEMENT = "0.625rem"        # --radius-element (10px)
RADIUS_CONTAINER = "0.75rem"       # --radius-container (12px)
RADIUS_PAGE = "1.125rem"           # Hero Banner outer radius (18px)
RADIUS_FULL = "9999px"             # --radius-full

SHADOW_LOW = (
    "0 1px 2px rgba(5, 54, 89, 0.05), "
    "0 4px 12px rgba(5, 54, 89, 0.06)"
)
SHADOW_HERO = (
    "0 6px 14px rgba(11, 21, 36, 0.16), "
    "0 16px 34px rgba(11, 21, 36, 0.18)"
)
DURATION_FAST = "125ms"            # --duration-fast
EASE_STANDARD = "cubic-bezier(0.24, 1, 0.4, 1)"
