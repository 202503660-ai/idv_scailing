"""Pure-Plotly chart builders for the Unified Flagship Edition (:8513).

Uses raw Epoch AI CSV columns and raw values verbatim with official Meta Astryx
`tokens-data` colorblind-safe categorical colors (#0171E3 Open circle vs
#EB6E00/#D66100 Closed square) and #02165E deep midnight-navy OLS line.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import astryx_tokens as T
from scaling_stats import ACCESS_OPTIONS, SCALE_META, format_p

FONT_FAMILY = T.FONT_FAMILY_BODY

# Official Astryx `tokens-data` colorblind-safe pair + dual shape encoding
ACCESS_STYLE: dict[str, dict[str, Any]] = {
    "Open": {
        "color": T.DATA_OPEN,
        "symbol": "circle",
        "fill": T.DATA_OPEN_FILL,
        "line": T.DATA_OPEN,
    },
    "Closed": {
        "color": T.DATA_CLOSED,
        "symbol": "square",
        "fill": T.DATA_CLOSED_FILL,
        "line": T.DATA_CLOSED_DARK,
    },
}

REG_LINE_COLOR = T.DATA_REG_LINE
CI_BAND_FILL = T.DATA_OPEN_BAND
PI_BAND_FILL = T.DATA_CLOSED_BAND
GRID_COLOR = T.DATA_GRID_Y
X_GRID_COLOR = T.DATA_GRID_X
AXIS_LINE_COLOR = T.DATA_GRAY_4
TEXT_COLOR = T.COLOR_TEXT_PRIMARY
MUTED_COLOR = T.COLOR_TEXT_SECONDARY

HOVERLABEL_STYLE: dict[str, Any] = dict(
    bgcolor="rgba(11, 21, 36, 0.94)",
    bordercolor=T.DATA_GRAY_5,
    font=dict(family=FONT_FAMILY, size=12.5, color=T.COLOR_ON_DARK),
    namelength=-1,
)

SUP_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _fmt_pow10(exp: int) -> str:
    return f"10{str(exp).translate(SUP_MAP)}"


def _fmt_scale_value(val: float, kind: str) -> str:
    if val is None or not math.isfinite(val) or val <= 0:
        return "-"
    if kind == "log10_flop":
        return f"{val:.2e}"
    if kind in ("log10_params", "log10_total_params"):
        if val >= 1e12:
            return f"{val / 1e12:.2f}T"
        if val >= 1e9:
            return f"{val / 1e9:.1f}B"
        return f"{val / 1e6:.0f}M"
    if kind == "log10_dataset":
        if val >= 1e12:
            return f"{val / 1e12:.2f}T"
        return f"{val / 1e9:,.0f}B"
    return f"{val:,.3g}"


def _log_axis_ticks(vals: pd.Series, kind: str) -> tuple[list[float], list[str], list[float]]:
    """Compute clean power-of-10 tick values, readable labels, and non-clipping log10 range."""
    pos = vals.dropna()
    pos = pos[pos > 0]
    if pos.empty:
        return [1.0], ["1"], [0.0, 1.0]
    lo_exp = math.floor(math.log10(float(pos.min())))
    hi_exp = math.ceil(math.log10(float(pos.max())))
    if hi_exp == lo_exp:
        hi_exp += 1
    span = hi_exp - lo_exp
    step = 2 if span >= 5 else 1
    exps = list(range(lo_exp, hi_exp + 1, step))

    tickvals = [10.0 ** e for e in exps]
    if kind == "log10_flop":
        ticktext = [_fmt_pow10(e) for e in exps]
    elif kind in ("log10_params", "log10_total_params"):
        labels_map = {7: "10M", 8: "100M", 9: "1B", 10: "10B", 11: "100B", 12: "1T"}
        ticktext = [labels_map.get(e, _fmt_pow10(e)) for e in exps]
    elif kind == "log10_dataset":
        labels_map = {10: "10B", 11: "100B", 12: "1T", 13: "10T", 14: "100T"}
        ticktext = [labels_map.get(e, _fmt_pow10(e)) for e in exps]
    else:
        ticktext = [_fmt_pow10(e) for e in exps]

    min_log = math.log10(float(pos.min()))
    max_log = math.log10(float(pos.max()))
    pad = max((max_log - min_log) * 0.12, 0.28)
    return tickvals, ticktext, [min_log - pad, max_log + pad]


def _base_layout(fig: go.Figure, height: int, margin: dict[str, Any], show_legend: bool) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=margin,
        font=dict(family=FONT_FAMILY, size=12.5, color=TEXT_COLOR),
        paper_bgcolor=T.COLOR_BG_CARD,
        plot_bgcolor=T.COLOR_BG_CARD,
        showlegend=show_legend,
        hovermode="closest",
        hoverlabel=HOVERLABEL_STYLE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.92)",
            font=dict(size=12, color=TEXT_COLOR),
            itemsizing="constant",
        ),
    )


def _add_regression_layers(
    fig: go.Figure,
    reg: dict[str, Any],
    show_pi: bool,
    x_is_log10: bool = False,
    y_is_log10: bool = False,
    show_band_legend: bool = True,
) -> None:
    """Overlay Week 6/7 OLS line + 95% mean CI + optional 95% obs PI strictly inside [x_min, x_max]."""
    if not reg.get("ok"):
        return

    xg = 10.0 ** reg["x_grid"] if x_is_log10 else reg["x_grid"]
    yhat = 10.0 ** reg["y_hat"] if y_is_log10 else reg["y_hat"]
    ci_lo = 10.0 ** reg["mean_ci_low"] if y_is_log10 else reg["mean_ci_low"]
    ci_hi = 10.0 ** reg["mean_ci_high"] if y_is_log10 else reg["mean_ci_high"]
    pi_lo = 10.0 ** reg["obs_ci_low"] if y_is_log10 else reg["obs_ci_low"]
    pi_hi = 10.0 ** reg["obs_ci_high"] if y_is_log10 else reg["obs_ci_high"]

    if show_pi:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xg, xg[::-1]]),
                y=np.concatenate([pi_hi, pi_lo[::-1]]),
                fill="toself",
                fillcolor=PI_BAND_FILL,
                line=dict(color=T.DATA_CLOSED_BAND_LINE, width=0.9, dash="dot"),
                name="95% PI",
                showlegend=show_band_legend,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xg, xg[::-1]]),
            y=np.concatenate([ci_hi, ci_lo[::-1]]),
            fill="toself",
            fillcolor=CI_BAND_FILL,
            line=dict(color="rgba(1,113,227,0.0)", width=0),
            name="95% CI",
            showlegend=show_band_legend,
            hoverinfo="skip",
        )
    )

    se_tag = " (HC3)" if reg.get("use_hc3") else ""
    fig.add_trace(
        go.Scatter(
            x=xg,
            y=yhat,
            mode="lines",
            line=dict(color=REG_LINE_COLOR, width=2.4),
            name=f"OLS{se_tag}",
            showlegend=show_band_legend,
            hovertemplate=f"OLS fit (R²={reg['r2']:.3f}, {format_p(reg['p'])})<extra></extra>",
        )
    )


def compute_shared_time_range(df: pd.DataFrame) -> list[float]:
    """Compute one synchronized Date X-axis range [lo, hi] for Tab 1 panels."""
    yrs = df["year_frac"].dropna()
    if yrs.empty:
        return [2018.8, 2025.0]
    lo = float(yrs.min())
    hi = float(yrs.max())
    pad = max((hi - lo) * 0.06, 0.22)
    return [lo - pad, hi + pad]


def compute_shared_perf_y_range(
    df: pd.DataFrame,
    bench_col: str,
    regs: list[dict[str, Any]],
    show_reg: bool,
    show_pi: bool,
) -> list[float]:
    """Compute non-clipping Y-axis range on raw 0~1 scale shared across panels."""
    y_lo, y_hi = 0.0, 1.04
    vals = pd.to_numeric(df[bench_col], errors="coerce").dropna()
    if not vals.empty:
        y_lo = min(y_lo, float(vals.min()) - 0.03)
        y_hi = max(y_hi, float(vals.max()) + 0.04)
    if show_reg:
        for reg in regs:
            if not reg.get("ok"):
                continue
            y_lo = min(y_lo, float(np.nanmin(reg["mean_ci_low"])) - 0.03)
            y_hi = max(y_hi, float(np.nanmax(reg["mean_ci_high"])) + 0.03)
            if show_pi:
                y_lo = min(y_lo, float(np.nanmin(reg["obs_ci_low"])) - 0.03)
                y_hi = max(y_hi, float(np.nanmax(reg["obs_ci_high"])) + 0.03)
    return [y_lo, y_hi]


def _select_yearly_top_models(sub: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Objectively select the single #1 model per release year by y_col (zero hardcoding).

    Ties on y_col (e.g., identical Dataset size within a release family) are broken
    deterministically by MMLU descending, Training compute (FLOP) descending, then name.
    """
    valid = sub.dropna(subset=["_year", y_col]).copy()
    if valid.empty:
        return valid
    valid["_y_sort"] = pd.to_numeric(valid[y_col], errors="coerce")
    valid = valid.dropna(subset=["_y_sort"])
    if valid.empty:
        return valid
    valid["_tie_mmlu"] = (
        pd.to_numeric(valid["MMLU"], errors="coerce").fillna(-1.0)
        if "MMLU" in valid.columns
        else -1.0
    )
    valid["_tie_flop"] = (
        pd.to_numeric(valid["_compute_flop"], errors="coerce").fillna(-1.0)
        if "_compute_flop" in valid.columns
        else -1.0
    )
    picked = (
        valid.sort_values(
            ["_year", "_y_sort", "_tie_mmlu", "_tie_flop", "_system"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        .groupby("_year", sort=True, as_index=False)
        .head(1)
        .sort_values("_year")
    )
    return picked


def _resolve_annotation_anchor(ax_px: int, ay_px: int) -> tuple[str, str]:
    """Deterministic Plotly text-box anchor matching arrow direction so geometry is exact."""
    xanchor = "left" if ax_px > 6 else ("right" if ax_px < -6 else "center")
    yanchor = "top" if ay_px > 6 else ("bottom" if ay_px < -6 else "middle")
    return xanchor, yanchor


def _compute_landmark_offset(
    xs: np.ndarray,
    ys: np.ndarray,
    x_val: float,
    y_val: float,
    x_range: list[float],
    y_range: list[float],
    placed: list[tuple[float, float, float, float]],
    ref_y: float | None = None,
    label_text: str = "",
) -> tuple[int, int]:
    """Compute minimum-collision annotation offset (ax, ay) in pixels for yearly #1 leaders."""
    x_span = float(x_range[1] - x_range[0])
    y_span = float(y_range[1] - y_range[0])
    if x_span <= 0 or y_span <= 0 or len(xs) == 0:
        return 0, -20

    xs_norm = (xs - float(x_range[0])) / x_span
    ys_norm = (ys - float(y_range[0])) / y_span
    x_norm = (x_val - float(x_range[0])) / x_span
    y_norm = (y_val - float(y_range[0])) / y_span

    # Calibrated to tightest 1280x800 split-view inner .gridlayer dimensions (376 x 332 px)
    plot_w, plot_h = 376.0, 332.0
    box_w_px = max(56.0, len(label_text) * 5.8 + 18.0)
    box_h_px = 22.0
    box_hw = box_w_px / (2.0 * plot_w)
    box_hh = box_h_px / (2.0 * plot_h)
    pad_x = 10.0 / plot_w
    pad_y = 8.0 / plot_h

    if y_norm > 0.84:
        ay_opts = [16, 32, 50, 68, 0, -14]
    elif y_norm > 0.74:
        ay_opts = [-16, 16, -32, 32, 50, 68]
    elif y_norm < 0.18:
        ay_opts = [-16, -32, -50, -68]
    else:
        ay_opts = [-16, 16, -32, 32, -50, 50, 68]

    if x_norm > 0.80:
        ax_opts = [-12, -26, -44, -62, 0]
    elif x_norm < 0.18:
        ax_opts = [12, 26, 44, 62, 0]
    else:
        ax_opts = [0, -14, 14, -32, 32, -52, 52]

    ref_norm = (ref_y - float(y_range[0])) / y_span if ref_y is not None else None

    best_score = float("inf")
    best_pair = (ax_opts[0], ay_opts[0])
    best_center = (x_norm, y_norm)
    for cand_ax in ax_opts:
        for cand_ay in ay_opts:
            if cand_ax == 0 and cand_ay == 0:
                continue
            xanchor, yanchor = _resolve_annotation_anchor(cand_ax, cand_ay)
            cx = (
                x_norm
                + cand_ax / plot_w
                + (box_hw if xanchor == "left" else (-box_hw if xanchor == "right" else 0.0))
            )
            cy = (
                y_norm
                - cand_ay / plot_h
                + (box_hh if yanchor == "bottom" else (-box_hh if yanchor == "top" else 0.0))
            )

            score = 0.0
            if cy + box_hh > 0.98:
                score += 260.0 + (cy + box_hh - 0.98) * 700.0
            if cy - box_hh < 0.03:
                score += 260.0 + (0.03 - (cy - box_hh)) * 700.0
            if cx - box_hw < 0.01:
                score += 220.0 + (0.01 - (cx - box_hw)) * 550.0
            if cx + box_hw > 0.99:
                score += 220.0 + (cx + box_hw - 0.99) * 550.0

            hits = np.sum(
                (np.abs(xs_norm - cx) <= box_hw + 0.010) & (np.abs(ys_norm - cy) <= box_hh + 0.010)
            )
            score += float(hits) * 8.0

            if ref_norm is not None and cx > 0.56 and abs(cy - (ref_norm - 0.045)) < 0.085:
                score += 90.0

            for pcx, pcy, phw, phh in placed:
                ox = (box_hw + phw + pad_x) - abs(cx - pcx)
                oy = (box_hh + phh + pad_y) - abs(cy - pcy)
                if ox > 0 and oy > 0:
                    score += 1800.0 + (ox + oy) * 1000.0

            score += (abs(cand_ax) * 0.022 + abs(cand_ay) * 0.038)

            if score < best_score:
                best_score = score
                best_pair = (cand_ax, cand_ay)
                best_center = (cx, cy)

    chosen_ax, chosen_ay = best_pair
    placed.append((best_center[0], best_center[1], box_hw, box_hh))
    return chosen_ax, chosen_ay


def _annotation_style(ax_px: int, ay_px: int, text: str) -> dict[str, Any]:
    xanchor, yanchor = _resolve_annotation_anchor(ax_px, ay_px)
    return dict(
        text=text,
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        arrowcolor=T.DATA_NEUTRAL,
        ax=ax_px,
        ay=ay_px,
        xanchor=xanchor,
        yanchor=yanchor,
        font=dict(size=10.5, color=TEXT_COLOR),
        bgcolor="rgba(255, 255, 255, 0.96)",
        bordercolor=T.DATA_GRAY_3,
        borderwidth=1,
        borderpad=3,
    )


def build_time_performance_chart(
    df: pd.DataFrame,
    bench_col: str,
    reg: dict[str, Any],
    show_reg: bool,
    show_pi: bool,
    x_range: list[float],
    random_baseline: float = 0.25,
) -> go.Figure:
    """Tab 1 top wide chart: Date -> MMLU or BBH (raw 0~1 scale)."""
    sub = df.dropna(subset=["year_frac", bench_col]).sort_values("year_frac").copy()

    fig = go.Figure()
    if show_reg and reg.get("ok"):
        _add_regression_layers(fig, reg, show_pi=show_pi, x_is_log10=False, y_is_log10=False, show_band_legend=True)

    for acc_label in ACCESS_OPTIONS:
        part = sub[sub["_access"] == acc_label]
        if part.empty:
            continue
        st_cfg = ACCESS_STYLE[acc_label]
        custom = np.stack(
            [
                part["_system"].to_numpy(),
                part["_date_str"].to_numpy(),
                part["_access"].to_numpy(),
                [_fmt_scale_value(v, "log10_flop") for v in part["_compute_flop"]],
                [_fmt_scale_value(v, "log10_params") for v in part["_active_params"]],
                [_fmt_scale_value(v, "log10_dataset") for v in part["_dataset_size"]],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=part["year_frac"],
                y=part[bench_col],
                mode="markers",
                name=f"{acc_label} (n={len(part)})",
                marker=dict(
                    color=st_cfg["fill"],
                    symbol=st_cfg["symbol"],
                    size=9.5,
                    line=dict(color=st_cfg["line"], width=1.5),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[2]})<br>"
                    "Date: %{customdata[1]}<br>"
                    f"{bench_col}: <b>%{{y:.3f}}</b><br>"
                    "Training compute (FLOP): %{customdata[3]}<br>"
                    "Active Parameters: %{customdata[4]}<br>"
                    "Dataset size: %{customdata[5]}<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=random_baseline,
        line_dash="dot",
        line_color=AXIS_LINE_COLOR,
        line_width=1.3,
        annotation_text=f"Random chance ({random_baseline:.2f})",
        annotation_position="bottom right",
        annotation_font=dict(size=12, color=MUTED_COLOR),
    )

    y_range = compute_shared_perf_y_range(sub, bench_col, [reg], show_reg=show_reg, show_pi=show_pi)

    landmarks = _select_yearly_top_models(sub, bench_col)
    xs_arr = sub["year_frac"].to_numpy(dtype=float)
    ys_arr = sub[bench_col].to_numpy(dtype=float)
    placed: list[tuple[float, float, float, float]] = []
    for _, row in landmarks.iterrows():
        x_val = float(row["year_frac"])
        y_val = float(row[bench_col])
        label_txt = str(row["_system"])
        ax_px, ay_px = _compute_landmark_offset(
            xs_arr, ys_arr, x_val, y_val, x_range, y_range, placed, ref_y=random_baseline, label_text=label_txt
        )
        fig.add_annotation(x=x_val, y=y_val, **_annotation_style(ax_px, ay_px, label_txt))

    _base_layout(fig, height=420, margin=dict(l=80, r=28, t=44, b=44), show_legend=True)
    fig.update_xaxes(
        title_text="Date",
        range=x_range,
        dtick=1,
        tickformat="d",
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=X_GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=bench_col,
        range=y_range,
        tickvals=[0.0, 0.25, 0.50, 0.75, 1.00],
        tickformat=".2f",
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_time_scale_chart(
    df: pd.DataFrame,
    scale_col: str,
    reg: dict[str, Any],
    show_reg: bool,
    show_pi: bool,
    x_range: list[float],
) -> go.Figure:
    """Tab 1 bottom toggle chart: Date -> Training compute / Active Parameters / Dataset size (log Y)."""
    meta = SCALE_META[scale_col]
    num_col = meta["num_col"]
    sub = df.dropna(subset=["year_frac", num_col]).copy()
    sub = sub[sub[num_col] > 0].sort_values("year_frac")

    fig = go.Figure()
    if show_reg and reg.get("ok"):
        _add_regression_layers(fig, reg, show_pi=show_pi, x_is_log10=False, y_is_log10=True, show_band_legend=True)

    for acc_label in ACCESS_OPTIONS:
        part = sub[sub["_access"] == acc_label]
        if part.empty:
            continue
        st_cfg = ACCESS_STYLE[acc_label]
        custom = np.stack(
            [
                part["_system"].to_numpy(),
                part["_date_str"].to_numpy(),
                part["_access"].to_numpy(),
                [_fmt_scale_value(v, scale_col) for v in part[num_col]],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=part["year_frac"],
                y=part[num_col],
                mode="markers",
                name=f"{acc_label} (n={len(part)})",
                showlegend=True,
                marker=dict(
                    color=st_cfg["fill"],
                    symbol=st_cfg["symbol"],
                    size=9.5,
                    line=dict(color=st_cfg["line"], width=1.5),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[2]})<br>"
                    "Date: %{customdata[1]}<br>"
                    f"{meta['raw_col']}: <b>%{{customdata[3]}}</b><extra></extra>"
                ),
            )
        )

    tickvals, ticktext, y_log_range = _log_axis_ticks(sub[num_col], scale_col)
    if show_reg and reg.get("ok"):
        y_log_range = [
            min(y_log_range[0], float(np.nanmin(reg["mean_ci_low"])) - 0.15),
            max(y_log_range[1], float(np.nanmax(reg["mean_ci_high"])) + 0.15),
        ]
        if show_pi:
            y_log_range = [
                min(y_log_range[0], float(np.nanmin(reg["obs_ci_low"])) - 0.15),
                max(y_log_range[1], float(np.nanmax(reg["obs_ci_high"])) + 0.15),
            ]

    landmarks = _select_yearly_top_models(sub, num_col)
    xs_arr = sub["year_frac"].to_numpy(dtype=float)
    ys_log_arr = np.log10(sub[num_col].to_numpy(dtype=float))
    placed: list[tuple[float, float, float, float]] = []
    for _, row in landmarks.iterrows():
        x_val = float(row["year_frac"])
        y_log = math.log10(float(row[num_col]))
        label_txt = str(row["_system"])
        ax_px, ay_px = _compute_landmark_offset(
            xs_arr, ys_log_arr, x_val, y_log, x_range, y_log_range, placed, ref_y=None, label_text=label_txt
        )
        fig.add_annotation(x=x_val, y=y_log, **_annotation_style(ax_px, ay_px, label_txt))

    _base_layout(fig, height=420, margin=dict(l=80, r=28, t=44, b=44), show_legend=True)
    fig.update_xaxes(
        title_text="Date",
        range=x_range,
        dtick=1,
        tickformat="d",
        tickangle=0,
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=X_GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    y_axis_title = meta["axis_label"].replace(" (", "<br>(", 1)
    fig.update_yaxes(
        title=dict(text=y_axis_title, standoff=8, font=dict(size=12.5, color=TEXT_COLOR)),
        type="log",
        tickvals=tickvals,
        ticktext=ticktext,
        range=y_log_range,
        tickfont=dict(size=12, color=MUTED_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_direct_scaling_chart(
    df: pd.DataFrame,
    scale_col: str,
    bench_col: str,
    reg: dict[str, Any],
    show_reg: bool,
    show_pi: bool,
    y_range: list[float],
    random_baseline: float = 0.25,
    show_y_title: bool = True,
) -> go.Figure:
    """Tab 3 direct scaling scatter: log X -> MMLU/BBH (shared raw 0~1 Y)."""
    s_meta = SCALE_META[scale_col]
    num_col = s_meta["num_col"]
    sub = df.dropna(subset=[num_col, bench_col]).copy()
    sub = sub[sub[num_col] > 0].sort_values(num_col)

    fig = go.Figure()
    if show_reg and reg.get("ok"):
        _add_regression_layers(fig, reg, show_pi=show_pi, x_is_log10=True, y_is_log10=False, show_band_legend=True)

    for acc_label in ACCESS_OPTIONS:
        part = sub[sub["_access"] == acc_label]
        if part.empty:
            continue
        st_cfg = ACCESS_STYLE[acc_label]
        custom = np.stack(
            [
                part["_system"].to_numpy(),
                part["_date_str"].to_numpy(),
                part["_access"].to_numpy(),
                [_fmt_scale_value(v, scale_col) for v in part[num_col]],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=part[num_col],
                y=part[bench_col],
                mode="markers",
                name=f"{acc_label} (n={len(part)})",
                showlegend=True,
                marker=dict(
                    color=st_cfg["fill"],
                    symbol=st_cfg["symbol"],
                    size=9.5,
                    line=dict(color=st_cfg["line"], width=1.5),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[2]})<br>"
                    "Date: %{customdata[1]}<br>"
                    f"{s_meta['raw_col']}: <b>%{{customdata[3]}}</b><br>"
                    f"{bench_col}: <b>%{{y:.3f}}</b><extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=random_baseline,
        line_dash="dot",
        line_color=AXIS_LINE_COLOR,
        line_width=1.3,
        annotation_text=f"Random chance ({random_baseline:.2f})",
        annotation_position="bottom right",
        annotation_font=dict(size=12, color=MUTED_COLOR),
    )

    tickvals, ticktext, x_log_range = _log_axis_ticks(sub[num_col], scale_col)
    landmarks = _select_yearly_top_models(sub, bench_col)
    xs_log_arr = np.log10(sub[num_col].to_numpy(dtype=float))
    ys_arr = sub[bench_col].to_numpy(dtype=float)
    placed: list[tuple[float, float, float, float]] = []
    for _, row in landmarks.iterrows():
        x_log = math.log10(float(row[num_col]))
        y_val = float(row[bench_col])
        label_txt = str(row["_system"])
        ax_px, ay_px = _compute_landmark_offset(
            xs_log_arr, ys_arr, x_log, y_val, x_log_range, y_range, placed, ref_y=random_baseline, label_text=label_txt
        )
        fig.add_annotation(x=x_log, y=y_val, **_annotation_style(ax_px, ay_px, label_txt))

    _base_layout(fig, height=430, margin=dict(l=72, r=28, t=46, b=48), show_legend=True)
    fig.update_xaxes(
        title_text=s_meta["axis_label"],
        type="log",
        tickvals=tickvals,
        ticktext=ticktext,
        range=x_log_range,
        tickangle=0,
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=X_GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=bench_col if show_y_title else "",
        range=y_range,
        tickvals=[0.0, 0.25, 0.50, 0.75, 1.00],
        tickformat=".2f",
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_multi_actual_vs_predicted_chart(
    reg: dict[str, Any],
    bench_col: str,
) -> go.Figure:
    """Tab 4 Multiple OLS Actual (y) vs Predicted (y_hat) scatter with 45-degree line."""
    line = reg["line"]
    y_pred = np.asarray(line["fitted"], dtype=float)
    y_act = np.asarray(reg["actual"], dtype=float)
    resid = np.asarray(line["resid"], dtype=float)
    models = np.asarray(line["models_ordered"], dtype=object)
    dates = np.asarray(line["dates_ordered"], dtype=object)
    access = np.asarray(reg["access_ordered"], dtype=object)

    lo = min(float(np.min(y_pred)), float(np.min(y_act)), 0.15) - 0.04
    hi = max(float(np.max(y_pred)), float(np.max(y_act)), 0.95) + 0.04
    lo = max(0.0, lo)
    hi = min(1.06, hi)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y = ŷ (완전 일치)",
            line=dict(color=REG_LINE_COLOR, width=2.0, dash="dash"),
            hoverinfo="skip",
        )
    )

    for acc_label in ACCESS_OPTIONS:
        mask = access == acc_label
        if not np.any(mask):
            continue
        st_cfg = ACCESS_STYLE[acc_label]
        custom = np.stack(
            [models[mask], dates[mask], access[mask], resid[mask]],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=y_pred[mask],
                y=y_act[mask],
                mode="markers",
                name=f"{acc_label} (n={int(np.sum(mask))})",
                marker=dict(
                    color=st_cfg["fill"],
                    symbol=st_cfg["symbol"],
                    size=9.5,
                    line=dict(color=st_cfg["line"], width=1.5),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[2]})<br>"
                    "Date: %{customdata[1]}<br>"
                    f"예측 {bench_col} (ŷ): %{{x:.3f}}<br>"
                    f"실측 {bench_col} (y): <b>%{{y:.3f}}</b><br>"
                    "잔차 (y - ŷ): %{customdata[3]:+.3f}<extra></extra>"
                ),
            )
        )

    _base_layout(fig, height=410, margin=dict(l=72, r=28, t=44, b=48), show_legend=True)
    fig.update_xaxes(
        title_text=f"예측 {bench_col} (ŷ)",
        range=[lo, hi],
        tickformat=".2f",
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=X_GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=f"실측 {bench_col} (y)",
        range=[lo, hi],
        tickformat=".2f",
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_residual_plot(line: dict[str, Any], y_label: str) -> go.Figure:
    """Tab 4 Week 7 Residual vs Fitted scatter (raw 0~1 scale, non-clipping range)."""
    fitted = line["fitted"]
    resid = line["resid"]
    models = line["models_ordered"]
    dates = line["dates_ordered"]
    sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=T.DATA_GRAY_5, line_width=1.4)
    if sd > 0:
        fig.add_hline(y=2 * sd, line_dash="dot", line_color=T.DATA_NEUTRAL, line_width=1.0)
        fig.add_hline(y=-2 * sd, line_dash="dot", line_color=T.DATA_NEUTRAL, line_width=1.0)

    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=resid,
            mode="markers",
            marker=dict(color=T.DATA_OPEN_FILL, size=9.0, line=dict(color=T.DATA_OPEN, width=1.4)),
            customdata=np.stack([models, dates], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Fitted: %{x:.3f}<br>Residual: %{y:+.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    max_abs = float(np.max(np.abs(resid))) if len(resid) else 0.1
    r_pad = max(max_abs * 0.18, 0.02)
    r_lim = max(max_abs + r_pad, 2.2 * sd, 0.05)

    x_lo = float(np.min(fitted)) if len(fitted) else 0.0
    x_hi = float(np.max(fitted)) if len(fitted) else 1.0
    x_pad = max((x_hi - x_lo) * 0.08, 0.02)

    _base_layout(fig, height=390, margin=dict(l=64, r=24, t=28, b=48), show_legend=False)
    fig.update_xaxes(
        title_text=f"Fitted ({y_label})",
        range=[x_lo - x_pad, x_hi + x_pad],
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=X_GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=f"Residual ({y_label})",
        range=[-r_lim, r_lim],
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_qq_plot(line: dict[str, Any]) -> go.Figure:
    """Tab 4 Week 7 Normal Q-Q plot of standardized residuals in pure Plotly."""
    theo = line["qq_theo"]
    obs = line["qq_obs"]
    models_qq = line["models_qq"]

    lo = float(min(np.min(theo), np.min(obs))) - 0.35 if len(theo) else -3.0
    hi = float(max(np.max(theo), np.max(obs))) + 0.35 if len(theo) else 3.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color=T.DATA_CLOSED_DARK, width=2.0),
            name="y = x",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=theo,
            y=obs,
            mode="markers",
            marker=dict(color=T.DATA_OPEN_FILL, size=9.0, line=dict(color=T.DATA_OPEN, width=1.4)),
            customdata=models_qq,
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Theoretical Quantile: %{x:.2f}<br>Standardized Residual: %{y:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    _base_layout(fig, height=390, margin=dict(l=64, r=24, t=28, b=48), show_legend=False)
    fig.update_xaxes(
        title_text="Theoretical Quantiles",
        range=[lo, hi],
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=X_GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Standardized Residuals",
        range=[lo, hi],
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_time_residual_plot(line: dict[str, Any], y_label: str) -> go.Figure:
    """Tab 4 Week 7 Independence (I) time-ordered residual plot with daily mean trend."""
    dates_str = line["dates_ordered"]
    dt = pd.to_datetime(dates_str)
    resid = line["resid"]
    models = line["models_ordered"]
    sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=T.DATA_GRAY_5, line_width=1.4)
    if sd > 0:
        fig.add_hline(y=2 * sd, line_dash="dot", line_color=T.DATA_NEUTRAL, line_width=1.0)
        fig.add_hline(y=-2 * sd, line_dash="dot", line_color=T.DATA_NEUTRAL, line_width=1.0)

    if len(dt) > 0 and dt.nunique() >= 2:
        daily = (
            pd.DataFrame({"d": dt, "r": np.asarray(resid, dtype=float)})
            .groupby("d", sort=True)["r"]
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily.values,
                mode="lines",
                line=dict(color="rgba(93, 108, 123, 0.45)", width=1.3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dt,
            y=resid,
            mode="markers",
            marker=dict(color=T.DATA_OPEN_FILL, size=9.0, line=dict(color=T.DATA_OPEN, width=1.4)),
            customdata=np.stack([models, dates_str], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Residual: %{y:+.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    max_abs = float(np.max(np.abs(resid))) if len(resid) else 0.1
    r_pad = max(max_abs * 0.18, 0.02)
    r_lim = max(max_abs + r_pad, 2.2 * sd, 0.05)

    _base_layout(fig, height=390, margin=dict(l=64, r=24, t=28, b=48), show_legend=False)
    x_kwargs: dict[str, Any] = dict(
        title_text="Date",
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=X_GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    if len(dt) > 0:
        span_td = dt.max() - dt.min()
        pad = max(span_td * 0.06, pd.Timedelta(days=15))
        x_kwargs["range"] = [dt.min() - pad, dt.max() + pad]
    fig.update_xaxes(**x_kwargs)
    fig.update_yaxes(
        title_text=f"Residual ({y_label})",
        range=[-r_lim, r_lim],
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_open_closed_bar_chart(
    df: pd.DataFrame,
    yr_min: int,
    yr_max: int,
) -> go.Figure:
    """Tab 2 yearly Open vs Closed model counts grouped bar with top labels and right legend."""
    years = list(range(int(yr_min), int(yr_max) + 1))
    active_access = [a for a in ACCESS_OPTIONS if (df["_access"] == a).any()]

    fig = go.Figure()
    max_val = 0
    for acc_label in active_access:
        counts = [int(((df["_year"] == y) & (df["_access"] == acc_label)).sum()) for y in years]
        if counts:
            max_val = max(max_val, max(counts))
        st_cfg = ACCESS_STYLE[acc_label]
        fig.add_trace(
            go.Bar(
                x=years,
                y=counts,
                name=f"{acc_label} ({sum(counts)})",
                marker=dict(color=st_cfg["color"], line=dict(color=T.COLOR_BG_CARD, width=0.9)),
                text=[str(c) if c > 0 else "" for c in counts],
                textposition="outside",
                textfont=dict(size=12, color=TEXT_COLOR),
                cliponaxis=False,
                hovertemplate=f"<b>%{{x}} · {acc_label}</b><br>Models: <b>%{{y}}</b><extra></extra>",
            )
        )

    _base_layout(
        fig,
        height=410,
        margin=dict(l=64, r=122, t=28, b=46, autoexpand=True),
        show_legend=True,
    )
    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.10,
        hovermode="x unified",
        legend=dict(
            title=dict(text="Open/Closed", font=dict(size=12, color=TEXT_COLOR)),
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.92)",
            font=dict(size=12, color=TEXT_COLOR),
        ),
    )
    fig.update_xaxes(
        title_text="Date",
        tickmode="array",
        tickvals=years,
        ticktext=[str(y) for y in years],
        tickangle=0,
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=False,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="모델 수 (개)",
        rangemode="tozero",
        range=[0, max(max_val * 1.22, 4)],
        tickfont=dict(size=12, color=MUTED_COLOR),
        title_font=dict(size=12.5, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_open_closed_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Tab 2 Open vs Closed overall share pie chart with right-side legend."""
    labels = [a for a in ACCESS_OPTIONS if (df["_access"] == a).any()]
    values = [int((df["_access"] == a).sum()) for a in labels]
    colors = [ACCESS_STYLE[a]["color"] for a in labels]
    inside_colors = [T.COLOR_ON_DARK if lbl == "Open" else T.COLOR_ON_LIGHT for lbl in labels]
    slice_border_w = 0.0 if len(labels) <= 1 else 2.0

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            sort=False,
            marker=dict(colors=colors, line=dict(color=T.COLOR_BG_CARD, width=slice_border_w)),
            texttemplate="<b>%{label}</b><br>%{percent:.1%}",
            textposition="inside",
            insidetextorientation="horizontal",
            insidetextfont=dict(size=13.5, color=inside_colors),
            hovertemplate="<b>%{label}</b><br>Models: <b>%{value}</b> (%{percent:.1%})<extra></extra>",
        )
    )
    _base_layout(
        fig,
        height=410,
        margin=dict(l=20, r=108, t=28, b=28, autoexpand=True),
        show_legend=True,
    )
    fig.update_layout(
        legend=dict(
            title=dict(text="Open/Closed", font=dict(size=12, color=TEXT_COLOR)),
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.92)",
            font=dict(size=12, color=TEXT_COLOR),
        )
    )
    return fig


# ── Tab 1 (01 · 원자료): Dataset Column Hierarchy Sunburst & Coverage Charts ──
RAW_COLUMN_GROUPS: list[dict[str, str]] = [
    {
        "key": "meta",
        "label": "기본 · 메타",
        "color": T.DATA_REG_LINE,   # #02165E (16.2:1 white text)
        "tint": "#DBE4F6",          # soft navy tint (14:1+ dark text)
        "criteria": "모델 식별·출시 시점·접근성 그룹·신뢰 플래그 (결측 0%, 100% 완비)",
    },
    {
        "key": "scale",
        "label": "규모 변수 (X)",
        "color": T.DATA_OPEN,       # #0171E3 (4.59:1 white text)
        "tint": "#D6EAFF",          # soft blue tint
        "criteria": "스케일링 법칙 투입 자원(연산량 C · 파라미터 N · 데이터 D) 수치형 독립변수",
    },
    {
        "key": "bench_used",
        "label": "채택 벤치마크 (Y)",
        "color": "#0F766E",         # deep teal (5.4:1 white text)
        "tint": "#CCFBF1",          # soft mint tint
        "criteria": "성능 지표 중 유효 관측률 50% 이상을 충족해 회귀 종속변수로 채택",
    },
    {
        "key": "bench_unused",
        "label": "미채택 벤치마크",
        "color": "#B84D00",         # deep burnt orange (5.3:1 white text)
        "tint": "#FFE6CF",          # soft peach tint
        "criteria": "성능 지표이나 유효 관측률 50% 미만으로 규모 축 교차 시 표본 부족",
    },
    {
        "key": "notes",
        "label": "메모 · 주석",
        "color": "#475569",         # cool slate (7.5:1 white text)
        "tint": "#E2E8F0",          # soft slate tint
        "criteria": "수치 변수가 아닌 출처·평가 조건·추정 근거·신뢰구간 텍스트 주석 열",
    },
]

RAW_COLUMN_META: dict[str, dict[str, str]] = {
    "System": {
        "group": "meta",
        "short": "System",
        "dtype": "텍스트",
        "usage": "채택 · 모델 식별자 (1행=1모델)",
        "desc": "AI 모델 명칭",
    },
    "Date": {
        "group": "meta",
        "short": "Date",
        "dtype": "날짜",
        "usage": "채택 · 출시 시점 X축",
        "desc": "모델 최초 공개일 (YYYY-MM-DD)",
    },
    "Open/Closed": {
        "group": "meta",
        "short": "Open/Cls",
        "dtype": "범주(2)",
        "usage": "채택 · 접근성 그룹 구분",
        "desc": "가중치 공개 여부 (Open / Closed)",
    },
    "Trust in benchmark results": {
        "group": "meta",
        "short": "Trust",
        "dtype": "정수(0/1)",
        "usage": "참고 · 원자료 신뢰 플래그",
        "desc": "벤치마크 오염·신뢰 여부 지표",
    },
    "Training compute (FLOP)": {
        "group": "scale",
        "short": "FLOP",
        "dtype": "수치(지수)",
        "usage": "채택 · 주 독립변수 X₁",
        "desc": "사전학습 연산량 (FLOP)",
    },
    "Active Parameters": {
        "group": "scale",
        "short": "Active P.",
        "dtype": "수치(정수)",
        "usage": "채택 · 주 독립변수 X₂",
        "desc": "추론 시 활성 파라미터 수",
    },
    "Model size (parameters)": {
        "group": "scale",
        "short": "Model size",
        "dtype": "수치(정수)",
        "usage": "보조 비교 · 총 파라미터",
        "desc": "전체 파라미터 수 (단일회귀 보조 비교)",
    },
    "Dataset size": {
        "group": "scale",
        "short": "Dataset",
        "dtype": "수치(정수)",
        "usage": "채택 · 주 독립변수 X₃",
        "desc": "사전학습 토큰/데이터 규모",
    },
    "MMLU": {
        "group": "bench_used",
        "short": "MMLU",
        "dtype": "수치(0~1)",
        "usage": "채택 · 주 종속변수 Y₁",
        "desc": "다분야 객관식 추론 정확도 (유효율 1위)",
    },
    "BBH": {
        "group": "bench_used",
        "short": "BBH",
        "dtype": "수치(0~1)",
        "usage": "채택 · 주 종속변수 Y₂",
        "desc": "고난도 추론(BIG-Bench Hard) 정확도",
    },
    "GPQA": {
        "group": "bench_unused",
        "short": "GPQA",
        "dtype": "수치(0~1)",
        "usage": "미채택 · 유효율 50% 미만",
        "desc": "대학원 수준 과학 객관식 정확도",
    },
    "LMSys Elo": {
        "group": "bench_unused",
        "short": "LMSys Elo",
        "dtype": "수치(점수)",
        "usage": "미채택 · 유효율 50% 미만",
        "desc": "Chatbot Arena 인간 선호 Elo 레이팅",
    },
    "SEAL Coding": {
        "group": "bench_unused",
        "short": "SEAL Code",
        "dtype": "수치(점수)",
        "usage": "미채택 · 표본 극소 (<10%)",
        "desc": "Scale AI 코딩 평가 점수",
    },
    "SEAL Instruction Following": {
        "group": "bench_unused",
        "short": "SEAL Inst",
        "dtype": "수치(0~1)",
        "usage": "미채택 · 표본 극소 (<10%)",
        "desc": "Scale AI 지시 이행 평가",
    },
    "SEAL Math": {
        "group": "bench_unused",
        "short": "SEAL Math",
        "dtype": "수치(0~1)",
        "usage": "미채택 · 표본 극소 (<10%)",
        "desc": "Scale AI 수학 추론 평가",
    },
    "HELM MMLU": {
        "group": "bench_unused",
        "short": "HELM",
        "dtype": "수치(0~1)",
        "usage": "미채택 · 표본 극소 (<10%)",
        "desc": "Stanford HELM 표준화 MMLU",
    },
    "GPQA Notes": {
        "group": "notes",
        "short": "GPQA Nt",
        "dtype": "텍스트",
        "usage": "참고 · 평가 출처 메모",
        "desc": "GPQA 평가 방식 주석",
    },
    "LMSys Elo Notes": {
        "group": "notes",
        "short": "Elo Nt",
        "dtype": "텍스트",
        "usage": "참고 · 수집일 메모",
        "desc": "LMSys 리더보드 스냅샷 기준일",
    },
    "Trust notes": {
        "group": "notes",
        "short": "Trust Nt",
        "dtype": "텍스트",
        "usage": "참고 · 오염 검증 메모",
        "desc": "GSM1k 등 벤치마크 오염 검증 근거",
    },
    "Training compute notes": {
        "group": "notes",
        "short": "FLOP Nt",
        "dtype": "텍스트",
        "usage": "참고 · 연산량 추정 메모",
        "desc": "비공개 모델 연산량 추정 근거",
    },
    "MMLU Notes": {
        "group": "notes",
        "short": "MMLU Nt",
        "dtype": "텍스트",
        "usage": "참고 · 평가 설정 메모",
        "desc": "few-shot / CoT 등 MMLU 조건 주석",
    },
    "LMSys Elo 95% CI": {
        "group": "notes",
        "short": "Elo CI",
        "dtype": "텍스트",
        "usage": "참고 · 신뢰구간 표기",
        "desc": "LMSys Elo 95% 신뢰구간 문자열",
    },
    "HELM MMLU Notes": {
        "group": "notes",
        "short": "HELM Nt",
        "dtype": "텍스트",
        "usage": "참고 · 수집일 메모",
        "desc": "HELM 평가 기준일 주석",
    },
    "BBH Notes": {
        "group": "notes",
        "short": "BBH Nt",
        "dtype": "텍스트",
        "usage": "참고 · 평가 설정 메모",
        "desc": "3-shot 등 BBH 평가 조건 주석",
    },
}

RAW_KEY_COMPARE_COLS: list[tuple[str, str]] = [
    ("Training compute (FLOP)", "Training compute (FLOP)"),
    ("Active Parameters", "Active Parameters"),
    ("Model size (parameters)", "Model size (parameters)"),
    ("Dataset size", "Dataset size"),
    ("MMLU", "MMLU"),
    ("BBH", "BBH"),
    ("GPQA", "GPQA"),
    ("LMSys Elo", "LMSys Elo"),
]


def _col_valid_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    ser = df[col]
    return ser.notna() & ser.astype(str).str.strip().ne("")


def build_dataset_column_sunburst(df: pd.DataFrame, raw_cols: list[str]) -> go.Figure:
    """Interactive 3-ring Dataset Column Wheel (Role 5 groups -> 24 columns -> Open/Closed/Missing on click)."""
    total_n = len(df)
    if total_n == 0 or not raw_cols:
        fig = go.Figure()
        _base_layout(fig, height=480, margin=dict(l=8, r=8, t=12, b=12), show_legend=False)
        return fig

    group_lookup = {g["key"]: g for g in RAW_COLUMN_GROUPS}
    cols_by_group: dict[str, list[str]] = {g["key"]: [] for g in RAW_COLUMN_GROUPS}
    for col in raw_cols:
        gkey = RAW_COLUMN_META.get(col, {}).get("group", "notes")
        cols_by_group.setdefault(gkey, []).append(col)

    is_open = df["_access"] == "Open"
    is_closed = df["_access"] == "Closed"
    n_open_all = int(is_open.sum())
    n_closed_all = int(is_closed.sum())

    ids: list[str] = ["root"]
    labels: list[str] = [f"원자료<br>{len(raw_cols)}컬럼"]
    parents: list[str] = [""]
    values: list[int] = [total_n * len(raw_cols)]
    colors: list[str] = [T.COLOR_BG_BODY]
    hover_texts: list[str] = [
        f"<b>benchmarks_with_model_accessibility.csv</b><br>"
        f"총 컬럼: <b>{len(raw_cols)}개</b> · 현재 필터 모델: <b>{total_n}종</b><br>"
        f"Open {n_open_all}종 · Closed {n_closed_all}종<br>"
        f"조각을 클릭하면 하위 구성(Open / Closed / 결측)으로 확대됩니다."
    ]

    for g in RAW_COLUMN_GROUPS:
        gkey = g["key"]
        gcols = cols_by_group.get(gkey, [])
        if not gcols:
            continue
        # Sort columns within each group by valid count descending for neat visual rhythm
        gcols_sorted = sorted(gcols, key=lambda c: (-int(_col_valid_mask(df, c).sum()), c))
        gid = f"grp::{gkey}"
        avg_cov = float(np.mean([_col_valid_mask(df, c).mean() * 100.0 for c in gcols_sorted]))

        ids.append(gid)
        labels.append(f"{g['label']}<br>({len(gcols_sorted)})")
        parents.append("root")
        values.append(total_n * len(gcols_sorted))
        colors.append(g["color"])
        hover_texts.append(
            f"<b>{g['label']}</b> ({len(gcols_sorted)}개 컬럼)<br>"
            f"분류 기준: {g['criteria']}<br>"
            f"그룹 평균 유효율: <b>{avg_cov:.1f}%</b><br>"
            f"포함 컬럼: {', '.join(gcols_sorted)}<br>"
            f"<i>클릭 시 그룹 확대</i>"
        )

        for col in gcols_sorted:
            meta = RAW_COLUMN_META.get(
                col,
                {"short": col, "dtype": "기타", "usage": "참고", "desc": col},
            )
            valid_mask = _col_valid_mask(df, col)
            n_valid = int(valid_mask.sum())
            pct_valid = 100.0 * n_valid / total_n
            n_open_v = int((valid_mask & is_open).sum())
            n_closed_v = int((valid_mask & is_closed).sum())
            n_missing = total_n - n_valid
            open_rate = (100.0 * n_open_v / n_open_all) if n_open_all else 0.0
            closed_rate = (100.0 * n_closed_v / n_closed_all) if n_closed_all else 0.0

            cid = f"col::{col}"
            ids.append(cid)
            labels.append(f"{meta['short']} {pct_valid:.0f}%")
            parents.append(gid)
            values.append(total_n)
            colors.append(g["tint"])
            hover_texts.append(
                f"<b>{col}</b> ({meta['dtype']})<br>"
                f"역할: {g['label']} · {meta['usage']}<br>"
                f"설명: {meta['desc']}<br>"
                f"유효 표본: <b>{n_valid}/{total_n}개 ({pct_valid:.1f}%)</b> · 결측 {n_missing}개<br>"
                f"Open 공개율: {open_rate:.1f}% ({n_open_v}/{n_open_all}) · "
                f"Closed 공개율: {closed_rate:.1f}% ({n_closed_v}/{n_closed_all})<br>"
                f"<i>클릭 시 Open / Closed / 결측 비율 확대</i>"
            )

            leaf_items = [
                ("open", "Open 유효", n_open_v, T.DATA_OPEN),
                ("closed", "Closed 유효", n_closed_v, T.DATA_CLOSED),
                ("missing", "결측 (NaN)", n_missing, "#CBD5E1"),
            ]
            for lkey, llabel, lcnt, lcolor in leaf_items:
                if lcnt <= 0:
                    continue
                lpct = 100.0 * lcnt / total_n
                ids.append(f"leaf::{col}::{lkey}")
                labels.append(f"{llabel}<br>{lcnt} ({lpct:.0f}%)")
                parents.append(cid)
                values.append(lcnt)
                colors.append(lcolor)
                hover_texts.append(
                    f"<b>{col} · {llabel}</b><br>"
                    f"모델 수: <b>{lcnt}/{total_n}개 ({lpct:.1f}%)</b>"
                )

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            maxdepth=3,
            sort=False,
            leaf=dict(opacity=0.95),
            insidetextorientation="radial",
            textinfo="label",
            textfont=dict(family=FONT_FAMILY, size=10.5),
            marker=dict(colors=colors, line=dict(color=T.COLOR_BG_CARD, width=1.5)),
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    _base_layout(fig, height=540, margin=dict(l=8, r=8, t=10, b=10), show_legend=False)
    fig.update_layout(uniformtext=dict(minsize=8, mode="hide"))
    return fig


def build_dataset_coverage_bar(df: pd.DataFrame, raw_cols: list[str]) -> go.Figure:
    """Horizontal bar chart of valid non-null coverage (%) for all 24 raw columns, colored by role."""
    total_n = len(df)
    if total_n == 0 or not raw_cols:
        fig = go.Figure()
        _base_layout(fig, height=500, margin=dict(l=20, r=20, t=20, b=20), show_legend=False)
        return fig

    ordered_rows: list[dict[str, Any]] = []
    for g in RAW_COLUMN_GROUPS:
        gcols = [c for c in raw_cols if RAW_COLUMN_META.get(c, {}).get("group", "notes") == g["key"]]
        gcols_sorted = sorted(gcols, key=lambda c: (-int(_col_valid_mask(df, c).sum()), c))
        for col in gcols_sorted:
            n_val = int(_col_valid_mask(df, col).sum())
            pct = 100.0 * n_val / total_n
            meta = RAW_COLUMN_META.get(col, {"usage": "참고", "desc": col, "dtype": "기타"})
            ordered_rows.append(
                {
                    "col": col,
                    "group_key": g["key"],
                    "group_label": g["label"],
                    "color": g["color"],
                    "n_valid": n_val,
                    "pct": pct,
                    "usage": meta["usage"],
                    "desc": meta["desc"],
                }
            )

    y_order = [r["col"] for r in reversed(ordered_rows)]
    fig = go.Figure()
    for g in RAW_COLUMN_GROUPS:
        part = [r for r in ordered_rows if r["group_key"] == g["key"]]
        if not part:
            continue
        fig.add_trace(
            go.Bar(
                y=[r["col"] for r in part],
                x=[r["pct"] for r in part],
                orientation="h",
                name=g["label"],
                marker=dict(color=g["color"], line=dict(color=T.COLOR_BG_CARD, width=0.6)),
                text=[f"{r['pct']:.1f}% ({r['n_valid']})" for r in part],
                textposition="outside",
                textfont=dict(size=10.5, color=TEXT_COLOR),
                cliponaxis=False,
                customdata=[[r["n_valid"], total_n, r["usage"], r["desc"]] for r in part],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "유효 표본: <b>%{customdata[0]}/%{customdata[1]}개 (%{x:.1f}%)</b><br>"
                    "%{customdata[2]}<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=50.0,
        line_dash="dot",
        line_color=AXIS_LINE_COLOR,
        line_width=1.4,
        annotation_text="50% 벤치마크 채택 기준",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color=TEXT_COLOR),
        annotation=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor=T.DATA_GRAY_3,
            borderwidth=1,
            borderpad=3,
        ),
    )

    _base_layout(fig, height=580, margin=dict(l=175, r=68, t=44, b=42), show_legend=True)
    fig.update_layout(
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            font=dict(size=11, color=TEXT_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title_text="유효 관측률 (%)",
        range=[0, 118],
        tickvals=[0, 25, 50, 75, 100],
        ticksuffix="%",
        tickfont=dict(size=11.5, color=MUTED_COLOR),
        title_font=dict(size=12, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=y_order,
        tickfont=dict(size=11, color=TEXT_COLOR),
        showgrid=False,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig


def build_open_closed_coverage_bar(df: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bar comparing Open vs Closed recording rates (%) across key numeric columns."""
    active_access = [a for a in ACCESS_OPTIONS if (df["_access"] == a).any()]
    if df.empty or not active_access:
        fig = go.Figure()
        _base_layout(fig, height=440, margin=dict(l=20, r=20, t=20, b=20), show_legend=False)
        return fig

    y_labels = [lbl for _, lbl in reversed(RAW_KEY_COMPARE_COLS)]
    fig = go.Figure()
    for acc_label in active_access:
        part = df[df["_access"] == acc_label]
        denom = max(len(part), 1)
        xs: list[float] = []
        texts: list[str] = []
        custom: list[list[int]] = []
        for raw_col, _ in reversed(RAW_KEY_COMPARE_COLS):
            cnt = int(_col_valid_mask(part, raw_col).sum())
            pct = 100.0 * cnt / denom
            xs.append(pct)
            texts.append(f"{pct:.1f}%")
            custom.append([cnt, len(part)])
        st_cfg = ACCESS_STYLE[acc_label]
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=xs,
                orientation="h",
                name=f"{acc_label} (n={len(part)})",
                marker=dict(color=st_cfg["color"], line=dict(color=T.COLOR_BG_CARD, width=0.8)),
                text=texts,
                textposition="outside",
                textfont=dict(size=11, color=TEXT_COLOR),
                cliponaxis=False,
                customdata=custom,
                hovertemplate=(
                    f"<b>%{{y}} · {acc_label}</b><br>"
                    "기록 모델: <b>%{customdata[0]}/%{customdata[1]}개 (%{x:.1f}%)</b><extra></extra>"
                ),
            )
        )

    _base_layout(fig, height=440, margin=dict(l=175, r=58, t=42, b=44), show_legend=True)
    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.10,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=12, color=TEXT_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title_text="공개·기록률 (%)",
        range=[0, 114],
        tickvals=[0, 25, 50, 75, 100],
        ticksuffix="%",
        tickfont=dict(size=11.5, color=MUTED_COLOR),
        title_font=dict(size=12, color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=y_labels,
        tickfont=dict(size=11.5, color=TEXT_COLOR),
        showgrid=False,
        linecolor=AXIS_LINE_COLOR,
        zeroline=False,
    )
    return fig
