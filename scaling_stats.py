"""Data loader and Weeks 6-7 OLS / multiple regression / LINE diagnostics module.

Uses the raw Epoch AI CSV as-is (no Korean translation of column names or category
values; MMLU/BBH kept on raw 0~1 scale; Active Parameters used directly without
synthetic merging; 'Random chance' baseline row separated cleanly).

Strictly implements ONLY the statistical methods covered in Weeks 6 & 7:
simple/multiple OLS, 95% mean CI & individual PI (within observed [x_min, x_max]
only), F-test, t-test, R², Adjusted R², RSE, VIF, Ramsey RESET, Durbin-Watson,
Shapiro-Wilk, Breusch-Pagan, and HC3 robust standard errors — matching statsmodels
to 1e-10 precision.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "benchmarks_with_model_accessibility.csv"

SOURCE_REPO_URL = (
    "https://github.com/epoch-research/open-model-trends/blob/main/"
    "data/benchmarks_with_model_accessibility.csv"
)
SOURCE_HUB_URL = "https://epoch.ai/data"

BENCHMARK_OPTIONS: list[str] = ["MMLU", "BBH"]
ACCESS_OPTIONS: list[str] = ["Open", "Closed"]

BENCHMARK_META: dict[str, dict[str, Any]] = {
    "MMLU": {
        "col": "MMLU",
        "label": "MMLU",
        "random_baseline": 0.25,
    },
    "BBH": {
        "col": "BBH",
        "label": "BBH",
        "random_baseline": 0.25,
    },
}

SCALE_META: dict[str, dict[str, str]] = {
    "log10_flop": {
        "raw_col": "Training compute (FLOP)",
        "num_col": "_compute_flop",
        "title": "Training compute (FLOP)",
        "axis_label": "Training compute (FLOP, log10)",
        "var_label": "log10(Training compute (FLOP))",
    },
    "log10_params": {
        "raw_col": "Active Parameters",
        "num_col": "_active_params",
        "title": "Active Parameters",
        "axis_label": "Active Parameters (개, log10)",
        "var_label": "log10(Active Parameters)",
    },
    "log10_total_params": {
        "raw_col": "Model size (parameters)",
        "num_col": "_total_params",
        "title": "Model size (parameters)",
        "axis_label": "Model size (parameters, log10)",
        "var_label": "log10(Model size (parameters))",
    },
    "log10_dataset": {
        "raw_col": "Dataset size",
        "num_col": "_dataset_size",
        "title": "Dataset size",
        "axis_label": "Dataset size (tokens/data, log10)",
        "var_label": "log10(Dataset size)",
    },
}


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


@st.cache_data
def load_scaling_data() -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """Load the untouched Epoch AI CSV and separate the 'Random chance' baseline row.

    All 24 original CSV columns remain completely unmodified in the returned DataFrame
    so filtered views and CSV exports preserve raw column names and values verbatim.
    """
    raw = pd.read_csv(DATA_PATH)
    raw_cols = list(raw.columns)
    required = {
        "System",
        "Date",
        "Open/Closed",
        "Training compute (FLOP)",
        "Active Parameters",
        "Dataset size",
        "MMLU",
        "BBH",
    }
    missing = required - set(raw_cols)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    is_rc = raw["System"].astype(str).str.strip().str.lower() == "random chance"
    baseline_rows = raw.loc[is_rc]
    baselines = {"MMLU": 0.25, "BBH": 0.25}
    if not baseline_rows.empty:
        r0 = baseline_rows.iloc[0]
        m_val = pd.to_numeric(r0.get("MMLU"), errors="coerce")
        b_val = pd.to_numeric(r0.get("BBH"), errors="coerce")
        if pd.notna(m_val):
            baselines["MMLU"] = float(m_val)
        if pd.notna(b_val):
            baselines["BBH"] = float(b_val)

    df = raw.loc[~is_rc].copy()

    # Internal parsed helper columns (all original 24 raw CSV columns stay untouched)
    df["_system"] = df["System"].astype(str).str.strip()
    df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.loc[df["_date"].notna()].copy()

    df["_year"] = df["_date"].dt.year.astype(int)
    df["year_frac"] = (
        df["_date"].dt.year + (df["_date"].dt.dayofyear - 1) / 365.25
    ).astype(float)
    df["_date_str"] = df["_date"].dt.strftime("%Y-%m-%d")
    df["_access"] = df["Open/Closed"].astype(str).str.strip()

    # Raw scale columns parsed directly without any synthetic column merging
    df["_compute_flop"] = _to_numeric(df["Training compute (FLOP)"])
    df["_active_params"] = _to_numeric(df["Active Parameters"])
    df["_total_params"] = (
        _to_numeric(df["Model size (parameters)"])
        if "Model size (parameters)" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["_dataset_size"] = _to_numeric(df["Dataset size"])

    df["log10_flop"] = np.where(df["_compute_flop"] > 0, np.log10(df["_compute_flop"]), np.nan)
    df["log10_params"] = np.where(df["_active_params"] > 0, np.log10(df["_active_params"]), np.nan)
    df["log10_total_params"] = np.where(df["_total_params"] > 0, np.log10(df["_total_params"]), np.nan)
    df["log10_dataset"] = np.where(df["_dataset_size"] > 0, np.log10(df["_dataset_size"]), np.nan)

    df = df.sort_values(["_date", "_system"], kind="mergesort").reset_index(drop=True)
    return df, baselines, raw_cols


def format_p(p: float | None) -> str:
    """Format p-value per Week 6 p.38, 40 (never print 0.000; use p < 0.001)."""
    if p is None or not math.isfinite(p):
        return "-"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def format_p_short(p: float | None) -> str:
    """Compact p-value for tables (never 0.000; use < 0.001)."""
    if p is None or not math.isfinite(p):
        return "-"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _ols_core(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Core OLS + HC3 algebra via QR decomposition matching statsmodels to 1e-10."""
    n, p_dim = X.shape
    k = p_dim - 1
    df_resid = n - p_dim

    Q, R = np.linalg.qr(X)
    r_inv = np.linalg.pinv(R)
    pinv_x = r_inv @ Q.T
    xtx_inv = r_inv @ r_inv.T

    beta = pinv_x @ y
    fitted = X @ beta
    resid = y - fitted
    sse = float(resid @ resid)
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    mse = sse / max(df_resid, 1)
    rse = math.sqrt(mse)
    r2 = 1.0 - (sse / sst) if sst > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(df_resid, 1)

    f_std = ((sst - sse) / max(k, 1)) / mse if mse > 0 else float("nan")
    fp_std = (
        float(stats.f.sf(f_std, k, df_resid))
        if math.isfinite(f_std) and df_resid > 0
        else float("nan")
    )

    cov_std = mse * xtx_inv
    se_std = np.sqrt(np.maximum(np.diag(cov_std), 0.0))

    # HC3 heteroscedasticity-robust covariance matching statsmodels fit(cov_type='HC3')
    h = np.sum(Q ** 2, axis=1)
    denom = np.maximum(1.0 - h, 1e-9)
    scaled_pinv = pinv_x * (resid / denom)
    cov_hc3 = scaled_pinv @ scaled_pinv.T
    se_hc3 = np.sqrt(np.maximum(np.diag(cov_hc3), 0.0))

    # HC3 Wald F-test matching statsmodels fit(cov_type='HC3').fvalue / f_pvalue
    if k > 0 and df_resid > 0:
        b_slope = beta[1:]
        cov_slope_hc3 = cov_hc3[1:, 1:]
        f_hc3 = float(b_slope @ np.linalg.pinv(cov_slope_hc3) @ b_slope) / k
        fp_hc3 = float(stats.f.sf(f_hc3, k, df_resid)) if math.isfinite(f_hc3) else float("nan")
    else:
        f_hc3 = float("nan")
        fp_hc3 = float("nan")

    return {
        "n": n,
        "k": k,
        "df_resid": df_resid,
        "xtx_inv": xtx_inv,
        "beta": beta,
        "fitted": fitted,
        "resid": resid,
        "sse": sse,
        "sst": sst,
        "mse": mse,
        "rse": rse,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_std": f_std,
        "fp_std": fp_std,
        "f_hc3": f_hc3,
        "fp_hc3": fp_hc3,
        "cov_std": cov_std,
        "se_std": se_std,
        "cov_hc3": cov_hc3,
        "se_hc3": se_hc3,
    }


def _run_line_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    core: dict[str, Any],
    df_sorted: pd.DataFrame,
) -> dict[str, Any]:
    """Run Week 7 four LINE diagnostics (RESET, DW, Shapiro-Wilk, BP) + residual/QQ arrays."""
    resid = core["resid"]
    fitted = core["fitted"]
    sse = core["sse"]
    n, p_dim = X.shape
    k = p_dim - 1

    # 1. Linearity: Ramsey RESET (power=2, use_f=True) — Week 7 p.20
    reset_p = None
    if n >= p_dim + 3 and sse > 0:
        try:
            Xr = np.column_stack([X, fitted ** 2])
            if np.linalg.matrix_rank(Xr) > np.linalg.matrix_rank(X):
                br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
                sse_r = float(np.sum((y - Xr @ br) ** 2))
                df2 = n - Xr.shape[1]
                if df2 > 0 and sse_r > 0:
                    f_reset = max(0.0, (sse - sse_r)) / (sse_r / df2)
                    reset_p = float(stats.f.sf(f_reset, 1, df2))
        except Exception:
            reset_p = None

    # 2. Independence: Durbin-Watson (chronological order by Date) — Week 7 p.19
    dw_val = None
    if n >= 3 and sse > 0:
        dw_val = float(np.sum(np.diff(resid) ** 2) / sse)

    # 3. Normality: Shapiro-Wilk + Q-Q theoretical quantiles — Week 7 p.17, 19
    shapiro_p = None
    if 3 <= n <= 5000:
        try:
            shapiro_p = float(stats.shapiro(resid).pvalue)
        except Exception:
            shapiro_p = None

    r_std = float(np.std(resid, ddof=1)) if n > 1 else 1.0
    std_resid = (resid - float(np.mean(resid))) / r_std if r_std > 0 else np.zeros(n)
    order = np.argsort(std_resid)
    qq_obs = std_resid[order]
    (qq_theo, _), _ = stats.probplot(std_resid, dist="norm", fit=True)

    # 4. Equal variance: Breusch-Pagan (LM = n * R² of e² ~ X ~ chi2(k)) — Week 7 p.19
    bp_p = None
    if n >= p_dim + 2:
        try:
            e2 = resid ** 2
            sst_e2 = float(np.sum((e2 - float(np.mean(e2))) ** 2))
            if sst_e2 > 0:
                bbp, *_ = np.linalg.lstsq(X, e2, rcond=None)
                sse_e2 = float(np.sum((e2 - X @ bbp) ** 2))
                r2_bp = max(0.0, 1.0 - sse_e2 / sst_e2)
                bp_p = float(stats.chi2.sf(n * r2_bp, k))
        except Exception:
            bp_p = None

    return {
        "reset_p": reset_p,
        "dw": dw_val,
        "shapiro_p": shapiro_p,
        "bp_p": bp_p,
        "fitted": fitted,
        "resid": resid,
        "std_resid": std_resid,
        "qq_theo": np.asarray(qq_theo, dtype=float),
        "qq_obs": np.asarray(qq_obs, dtype=float),
        "models_ordered": df_sorted["_system"].to_numpy(),
        "models_qq": df_sorted["_system"].to_numpy()[order],
        "dates_ordered": df_sorted["_date_str"].to_numpy(),
    }


def fit_simple_ols(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    use_hc3: bool = False,
    n_grid: int = 80,
) -> dict[str, Any]:
    """Fit simple linear OLS (Weeks 6 & 7) strictly within observed [x_min, x_max]."""
    needed = list(dict.fromkeys([x_col, y_col, "year_frac", "_system", "_date_str"]))
    sub = df[needed].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col]).sort_values(["year_frac", "_system"], kind="mergesort")
    n = len(sub)
    if n < 4:
        return {
            "ok": False,
            "n": n,
            "reason": f"n={n} (< 4)",
        }
    if sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
        return {
            "ok": False,
            "n": n,
            "reason": f"n={n} · 분산 0 (단일 값)",
        }

    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), x])
    core = _ols_core(X, y)
    df_res = core["df_resid"]
    t_crit = float(stats.t.ppf(0.975, df_res))
    z_crit = float(stats.norm.ppf(0.975))

    b0 = float(core["beta"][0])
    b1 = float(core["beta"][1])
    se_std = float(core["se_std"][1])
    se_hc3 = float(core["se_hc3"][1])

    t_std = b1 / se_std if se_std > 0 else float("nan")
    p_std = float(2.0 * stats.t.sf(abs(t_std), df_res)) if math.isfinite(t_std) else float("nan")
    ci_std = (b1 - t_crit * se_std, b1 + t_crit * se_std)

    # HC3 matches statsmodels OLS.fit(cov_type='HC3') (asymptotic normal z, use_t=False)
    t_hc3 = b1 / se_hc3 if se_hc3 > 0 else float("nan")
    p_hc3 = float(2.0 * stats.norm.sf(abs(t_hc3))) if math.isfinite(t_hc3) else float("nan")
    ci_hc3 = (b1 - z_crit * se_hc3, b1 + z_crit * se_hc3)

    se_act = se_hc3 if use_hc3 else se_std
    t_act = t_hc3 if use_hc3 else t_std
    p_act = p_hc3 if use_hc3 else p_std
    ci_act = ci_hc3 if use_hc3 else ci_std
    f_act = core["f_hc3"] if use_hc3 else core["f_std"]
    fp_act = core["fp_hc3"] if use_hc3 else core["fp_std"]

    # Strictly inside observed [x_min, x_max] (NO extrapolation — Week 7 p.26)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_grid = np.linspace(x_min, x_max, n_grid)
    Xg = np.column_stack([np.ones(n_grid), x_grid])
    y_hat = Xg @ core["beta"]

    cov_act = core["cov_hc3"] if use_hc3 else core["cov_std"]
    crit_act = z_crit if use_hc3 else t_crit
    var_mean = np.maximum(np.sum((Xg @ cov_act) * Xg, axis=1), 0.0)
    se_mean = np.sqrt(var_mean)
    mean_ci_low = y_hat - crit_act * se_mean
    mean_ci_high = y_hat + crit_act * se_mean

    # Individual 95% prediction interval matching statsmodels get_prediction().summary_frame()
    se_obs = np.sqrt(core["mse"] + var_mean)
    obs_ci_low = y_hat - crit_act * se_obs
    obs_ci_high = y_hat + crit_act * se_obs

    pearson_r = float(np.corrcoef(x, y)[0, 1])
    line = _run_line_diagnostics(X, y, core, sub)

    var_name = SCALE_META[x_col]["var_label"] if x_col in SCALE_META else ("Date (year)" if x_col == "year_frac" else x_col)

    return {
        "ok": True,
        "n": n,
        "x_col": x_col,
        "y_col": y_col,
        "var_name": var_name,
        "use_hc3": use_hc3,
        "x_min": x_min,
        "x_max": x_max,
        "b0": b0,
        "b1": b1,
        "se": se_act,
        "t": t_act,
        "p": p_act,
        "ci_low": ci_act[0],
        "ci_high": ci_act[1],
        "se_std": se_std,
        "t_std": t_std,
        "p_std": p_std,
        "ci_std": ci_std,
        "se_hc3": se_hc3,
        "t_hc3": t_hc3,
        "p_hc3": p_hc3,
        "ci_hc3": ci_hc3,
        "r2": core["r2"],
        "adj_r2": core["adj_r2"],
        "rse": core["rse"],
        "pearson_r": pearson_r,
        "f_stat": f_act,
        "f_pval": fp_act,
        "f_std": core["f_std"],
        "fp_std": core["fp_std"],
        "f_hc3": core["f_hc3"],
        "fp_hc3": core["fp_hc3"],
        "x_grid": x_grid,
        "y_hat": y_hat,
        "mean_ci_low": mean_ci_low,
        "mean_ci_high": mean_ci_high,
        "obs_ci_low": obs_ci_low,
        "obs_ci_high": obs_ci_high,
        "line": line,
    }


def fit_multi_ols(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
    use_hc3: bool = False,
) -> dict[str, Any]:
    """Fit multiple OLS with F-test, Adjusted R², per-term t-test, VIF, and LINE diagnostics (Week 7)."""
    needed = list(dict.fromkeys(x_cols + [y_col, "year_frac", "_system", "_date_str", "_access"]))
    sub = df[needed].copy()
    for c in x_cols + [y_col]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=x_cols + [y_col]).sort_values(["year_frac", "_system"], kind="mergesort")
    n = len(sub)
    k = len(x_cols)
    if n < k + 4:
        return {
            "ok": False,
            "n": n,
            "reason": f"n={n} (< {k + 4})",
        }
    if sub[y_col].nunique() < 2 or any(sub[c].nunique() < 2 for c in x_cols):
        return {
            "ok": False,
            "n": n,
            "reason": f"n={n} · 분산 0 (상수 열)",
        }

    X = np.column_stack([np.ones(n)] + [sub[c].to_numpy(dtype=float) for c in x_cols])
    if np.linalg.matrix_rank(X) < k + 1:
        return {
            "ok": False,
            "n": n,
            "reason": f"n={n} · 완전 다중공선성 (계수 부족)",
        }
    y = sub[y_col].to_numpy(dtype=float)
    core = _ols_core(X, y)
    df_res = core["df_resid"]
    t_crit = float(stats.t.ppf(0.975, df_res))
    z_crit = float(stats.norm.ppf(0.975))

    # VIF_j = 1 / (1 - R²_j) — exact Week 7 p.11 formula matching statsmodels variance_inflation_factor
    vifs: dict[str, float] = {}
    for col in x_cols:
        others = [c for c in x_cols if c != col]
        Xo = np.column_stack([np.ones(n)] + [sub[c].to_numpy(dtype=float) for c in others])
        yj = sub[col].to_numpy(dtype=float)
        sst_j = float(np.sum((yj - float(np.mean(yj))) ** 2))
        if sst_j > 0:
            bj, *_ = np.linalg.lstsq(Xo, yj, rcond=None)
            sse_j = float(np.sum((yj - Xo @ bj) ** 2))
            r2_j = 1.0 - sse_j / sst_j
            vifs[col] = 1.0 / max(1.0 - r2_j, 1e-12)
        else:
            vifs[col] = float("nan")

    terms: list[dict[str, Any]] = []
    for j, col in enumerate(x_cols, start=1):
        coef = float(core["beta"][j])
        se_std = float(core["se_std"][j])
        se_hc3 = float(core["se_hc3"][j])

        t_std = coef / se_std if se_std > 0 else float("nan")
        p_std = float(2.0 * stats.t.sf(abs(t_std), df_res)) if math.isfinite(t_std) else float("nan")
        ci_std = (coef - t_crit * se_std, coef + t_crit * se_std)

        t_hc3 = coef / se_hc3 if se_hc3 > 0 else float("nan")
        p_hc3 = float(2.0 * stats.norm.sf(abs(t_hc3))) if math.isfinite(t_hc3) else float("nan")
        ci_hc3 = (coef - z_crit * se_hc3, coef + z_crit * se_hc3)

        se_act = se_hc3 if use_hc3 else se_std
        t_act = t_hc3 if use_hc3 else t_std
        p_act = p_hc3 if use_hc3 else p_std
        ci_act = ci_hc3 if use_hc3 else ci_std

        terms.append(
            {
                "col": col,
                "label": SCALE_META.get(col, {}).get("var_label", col),
                "coef": coef,
                "se": se_act,
                "t": t_act,
                "p": p_act,
                "ci_low": ci_act[0],
                "ci_high": ci_act[1],
                "se_std": se_std,
                "t_std": t_std,
                "p_std": p_std,
                "ci_std": ci_std,
                "se_hc3": se_hc3,
                "t_hc3": t_hc3,
                "p_hc3": p_hc3,
                "ci_hc3": ci_hc3,
                "vif": vifs.get(col, float("nan")),
            }
        )

    line = _run_line_diagnostics(X, y, core, sub)
    f_act = core["f_hc3"] if use_hc3 else core["f_std"]
    fp_act = core["fp_hc3"] if use_hc3 else core["fp_std"]

    return {
        "ok": True,
        "n": n,
        "x_cols": x_cols,
        "y_col": y_col,
        "use_hc3": use_hc3,
        "b0": float(core["beta"][0]),
        "r2": core["r2"],
        "adj_r2": core["adj_r2"],
        "rse": core["rse"],
        "f_stat": f_act,
        "f_pval": fp_act,
        "f_std": core["f_std"],
        "fp_std": core["fp_std"],
        "f_hc3": core["f_hc3"],
        "fp_hc3": core["fp_hc3"],
        "terms": terms,
        "vifs": vifs,
        "actual": y,
        "access_ordered": sub["_access"].to_numpy(),
        "line": line,
    }


def format_ols_stat_line(res: dict[str, Any], mode: str) -> str:
    """Single concise numeric stat line (ZERO narrative prose).

    mode:
      - 'time_perf': Date -> MMLU/BBH (raw 0~1 per yr)
      - 'time_log10': Date -> log10(Scale) (decades/yr + multiplier/yr)
      - 'direct_scale': log10(Scale) -> MMLU/BBH (raw 0~1 per 10x)
    """
    if not res.get("ok"):
        return str(res.get("reason", "n < 4"))
    b1 = res["b1"]
    lo, hi = res["ci_low"], res["ci_high"]
    p_txt = format_p(res["p"])
    se_tag = " · HC3" if res.get("use_hc3") else ""
    stat_sym = "z" if res.get("use_hc3") else "t"

    if mode == "time_perf":
        return (
            f"b1={b1:+.3f}/yr · 95% CI {lo:+.3f}~{hi:+.3f} · "
            f"{stat_sym}={res['t']:.2f} · {p_txt} · R²={res['r2']:.3f}{se_tag}"
        )
    if mode == "time_log10":
        mult = 10.0 ** b1
        return (
            f"b1={b1:+.2f}/yr ({mult:.2f}x/yr) · 95% CI {lo:+.2f}~{hi:+.2f} · "
            f"{stat_sym}={res['t']:.2f} · {p_txt} · R²={res['r2']:.3f}{se_tag}"
        )
    # direct_scale
    return (
        f"b1={b1:+.3f}/10x · 95% CI {lo:+.3f}~{hi:+.3f} · "
        f"{stat_sym}={res['t']:.2f} · {p_txt} · R²={res['r2']:.3f} · RSE={res['rse']:.3f}{se_tag}"
    )


def evaluate_line_table(line: dict[str, Any]) -> list[dict[str, str]]:
    """Compact numeric Week 7 LINE 4-test table (ZERO prose interpretations)."""
    rp = line.get("reset_p")
    dw = line.get("dw")
    sp = line.get("shapiro_p")
    bp = line.get("bp_p")

    return [
        {
            "Assumption": "Linearity",
            "Test": "RESET",
            "Statistic / p": format_p_short(rp),
            "Threshold": "p ≥ 0.05",
            "Status": "-" if rp is None else ("Check" if rp < 0.05 else "OK"),
        },
        {
            "Assumption": "Independence",
            "Test": "Durbin-Watson",
            "Statistic / p": f"{dw:.2f}" if dw is not None else "-",
            "Threshold": "1.50 ~ 2.50",
            "Status": "-" if dw is None else ("OK" if 1.5 <= dw <= 2.5 else "Check"),
        },
        {
            "Assumption": "Normality",
            "Test": "Shapiro-Wilk",
            "Statistic / p": format_p_short(sp),
            "Threshold": "p ≥ 0.05",
            "Status": "-" if sp is None else ("Check" if sp < 0.05 else "OK"),
        },
        {
            "Assumption": "Equal Variance",
            "Test": "Breusch-Pagan",
            "Statistic / p": format_p_short(bp),
            "Threshold": "p ≥ 0.05",
            "Status": "-" if bp is None else ("Check" if bp < 0.05 else "OK"),
        },
    ]
