import math
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "figure.figsize": (6.2, 4.0),
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )

def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _robust_limits(x: pd.Series, y: pd.Series, pad_frac: float = 0.05) -> Tuple[float, float]:
    df = pd.DataFrame({"x": _as_num(x), "y": _as_num(y)}).dropna()
    if df.empty:
        return 0.0, 1.0
    lo = float(min(df["x"].quantile(0.01), df["y"].quantile(0.01)))
    hi = float(max(df["x"].quantile(0.99), df["y"].quantile(0.99)))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo = float(min(df["x"].min(), df["y"].min()))
        hi = float(max(df["x"].max(), df["y"].max()))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad

def _regression_data(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame | None:
    y_true = _as_num(y_true)
    y_pred = _as_num(y_pred)
    df = pd.DataFrame({'y': y_true, 'p': y_pred}).dropna()
    if df.empty:
        return None
    return df

def plot_pred_vs_actual(y_true: pd.Series, y_pred: pd.Series, out_prefix: Path, title_prefix: str) -> Path | None:
    apply_pub_style()
    df = _regression_data(y_true, y_pred)
    if df is None:
        return None
    lo, hi = _robust_limits(df['y'], df['p'])
    pva_path = out_prefix.with_name(out_prefix.name + '_pred_vs_actual.png')
    plt.figure()
    plt.scatter(df['y'], df['p'], s=18, alpha=0.85, edgecolors='none')
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.2, alpha=0.9)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{title_prefix} - Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(pva_path)
    plt.close()
    return pva_path

def plot_residuals_vs_pred(y_true: pd.Series, y_pred: pd.Series, out_prefix: Path, title_prefix: str) -> Path | None:
    apply_pub_style()
    df = _regression_data(y_true, y_pred)
    if df is None:
        return None
    resid = df["p"] - df["y"]
    rvp_path = out_prefix.with_name(out_prefix.name + '_residuals_vs_pred.png')
    plt.figure()
    plt.scatter(df['p'], resid, s=18, alpha=0.85, edgecolors='none')
    plt.axhline(0, linestyle='--', linewidth=1.2, alpha=0.9)
    plt.xlabel('Predicted')
    plt.ylabel('Residual (pred − actual)')
    plt.title(f'{title_prefix} - Residuals vs Predicted')
    plt.tight_layout()
    plt.savefig(rvp_path)
    plt.close()
    return rvp_path

def plot_residuals_hist(y_true: pd.Series, y_pred: pd.Series, out_prefix: Path, title_prefix: str) -> Path | None:
    apply_pub_style()
    df = _regression_data(y_true, y_pred)
    if df is None:
        return None
    resid = df["p"] - df["y"]
    rh_path = out_prefix.with_name(out_prefix.name + '_residuals_hist.png')
    plt.figure()
    plt.hist(resid.dropna(), bins=min(30, max(8, int(len(resid) / 2))), alpha=0.9)
    plt.axvline(0, linestyle='--', linewidth=1.2, alpha=0.9)
    plt.xlabel('Residual (pred − actual)')
    plt.ylabel('Count')
    plt.title(f'{title_prefix} - Residual distribution')
    plt.tight_layout()
    plt.savefig(rh_path)
    plt.close()
    return rh_path

def _classification_curves(
    y_true: pd.Series, y_score: pd.Series,
) -> tuple[list[float], list[float], list[float], list[float]] | None:
    y_true = _as_num(y_true)
    y_score = _as_num(y_score)
    df = pd.DataFrame({'y': y_true, 's': y_score}).dropna()
    if df.empty:
        return None
    uniq = set(df['y'].unique().tolist())
    if not uniq.issubset({0, 1}):
        return None
    df = df.sort_values('s', ascending=False).reset_index(drop=True)
    P = int((df['y'] == 1).sum())
    N = int((df['y'] == 0).sum())
    if P == 0 or N == 0:
        return None
    tp = 0
    fp = 0
    tps, fps = ([], [])
    for _, row in df.iterrows():
        if row['y'] == 1:
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)
    tpr = [x / P for x in tps]
    fpr = [x / N for x in fps]
    precision = [tps[i] / max(1, tps[i] + fps[i]) for i in range(len(tps))]
    recall = [tps[i] / P for i in range(len(tps))]
    return fpr, tpr, recall, precision

def plot_roc(y_true: pd.Series, y_score: pd.Series, out_prefix: Path, title_prefix: str) -> Path | None:
    apply_pub_style()
    curves = _classification_curves(y_true, y_score)
    if curves is None:
        return None
    fpr, tpr, recall, precision = curves
    roc_path = out_prefix.with_name(out_prefix.name + '_roc.png')
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.1, alpha=0.8)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'{title_prefix} - ROC curve')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    return roc_path

def plot_pr(y_true: pd.Series, y_score: pd.Series, out_prefix: Path, title_prefix: str) -> Path | None:
    apply_pub_style()
    curves = _classification_curves(y_true, y_score)
    if curves is None:
        return None
    fpr, tpr, recall, precision = curves
    pr_path = out_prefix.with_name(out_prefix.name + '_pr.png')
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title_prefix} - Precision–Recall curve')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    return pr_path

_RENDERERS = {
    "roc": plot_roc,
    "pr": plot_pr,
    "pred_vs_actual": plot_pred_vs_actual,
    "residuals_vs_pred": plot_residuals_vs_pred,
    "residuals_hist": plot_residuals_hist,
}

def render_plots(
    plot_data: dict[str, tuple[pd.Series, pd.Series]], out_prefix: Path, title_prefix: str,
) -> dict[str, Path]:
    plots = {}
    for name, (labels, predictions) in plot_data.items():
        if name not in _RENDERERS:
            raise ValueError(f"No renderer registered for plot '{name}'")
        path = _RENDERERS[name](labels, predictions, out_prefix, title_prefix)
        if path is not None:
            plots[name] = path
    return plots
