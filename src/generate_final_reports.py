#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# headless safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


# =========================
# Matplotlib style (publication-ish, no extra deps)
# =========================
def apply_pub_style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "figure.figsize": (6.2, 4.2),
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
    })


# =========================
# Helpers
# =========================
def find_first(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def safe_read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def parse_metrics_txt(path: Optional[Path]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not path or not path.exists():
        return metrics
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            metrics[k] = float(v)
        except Exception:
            continue
    return metrics


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["numeric_label", "class", "target", "label", "y"]:
        if c in df.columns:
            return c
    return None


def detect_prediction_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["prediction", "predictions", "y_pred", "pred", "score", "prob", "proba", "p"]:
        if c in df.columns:
            return c
    numeric_cols = [c for c in df.columns if c != "id" and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else None


def merge_labels_and_preds(labeled: pd.DataFrame, preds: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "id" not in labeled.columns or "id" not in preds.columns:
        raise ValueError("Both labeled data and predictions must include an 'id' column.")
    y_col = detect_label_column(labeled)
    if not y_col:
        raise ValueError("Could not detect label column in labeled data.")
    pred_col = detect_prediction_column(preds)
    if not pred_col:
        raise ValueError("Could not detect prediction column in predictions file.")
    merged = labeled[["id", y_col]].merge(preds[["id", pred_col]], on="id", how="inner")
    return merged[y_col], merged[pred_col]


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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Task detection + config
# =========================
def load_config_json(agent_dir: Path) -> Dict:
    cfg = agent_dir / "best_run_files" / "config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text())
        except Exception:
            return {}
    return {}


def load_task_type(agent_dir: Path) -> str:
    d = load_config_json(agent_dir)
    t = d.get("task_type")
    return str(t).strip().lower() if isinstance(t, str) else "unknown"


def load_val_metric(agent_dir: Path) -> Optional[str]:
    d = load_config_json(agent_dir)
    vm = d.get("val_metric")
    return vm.strip() if isinstance(vm, str) else None


# =========================
# Iteration discovery / inputs
# =========================
def discover_iterations(agent_dir: Path) -> List[int]:
    run_files = agent_dir / "run_files"
    iters: List[int] = []
    if run_files.exists():
        for p in run_files.iterdir():
            if p.is_dir() and p.name.startswith("iteration_"):
                try:
                    iters.append(int(p.name.split("_", 1)[1]))
                except Exception:
                    pass
    return sorted(set(iters))


@dataclass
class SplitArtifacts:
    split_name: str  # train/validation/test
    labeled_csv: Optional[Path]
    preds_csv: Optional[Path]
    metrics: Dict[str, float]


@dataclass
class IterationInputs:
    iteration: int
    report_md: Optional[Path]
    splits: List[SplitArtifacts]  # train/validation/(test if exists)


def gather_iteration_inputs(agent_dir: Path, iteration: int) -> IterationInputs:
    reports_dir = agent_dir / "reports"
    run_files = agent_dir / "run_files"
    best = agent_dir / "best_run_files"
    iter_dir = run_files / f"iteration_{iteration}"

    report_md = reports_dir / f"run_report_iter_{iteration}.md"
    if not report_md.exists():
        report_md = None

    # labeled data (host output structure)
    train_csv = run_files / "train.csv"
    val_csv = run_files / "validation.csv"
    test_csv = run_files / "test.csv"  # may not exist in outputs

    if not train_csv.exists():
        train_csv = None
    if not val_csv.exists():
        val_csv = None
    if not test_csv.exists():
        test_csv = None

    # predictions (iteration folder names from your pipeline)
    train_preds = find_first([
        iter_dir / "eval_predictions_train.csv",
        iter_dir / "train_predictions.csv",
        iter_dir / "predictions_train.csv",
    ])
    val_preds = find_first([
        iter_dir / "eval_predictions_validation.csv",
        iter_dir / "validation_predictions.csv",
        iter_dir / "predictions_validation.csv",
    ])
    test_preds = find_first([
        iter_dir / "eval_predictions_test.csv",
        iter_dir / "test_predictions.csv",
        iter_dir / "predictions_test.csv",
    ])

    # metrics
    train_metrics_path = find_first([iter_dir / "train_metrics.txt", best / "train_metrics.txt"])
    val_metrics_path = find_first([iter_dir / "validation_metrics.txt", best / "validation_metrics.txt"])
    test_metrics_path = find_first([iter_dir / "test_metrics.txt", best / "test_metrics.txt"])

    splits: List[SplitArtifacts] = [
        SplitArtifacts("train", train_csv, train_preds, parse_metrics_txt(train_metrics_path)),
        SplitArtifacts("validation", val_csv, val_preds, parse_metrics_txt(val_metrics_path)),
    ]
    # include test if we have metrics or preds or labeled
    if (test_csv and test_csv.exists()) or (test_preds and test_preds.exists()) or parse_metrics_txt(test_metrics_path):
        splits.append(SplitArtifacts("test", test_csv, test_preds, parse_metrics_txt(test_metrics_path)))

    return IterationInputs(iteration=iteration, report_md=report_md, splits=splits)


# =========================
# Plotting
# =========================
def plot_regression_publication(y_true: pd.Series, y_pred: pd.Series, out_prefix: Path, title_prefix: str) -> List[Path]:
    apply_pub_style()

    y_true = _as_num(y_true)
    y_pred = _as_num(y_pred)
    df = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
    if df.empty:
        return []

    # 1) Pred vs Actual
    pva_path = out_prefix.with_name(out_prefix.name + "_pred_vs_actual.png")
    lo, hi = _robust_limits(df["y"], df["p"])
    plt.figure()
    plt.scatter(df["y"], df["p"], s=18, alpha=0.85, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, alpha=0.9)  # y=x
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title_prefix} — Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(pva_path)
    plt.close()

    resid = df["p"] - df["y"]
    rlo = float(resid.quantile(0.01))
    rhi = float(resid.quantile(0.99))
    if not math.isfinite(rlo) or not math.isfinite(rhi) or rlo == rhi:
        rlo, rhi = float(resid.min()), float(resid.max())
        if rlo == rhi:
            rlo, rhi = rlo - 1.0, rhi + 1.0

    # 2) Residuals vs Predicted
    rvp_path = out_prefix.with_name(out_prefix.name + "_residuals_vs_pred.png")
    plt.figure()
    plt.scatter(df["p"], resid, s=18, alpha=0.85, edgecolors="none")
    plt.axhline(0, linestyle="--", linewidth=1.2, alpha=0.9)
    plt.ylim(rlo, rhi)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (pred − actual)")
    plt.title(f"{title_prefix} — Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(rvp_path)
    plt.close()

    # 3) Residuals histogram
    rh_path = out_prefix.with_name(out_prefix.name + "_residuals_hist.png")
    plt.figure()
    plt.hist(resid.dropna(), bins=min(30, max(8, int(len(resid) / 2))), alpha=0.9)
    plt.axvline(0, linestyle="--", linewidth=1.2, alpha=0.9)
    plt.xlabel("Residual (pred − actual)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} — Residual distribution")
    plt.tight_layout()
    plt.savefig(rh_path)
    plt.close()

    return [pva_path, rvp_path, rh_path]


def plot_classification_publication(y_true: pd.Series, y_score: pd.Series, out_prefix: Path, title_prefix: str) -> List[Path]:
    apply_pub_style()

    y_true = _as_num(y_true)
    y_score = _as_num(y_score)
    df = pd.DataFrame({"y": y_true, "s": y_score}).dropna()
    if df.empty:
        return []
    uniq = set(df["y"].unique().tolist())
    if not uniq.issubset({0, 1}):
        return []

    df = df.sort_values("s", ascending=False).reset_index(drop=True)
    P = int((df["y"] == 1).sum())
    N = int((df["y"] == 0).sum())
    if P == 0 or N == 0:
        return []

    tp = 0
    fp = 0
    tps, fps = [], []
    for _, row in df.iterrows():
        if row["y"] == 1:
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)

    tpr = [x / P for x in tps]
    fpr = [x / N for x in fps]

    roc_path = out_prefix.with_name(out_prefix.name + "_roc.png")
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.1, alpha=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{title_prefix} — ROC curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    precision = [tps[i] / max(1, (tps[i] + fps[i])) for i in range(len(tps))]
    recall = [tps[i] / P for i in range(len(tps))]
    pr_path = out_prefix.with_name(out_prefix.name + "_pr.png")
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} — Precision–Recall curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    return [roc_path, pr_path]


def _guess_task_from_labels(y: pd.Series) -> str:
    yy = pd.to_numeric(y, errors="coerce").dropna()
    uniq = set(yy.unique().tolist())
    if uniq.issubset({0, 1}):
        return "classification"
    return "regression"


def build_plots_for_split(task_type: str, split_name: str, labeled_csv: Optional[Path], preds_csv: Optional[Path], plots_dir: Path, iteration: int) -> List[Path]:
    labeled = safe_read_csv(labeled_csv)
    preds = safe_read_csv(preds_csv)
    if labeled is None or preds is None:
        return []

    try:
        y_true, y_pred = merge_labels_and_preds(labeled, preds)
    except Exception:
        return []

    prefix = plots_dir / f"iter_{iteration}_{split_name}"
    title_prefix = f"{split_name.title()}"

    t = task_type
    if t not in ("classification", "regression"):
        t = _guess_task_from_labels(y_true)

    if t == "classification":
        return plot_classification_publication(y_true, y_pred, prefix, title_prefix)
    return plot_regression_publication(y_true, y_pred, prefix, title_prefix)


# =========================
# PDF generation (metrics table only + plots pages)
# =========================
def _draw_page_number(c: canvas.Canvas, page_no: int):
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawRightString(A4[0] - 1.5 * cm, 1.2 * cm, f"Page {page_no}")
    c.setFillColor(colors.black)


def _strip_md_noise(md: str) -> str:
    s = md.replace("\r\n", "\n")
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*[-*]\s+", "• ", s, flags=re.MULTILINE)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    return s.strip()


def _wrap_preserve_empty(text: str, width: int) -> List[str]:
    out: List[str] = []
    for raw in str(text).splitlines():
        if raw.strip() == "":
            out.append("")
        else:
            out.extend(textwrap.wrap(raw, width=width))
    return out


def draw_metrics_table_3cols(
    c: canvas.Canvas,
    x: float,
    y: float,
    metrics_by_split: Dict[str, Dict[str, float]],
    split_order: List[str],
    title: str,
) -> float:
    # union of metric keys
    all_keys = set()
    for s in split_order:
        all_keys |= set(metrics_by_split.get(s, {}).keys())
    keys = sorted(all_keys)
    if not keys:
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.grey)
        c.drawString(x, y, "No metrics found.")
        c.setFillColor(colors.black)
        return y - 16

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, title)
    y -= 14

    # columns: metric + each split
    col_metric = 6.5 * cm
    col_w = 4.2 * cm
    row_h = 14
    table_w = col_metric + col_w * len(split_order)

    # header
    c.setFillColor(colors.whitesmoke)
    c.rect(x, y - row_h + 2, table_w, row_h, fill=1, stroke=0)
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 6, y - 10, "Metric")
    for i, s in enumerate(split_order):
        c.drawRightString(x + col_metric + col_w * (i + 1) - 6, y - 10, s.title())
    y -= row_h

    c.setFont("Helvetica", 9)
    alt = False

    def fmt(z):
        if z is None:
            return "—"
        if isinstance(z, (int, float)):
            if abs(z) >= 1e6 or (abs(z) > 0 and abs(z) < 1e-3):
                return f"{z:.3g}"
            return f"{z:.6g}"
        return str(z)

    for k in keys:
        if y < 3.2 * cm:
            c.showPage()
            return y  # caller handles continuation (we keep it simple: only use on first page)

        if alt:
            c.setFillColor(colors.Color(0.97, 0.97, 0.97))
            c.rect(x, y - row_h + 2, table_w, row_h, fill=1, stroke=0)
            c.setFillColor(colors.black)
        alt = not alt

        c.drawString(x + 6, y - 10, str(k))
        for i, s in enumerate(split_order):
            v = metrics_by_split.get(s, {}).get(k)
            c.drawRightString(x + col_metric + col_w * (i + 1) - 6, y - 10, fmt(v))
        y -= row_h

    y -= 6
    return y


def write_iteration_pdf(
    out_pdf: Path,
    iteration: int,
    task_type: str,
    val_metric: Optional[str],
    report_text: Optional[str],
    metrics_by_split: Dict[str, Dict[str, float]],
    plot_groups: Dict[str, List[Path]],  # split -> list of plots
    split_order: List[str],
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    width, height = A4
    page_no = 1

    # ---- Title block
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 2.3 * cm, f"Report — Iteration {iteration}")
    c.setFont("Helvetica", 10)
    sub = f"Task: {task_type}"
    if val_metric:
        sub += f"   |   Optimized: {val_metric}"
    sub += f"   |   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    c.setFillColor(colors.grey)
    c.drawString(2 * cm, height - 3.0 * cm, sub)
    c.setFillColor(colors.black)

    y = height - 3.8 * cm

    # ---- Metrics table (ONLY place metrics appear)
    y = draw_metrics_table_3cols(
        c=c,
        x=2 * cm,
        y=y,
        metrics_by_split=metrics_by_split,
        split_order=split_order,
        title="Metrics",
    )

    _draw_page_number(c, page_no)

    # ---- Report text
    c.showPage()
    page_no += 1
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, height - 2.2 * cm, "Run report")
    c.setFont("Helvetica", 10)
    y = height - 3.0 * cm

    if report_text:
        clean = _strip_md_noise(report_text)
        for line in _wrap_preserve_empty(clean, 112):
            if y < 2.2 * cm:
                _draw_page_number(c, page_no)
                c.showPage()
                page_no += 1
                c.setFont("Helvetica", 10)
                y = height - 2.2 * cm
            c.drawString(2 * cm, y, line)
            y -= 12
    else:
        c.setFillColor(colors.grey)
        c.drawString(2 * cm, y, "No report markdown found for this iteration.")
        c.setFillColor(colors.black)

    _draw_page_number(c, page_no)

    # ---- Plots: include train/validation/test, each section titled
    if any(plot_groups.get(s) for s in split_order):
        c.showPage()
        page_no += 1
        y = height - 2.2 * cm

        img_w = width - 4 * cm
        img_h = 8.2 * cm

        for split in split_order:
            plots = plot_groups.get(split, [])
            if not plots:
                continue

            # section title
            c.setFont("Helvetica-Bold", 14)
            c.drawString(2 * cm, y, f"Plots — {split.title()}")
            y -= 18
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.grey)
            c.drawString(2 * cm, y, "Generated from labeled data + prediction outputs (when available).")
            c.setFillColor(colors.black)
            y -= 14

            for p in plots:
                if not p.exists():
                    continue

                # new page if needed
                if y - img_h < 2.2 * cm:
                    _draw_page_number(c, page_no)
                    c.showPage()
                    page_no += 1
                    y = height - 2.2 * cm

                c.setFont("Helvetica", 9)
                c.setFillColor(colors.grey)
                c.drawString(2 * cm, y, p.name)
                c.setFillColor(colors.black)
                y -= 10

                c.drawImage(
                    str(p),
                    2 * cm,
                    y - img_h,
                    width=img_w,
                    height=img_h,
                    preserveAspectRatio=True,
                    anchor="n",
                )
                y -= (img_h + 18)

            # spacing before next split section
            y -= 6
            if y < 4 * cm:
                _draw_page_number(c, page_no)
                c.showPage()
                page_no += 1
                y = height - 2.2 * cm

        _draw_page_number(c, page_no)

    c.save()


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", type=Path, required=True, help="Path to outputs/<agent_id>")
    args = ap.parse_args()

    agent_dir: Path = args.agent_dir.resolve()
    if not agent_dir.exists():
        raise SystemExit(f"Agent dir not found: {agent_dir}")

    task_type = load_task_type(agent_dir)
    val_metric = load_val_metric(agent_dir)
    iterations = discover_iterations(agent_dir)
    if not iterations:
        raise SystemExit("No iteration folders found under run_files/iteration_*")

    out_dir = agent_dir / "final_reports"
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    # preferred ordering in tables/plots
    split_order = ["train", "validation", "test"]

    for it in iterations:
        inp = gather_iteration_inputs(agent_dir, it)
        report_text = inp.report_md.read_text(encoding="utf-8") if inp.report_md else None

        # metrics map: split -> metrics
        metrics_by_split: Dict[str, Dict[str, float]] = {
            s.split_name: s.metrics for s in inp.splits if s.metrics
        }

        # plots per split (train/val/test if possible)
        plot_groups: Dict[str, List[Path]] = {}
        for s in inp.splits:
            # Only plot if we have labeled + preds
            plots = build_plots_for_split(
                task_type=task_type,
                split_name=s.split_name,
                labeled_csv=s.labeled_csv,
                preds_csv=s.preds_csv,
                plots_dir=plots_dir,
                iteration=it,
            )
            if plots:
                plot_groups[s.split_name] = plots

        out_pdf = out_dir / f"iteration_{it}.pdf"
        write_iteration_pdf(
            out_pdf=out_pdf,
            iteration=it,
            task_type=task_type,
            val_metric=val_metric,
            report_text=report_text,
            metrics_by_split=metrics_by_split,
            plot_groups=plot_groups,
            split_order=split_order,
        )
        print(f"Wrote: {out_pdf}")

    print(f"\nDone.\nPDFs: {out_dir}\nPlots: {plots_dir}\n")


if __name__ == "__main__":
    main()
