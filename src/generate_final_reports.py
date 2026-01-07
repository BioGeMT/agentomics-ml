#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
# Headless safe (works inside Docker)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# Matplotlib style
def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.black,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.black,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Muted",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.Color(0.25, 0.25, 0.25),  # dark slate grey
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Quote",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9.8,
        leading=13,
        textColor=colors.Color(0.15, 0.15, 0.15),
        leftIndent=14,
        spaceBefore=4,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="MiniHeader",
        parent=styles["Body"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=9,
        leading=11,
        textColor=colors.black,
        backColor=colors.Color(0.96, 0.96, 0.96),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8,
    ))
    return styles

def apply_pub_style() -> None:
    plt.rcParams.update({
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
    })

# Helpers
def _esc(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))

def step_body_to_flowables(text: str, styles):
    flows = []
    buf = []
    if "MiniHeader" not in styles:
        styles.add(ParagraphStyle(
            name="MiniHeader",
            parent=styles["Body"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=colors.black,
            spaceBefore=6,
            spaceAfter=3,
        ))

    def flush_paragraph():
        nonlocal buf
        if not buf:
            return
        ptxt = " ".join([x.strip() for x in buf]).strip()
        if ptxt:
            flows.append(Paragraph(_esc(ptxt), styles["Body"]))
        buf = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush_paragraph()
            flows.append(Spacer(1, 4))
            continue
        if line.lstrip().startswith(">"):
            flush_paragraph()
            q = line.lstrip()[1:].lstrip()
            flows.append(Paragraph(_esc(q), styles["Quote"]))
            continue
        s = line.strip()
        if s.endswith(":") and len(s) <= 40 and " " in s:
            flush_paragraph()
            flows.append(Paragraph(_esc(s), styles["MiniHeader"]))
            continue
        if re.match(r"^(\/workspace\/|Path To |Train Path:|Val Path:|Test Path:)", line):
            flush_paragraph()
            flows.append(Paragraph(_esc(line), styles["CodeBlock"]))
            continue

        buf.append(line)

    flush_paragraph()
    return flows

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
    for c in ["numeric_label", "numericlabel", "class", "target", "label", "y"]:
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

# Task detection + config
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

# Iteration discovery / inputs
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

    train_csv = run_files / "train.csv"
    val_csv = run_files / "validation.csv"
    test_csv = run_files / "test.csv"  # may not exist

    if not train_csv.exists():
        train_csv = None
    if not val_csv.exists():
        val_csv = None
    if not test_csv.exists():
        test_csv = None

    # predictions
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

    test_metrics = parse_metrics_txt(test_metrics_path)
    # include test if ANY of these exist
    if (test_csv and test_csv.exists()) or (test_preds and test_preds.exists()) or bool(test_metrics):
        splits.append(SplitArtifacts("test", test_csv, test_preds, test_metrics))

    return IterationInputs(iteration=iteration, report_md=report_md, splits=splits)

# Plotting
def _guess_task_from_labels(y: pd.Series) -> str:
    yy = pd.to_numeric(y, errors="coerce").dropna()
    uniq = set(yy.unique().tolist())
    if uniq.issubset({0, 1}):
        return "classification"
    return "regression"

def plot_regression_publication(y_true: pd.Series, y_pred: pd.Series, out_prefix: Path, title_prefix: str) -> List[Path]:
    apply_pub_style()

    y_true = _as_num(y_true)
    y_pred = _as_num(y_pred)
    df = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
    if df.empty:
        return []
    pva_path = out_prefix.with_name(out_prefix.name + "_pred_vs_actual.png")
    lo, hi = _robust_limits(df["y"], df["p"])
    plt.figure()
    plt.scatter(df["y"], df["p"], s=18, alpha=0.85, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, alpha=0.9)  # y=x
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title_prefix} - Predicted vs Actual")
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

    rvp_path = out_prefix.with_name(out_prefix.name + "_residuals_vs_pred.png")
    plt.figure()
    plt.scatter(df["p"], resid, s=18, alpha=0.85, edgecolors="none")
    plt.axhline(0, linestyle="--", linewidth=1.2, alpha=0.9)
    plt.ylim(rlo, rhi)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (pred − actual)")
    plt.title(f"{title_prefix} - Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(rvp_path)
    plt.close()

    rh_path = out_prefix.with_name(out_prefix.name + "_residuals_hist.png")
    plt.figure()
    plt.hist(resid.dropna(), bins=min(30, max(8, int(len(resid) / 2))), alpha=0.9)
    plt.axvline(0, linestyle="--", linewidth=1.2, alpha=0.9)
    plt.xlabel("Residual (pred − actual)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} - Residual distribution")
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
    plt.title(f"{title_prefix} - ROC curve")
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
    plt.title(f"{title_prefix} - Precision–Recall curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    return [roc_path, pr_path]


def build_plots_for_split(
    task_type: str,
    split_name: str,
    labeled_csv: Optional[Path],
    preds_csv: Optional[Path],
    plots_dir: Path,
    iteration: int,
) -> List[Path]:
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


# -------------------------
# Markdown cleanup + STEPS extraction
# -------------------------
DETAILS_BLOCK_RE = re.compile(r"<details>.*?</details>", flags=re.DOTALL | re.IGNORECASE)

def _remove_details_blocks(md: str) -> str:
    return re.sub(DETAILS_BLOCK_RE, "", md)

def _remove_metrics_section(md: str) -> str:
    # Remove "## Metrics" section completely (until next "## " heading or end)
    lines = md.replace("\r\n", "\n").split("\n")
    out: List[str] = []
    skipping = False
    for line in lines:
        if re.match(r"^\s*##\s+metrics\s*$", line, flags=re.IGNORECASE):
            skipping = True
            continue
        if skipping and re.match(r"^\s*##\s+", line):
            skipping = False
        if skipping:
            continue
        out.append(line)
    return "\n".join(out)

def _remove_inline_metric_bullets(md: str) -> str:
    # Remove bullet lines like "- **train/MSE**: ..." or "• validation/ACC: ..."
    lines = md.replace("\r\n", "\n").split("\n")
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if re.match(r"^[-•]\s+\*\*(train|validation|test)/", s, flags=re.IGNORECASE):
            continue
        if re.match(r"^[-•]\s+(train|validation|test)/", s, flags=re.IGNORECASE):
            continue
        out.append(line)
    return "\n".join(out)

def _strip_basic_md(md: str) -> str:
    s = md.replace("\r\n", "\n")
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    return s

def clean_report_md(md: str) -> str:
    s = _remove_details_blocks(md)
    s = _remove_metrics_section(s)
    s = _remove_inline_metric_bullets(s)
    s = _strip_basic_md(s)
    return s.strip()

@dataclass
class Step:
    title: str
    body: str

def extract_steps(md: str) -> Tuple[List[str], List[Step]]:
    s = md.replace("\r\n", "\n")
    lines = s.split("\n")

    if lines and re.match(r"^\s*#\s+", lines[0]):
        lines = lines[1:]

    first_h2 = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*##\s+", line):
            first_h2 = i
            break

    run_info = []
    if first_h2 is None:
        run_info = [ln.strip() for ln in lines if ln.strip()]
        return run_info, []
    else:
        run_info = [ln.strip() for ln in lines[:first_h2] if ln.strip() and ln.strip() != "---"]

    steps: List[Step] = []
    cur_title = None
    cur_body: List[str] = []

    def flush():
        nonlocal cur_title, cur_body
        if cur_title is None:
            return
        body_txt = "\n".join(cur_body).strip()
        steps.append(Step(title=cur_title.strip(), body=body_txt))
        cur_title, cur_body = None, []

    for line in lines[first_h2:]:
        m = re.match(r"^\s*##\s+(.*)$", line)
        if m:
            flush()
            cur_title = m.group(1)
            continue
        cur_body.append(line)

    flush()
    return run_info, steps

# PDF rendering helpers
def _draw_page_number(c: canvas.Canvas, page_no: int) -> None:
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawRightString(A4[0] - 1.5 * cm, 1.2 * cm, f"Page {page_no}")
    c.setFillColor(colors.black)

def _wrap_preserve_empty(text: str, width: int) -> List[str]:
    out: List[str] = []
    for raw in str(text).splitlines():
        if raw.strip() == "":
            out.append("")
        else:
            out.extend(textwrap.wrap(raw, width=width))
    return out

def _fmt_metric(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if abs(v) >= 1e6 or (abs(v) > 0 and abs(v) < 1e-3):
            return f"{v:.3g}"
        return f"{v:.6g}"
    return str(v)

def metrics_table_flowable(metrics_by_split, split_order, val_metric, styles):
    present = [s for s in split_order if metrics_by_split.get(s)]
    keys = sorted({k for s in present for k in metrics_by_split[s].keys()})
    if not present or not keys:
        return Paragraph("No metrics found.", styles["Muted"])

    data = [["Metric"] + [s.title() for s in present]]
    for k in keys:
        row = [k] + [_fmt_metric(metrics_by_split[s].get(k)) for s in present]
        data.append(row)

    tbl = Table(data, hAlign="LEFT")
    ts = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.Color(0.97,0.97,0.97)]),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
    ])

    # highlight the optimized metric row if present
    if val_metric and val_metric in keys:
        r = 1 + keys.index(val_metric)
        ts.add("BACKGROUND", (0,r), (-1,r), colors.Color(1.0, 0.98, 0.90))

    tbl.setStyle(ts)
    return tbl



def draw_steps_page(
    c: canvas.Canvas,
    width: float,
    height: float,
    page_no_start: int,
    run_info: List[str],
    steps: List[Step],
) -> int:
    page_no = page_no_start
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, height - 2.2 * cm, "STEPS")
    y = height - 3.0 * cm

    # Run info block (small, grey)
    if run_info:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        for line in _wrap_preserve_empty("\n".join(run_info), 120):
            if y < 2.2 * cm:
                _draw_page_number(c, page_no)
                c.showPage()
                page_no += 1
                c.setFont("Helvetica", 9)
                c.setFillColor(colors.grey)
                y = height - 2.2 * cm
            c.drawString(2 * cm, y, line)
            y -= 11
        c.setFillColor(colors.black)
        y -= 6

    if not steps:
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.grey)
        c.drawString(2 * cm, y, "No step headings (## ...) found in the run report.")
        c.setFillColor(colors.black)
        return page_no

    for idx, step in enumerate(steps, start=1):
        if y < 3.0 * cm:
            _draw_page_number(c, page_no)
            c.showPage()
            page_no += 1
            y = height - 2.2 * cm

        # Step title
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y, f"{idx}. {step.title.strip()}")
        y -= 14

        # Step body
        body = step.body.strip()
        if body:
            c.setFont("Helvetica", 10)
            for line in _wrap_preserve_empty(body, 112):
                if y < 2.2 * cm:
                    _draw_page_number(c, page_no)
                    c.showPage()
                    page_no += 1
                    c.setFont("Helvetica", 10)
                    y = height - 2.2 * cm
                c.drawString(2 * cm, y, line)
                y -= 12
        else:
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.grey)
            c.drawString(2 * cm, y, "(No details provided.)")
            c.setFillColor(colors.black)
            y -= 12

        y -= 6

    return page_no


def write_iteration_pdf(
    out_pdf: Path,
    iteration: int,
    task_type: str,
    val_metric: Optional[str],
    report_text_raw: Optional[str],
    metrics_by_split: Dict[str, Dict[str, float]],
    plot_groups: Dict[str, List[Path]],  # split -> plots
    split_order: List[str],
) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=1.6 * cm
    )

    story = []
    story.append(Paragraph(f"Run Report | Iteration {iteration}", styles["H1"]))
    story.append(
        Paragraph(f"Task: {task_type} &nbsp;&nbsp;|&nbsp;&nbsp; Optimized: {val_metric or '—'}", styles["Muted"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Metrics", styles["H2"]))
    story.append(metrics_table_flowable(metrics_by_split, split_order, val_metric, styles))
    story.append(PageBreak())

    # Steps
    if report_text_raw:
        cleaned = clean_report_md(report_text_raw)  # update cleaner as noted above
        run_info, steps = extract_steps(cleaned)

        # Optional: summary extraction (from the “Summary” step)
        # Optional: filter the “Summarizing user request” meta-lines for classification
        story.append(Paragraph("Steps", styles["H1"]))
        for i, st in enumerate(steps, start=1):
            story.append(Paragraph(f"{i}. {st.title}", styles["H2"]))
            story.extend(step_body_to_flowables(st.body, styles))
            story.append(Spacer(1, 6))

    # Plots
    for split in split_order:
        plots = plot_groups.get(split, [])
        if not plots:
            continue
        story.append(PageBreak())
        story.append(Paragraph(f"Plots - {split.title()}", styles["H1"]))
        story.append(Paragraph("Generated from labeled data + prediction outputs (when available).", styles["Muted"]))
        story.append(Spacer(1, 8))

        for p in plots:
            if not p.exists():
                continue
            story.append(
                Paragraph(p.stem.replace(f"iter_{iteration}_{split}_", "").replace("_", " ").title(), styles["Muted"]))
            story.append(RLImage(str(p), width=A4[0] - 4 * cm, height=8.5 * cm))
            story.append(Spacer(1, 10))

    def on_page(c, d):
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        c.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Page {d.page}")
        c.setFillColor(colors.black)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    # If MPLCONFIGDIR was not set, ensure Matplotlib doesn't try to write to /.config
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

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

    split_order = ["train", "validation", "test"]

    for it in iterations:
        inp = gather_iteration_inputs(agent_dir, it)
        report_text = inp.report_md.read_text(encoding="utf-8") if inp.report_md else None

        metrics_by_split: Dict[str, Dict[str, float]] = {
            s.split_name: s.metrics for s in inp.splits if s.metrics
        }

        plot_groups: Dict[str, List[Path]] = {}
        for s in inp.splits:
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
            report_text_raw=report_text,
            metrics_by_split=metrics_by_split,
            plot_groups=plot_groups,
            split_order=split_order,
        )
        print(f"Wrote: {out_pdf}")

    print(f"\nDone.\nPDFs: {out_dir}\nPlots: {plots_dir}\n")


if __name__ == "__main__":
    main()
