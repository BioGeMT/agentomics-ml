from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agentomics.agents.steps.validation_evaluation import ValidationEvaluationStep
from agentomics.runtime.step_outputs import load_step_output
from agentomics.utils.config import Config
from agentomics.utils.task_types import TaskTypes
from agentomics.runtime.iteration_reports import write_iteration_report
from agentomics.runtime.read_write_utils import get_archived_iterations, load_best_iteration_snapshot_iteration, load_config_from_run_dir_and_reroot

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    LongTable,
    TableStyle,
    PageBreak,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from agentomics.datasets.data_contract import LABELS_FILE_NAME, NUMERIC_LABEL_COLUMN_NAME, TRAIN_SPLIT, VALIDATION_SPLIT

# Shared palette for report PDFs and plots. The high-contrast navy and teal
# carry the main structure; orange and vermilion distinguish secondary series.
REPORT_PALETTE = {
    "navy": "#17324D",
    "teal": "#007F86",
    "green": "#00876C",
    "orange": "#E69F00",
    "vermilion": "#D55E00",
    "slate": "#526575",
    "grid": "#D5DEE5",
    "pale_blue": "#EAF1F5",
    "pale_teal": "#DCEFEA",
    "pale_gold": "#FFF1CC",
    "pale_gray": "#F3F6F8",
    "white": "#FFFFFF",
}


def _pdf_color(name: str):
    return colors.HexColor(REPORT_PALETTE[name])


# Data models
@dataclass(frozen=True)
class DatasetMeta:
    task_type: str
    numeric_label_col: str
    label_to_scalar: Optional[Dict[str, int]] = None

@dataclass(frozen=True)
class RunMeta:
    agent_id: str
    model_name: str
    dataset: str
    task_type: str
    val_metric: Optional[str]
    split_allowed_iterations: Optional[int]
    exploration_iterations: Optional[int]

@dataclass
class SplitArtifacts:
    split_name: str  # train/validation/test*
    labeled_csv: Optional[Path]
    preds_csv: Optional[Path]
    metrics: Dict[str, float]

@dataclass
class IterationInputs:
    iteration: int
    report_md: Optional[Path]
    splits: List[SplitArtifacts]

@dataclass(frozen=True)
class IterationSummary:
    iteration: int
    approach: str
    split_version: int | None
    validation_value: float | None
    change_from_previous: float | None
    split_changed: bool
    is_best: bool

@dataclass
class Step:
    title: str
    body: str

# Styles
def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=_pdf_color("navy"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=_pdf_color("teal"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=_pdf_color("slate"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Muted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            textColor=_pdf_color("slate"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Quote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.8,
            leading=13,
            textColor=_pdf_color("navy"),
            leftIndent=14,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MiniHeader",
            parent=styles["Body"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            textColor=_pdf_color("teal"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=9,
            leading=11,
            textColor=_pdf_color("navy"),
            backColor=_pdf_color("pale_gray"),
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles

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
            "grid.color": REPORT_PALETTE["grid"],
            "grid.alpha": 0.8,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "axes.prop_cycle": plt.cycler(
                color=[
                    REPORT_PALETTE["teal"],
                    REPORT_PALETTE["vermilion"],
                    REPORT_PALETTE["navy"],
                    REPORT_PALETTE["orange"],
                    REPORT_PALETTE["green"],
                ]
            ),
            "axes.labelcolor": REPORT_PALETTE["navy"],
            "xtick.color": REPORT_PALETTE["slate"],
            "ytick.color": REPORT_PALETTE["slate"],
            "text.color": REPORT_PALETTE["navy"],
            "figure.facecolor": REPORT_PALETTE["white"],
            "axes.facecolor": REPORT_PALETTE["white"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )

# IO helpers
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_validation_metrics(config: Config, iter_dir: Path) -> Dict[str, float]:
    validation_output = load_step_output(
        config=config,
        step_id=ValidationEvaluationStep.step_id,
        iteration_dir=iter_dir,
    )
    if validation_output is None:
        return {}
    metrics = getattr(validation_output, "metrics", None)
    if not isinstance(metrics, dict):
        return {}
    return {str(key): float(value) for key, value in metrics.items()}

def get_split_metrics(metrics: Dict[str, float], split_name: str) -> Dict[str, float]:
    split_prefix = f"{split_name}/"
    return {
        metric_name.removeprefix(split_prefix): metric_value
        for metric_name, metric_value in metrics.items()
        if metric_name.startswith(split_prefix)
    }

def load_saved_metrics(metrics_path: Path) -> Dict[str, float]:
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {metrics_path}.")
    return {str(metric_name): float(metric_value) for metric_name, metric_value in payload.items()}

def load_run_meta(config: Config) -> RunMeta:
    return RunMeta(
        agent_id=config.agent_id,
        model_name=config.model_name,
        dataset=config.dataset,
        task_type=str(config.task_type).strip().lower(),
        val_metric=config.val_metric,
        split_allowed_iterations=config.split_allowed_iterations,
        exploration_iterations=config.exploration_iterations,
    )

def load_dataset_meta_from_config(config: Config) -> DatasetMeta:
    if config.task_type == TaskTypes.CLASSIFICATION and not config.label_to_scalar:
        raise SystemExit("Classification run config must include non-empty 'label_to_scalar'")
    return DatasetMeta(
        task_type=config.task_type,
        numeric_label_col=NUMERIC_LABEL_COLUMN_NAME,
        label_to_scalar=config.label_to_scalar,
    )

# Iteration discovery / inputs
def _find_versioned_split_labels(config: Config, split_name: str, split_version: int | None) -> Optional[Path]:
    if split_version is None:
        return None
    candidate = config.splits_dir / f"split_{split_version}" / split_name / LABELS_FILE_NAME
    return candidate if candidate.exists() else None

def _get_iteration_split_version(config: Config, iter_dir: Path) -> int | None:
    data_split_output = load_step_output(config, "data_split", iter_dir)
    return getattr(data_split_output, "split_version", None)

def gather_iteration_inputs(
    config: Config,
    iteration: int,
) -> IterationInputs:
    iter_dir = config.iteration_dir(iteration)
    validation_metrics = load_validation_metrics(config, iter_dir)

    report_md = config.markdown_reports_dir / f"run_report_iter_{iteration}.md"
    if not report_md.exists():
        report_md = None

    split_version = _get_iteration_split_version(config, iter_dir)
    train_csv = _find_versioned_split_labels(config, TRAIN_SPLIT, split_version)
    val_csv = _find_versioned_split_labels(config, VALIDATION_SPLIT, split_version)

    validation_evaluation_dir = iter_dir / ValidationEvaluationStep.step_id
    train_preds = validation_evaluation_dir / "eval_predictions_train.csv"
    val_preds = validation_evaluation_dir / "eval_predictions_validation.csv"

    splits: List[SplitArtifacts] = [
        SplitArtifacts("train", train_csv, train_preds, get_split_metrics(validation_metrics, "train")),
        SplitArtifacts("validation", val_csv, val_preds, get_split_metrics(validation_metrics, "validation")),
    ]

    best_iter = load_best_iteration_snapshot_iteration(config)
    if best_iter is not None and iteration == best_iter:
        snapshot_dir = config.best_iteration_snapshot_dir
        for test_metrics_path in sorted(snapshot_dir.glob("eval_predictions_test*.metrics.json")):
            predictions_stem = test_metrics_path.name.removesuffix(".metrics.json")
            split_name = predictions_stem.removeprefix("eval_predictions_")
            test_preds = snapshot_dir / f"{predictions_stem}.csv"
            test_labeled = snapshot_dir / f"{predictions_stem}.numeric_labels.csv"
            test_metrics = load_saved_metrics(test_metrics_path)
            splits.append(
                SplitArtifacts(split_name, test_labeled, test_preds, test_metrics)
            )

    return IterationInputs(iteration=iteration, report_md=report_md, splits=splits)

# Deterministic merge labels + predictions (metadata-driven)
def merge_labels_and_preds(labeled: pd.DataFrame, preds: pd.DataFrame, meta: DatasetMeta) -> Tuple[pd.Series, pd.Series]:
    if "id" not in labeled.columns or "id" not in preds.columns:
        raise ValueError("Both labeled and predictions CSV must include an 'id' column.")

    y_col = meta.numeric_label_col
    if y_col not in labeled.columns:
        raise ValueError(f"Label column '{y_col}' not found in labeled CSV. Available: {list(labeled.columns)}")

    if meta.task_type == TaskTypes.REGRESSION:
        pred_col = "prediction"
        if pred_col not in preds.columns:
            raise ValueError(f"Expected regression prediction column '{pred_col}'. Available: {list(preds.columns)}")
    else:
        pos_scalar = max(meta.label_to_scalar.values())  # authoritative positive class
        pred_col = f"probability_{pos_scalar}"
        if pred_col not in preds.columns:
            raise ValueError(
                f"Expected classification probability column '{pred_col}'. Available: {list(preds.columns)}"
            )

    merged = labeled[["id", y_col]].merge(preds[["id", pred_col]], on="id", how="inner")
    if merged.empty:
        raise ValueError("After merging on 'id', no rows matched between labeled and predictions CSV.")

    return merged[y_col], merged[pred_col]

# Plotting
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

def plot_regression(y_true: pd.Series, y_pred: pd.Series, out_prefix: Path, title_prefix: str) -> List[Path]:
    apply_pub_style()

    y_true = _as_num(y_true)
    y_pred = _as_num(y_pred)
    df = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
    if df.empty:
        return []

    out = []
    lo, hi = _robust_limits(df["y"], df["p"])

    pva_path = out_prefix.with_name(out_prefix.name + "_pred_vs_actual.png")
    plt.figure()
    plt.scatter(
        df["y"], df["p"], s=18, alpha=0.82, color=REPORT_PALETTE["teal"], edgecolors="none"
    )
    plt.plot(
        [lo, hi], [lo, hi], color=REPORT_PALETTE["orange"],
        linestyle="--", linewidth=1.2, alpha=0.95,
    )
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title_prefix} - Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(pva_path)
    plt.close()
    out.append(pva_path)

    resid = df["p"] - df["y"]
    rvp_path = out_prefix.with_name(out_prefix.name + "_residuals_vs_pred.png")
    plt.figure()
    plt.scatter(
        df["p"], resid, s=18, alpha=0.82, color=REPORT_PALETTE["teal"], edgecolors="none"
    )
    plt.axhline(
        0, color=REPORT_PALETTE["vermilion"], linestyle="--", linewidth=1.2, alpha=0.95
    )
    plt.xlabel("Predicted")
    plt.ylabel("Residual (pred − actual)")
    plt.title(f"{title_prefix} - Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(rvp_path)
    plt.close()
    out.append(rvp_path)

    rh_path = out_prefix.with_name(out_prefix.name + "_residuals_hist.png")
    plt.figure()
    plt.hist(
        resid.dropna(),
        bins=min(30, max(8, int(len(resid) / 2))),
        color=REPORT_PALETTE["teal"],
        alpha=0.86,
        edgecolor=REPORT_PALETTE["white"],
    )
    plt.axvline(
        0, color=REPORT_PALETTE["vermilion"], linestyle="--", linewidth=1.2, alpha=0.95
    )
    plt.xlabel("Residual (pred − actual)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} - Residual distribution")
    plt.tight_layout()
    plt.savefig(rh_path)
    plt.close()
    out.append(rh_path)

    return out

def plot_classification(y_true: pd.Series, y_score: pd.Series, out_prefix: Path, title_prefix: str) -> List[Path]:
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
    plt.plot(fpr, tpr, color=REPORT_PALETTE["teal"], linewidth=2.0)
    plt.plot(
        [0, 1], [0, 1], color=REPORT_PALETTE["slate"],
        linestyle="--", linewidth=1.1, alpha=0.75,
    )
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
    plt.plot(recall, precision, color=REPORT_PALETTE["vermilion"], linewidth=2.0)
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
    meta: DatasetMeta,
    split: SplitArtifacts,
    plots_dir: Path,
    iteration: int,
) -> List[Path]:
    labeled = safe_read_csv(split.labeled_csv)
    preds = safe_read_csv(split.preds_csv)
    if labeled is None or preds is None:
        return []

    try:
        y_true, y_pred = merge_labels_and_preds(labeled, preds, meta)
    except Exception as e:
        print(f"[PLOTS SKIP] iter={iteration} split={split.split_name} :: {e}")
        return []

    prefix = plots_dir / f"iter_{iteration}_{split.split_name}"
    title_prefix = split.split_name.title()

    if meta.task_type == TaskTypes.CLASSIFICATION:
        return plot_classification(y_true, y_pred, prefix, title_prefix)
    return plot_regression(y_true, y_pred, prefix, title_prefix)

# Markdown cleanup + step extraction
def _remove_section_by_h2(md: str, title: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: List[str] = []
    skipping = False
    for line in lines:
        if re.match(rf"^\s*##\s+{re.escape(title)}\s*$", line, flags=re.IGNORECASE):
            skipping = True
            continue
        if skipping and re.match(r"^\s*##\s+", line):
            skipping = False
        if skipping:
            continue
        out.append(line)
    return "\n".join(out)

def clean_report_md(md: str) -> str:
    s = md.replace("\r\n", "\n")
    s = _remove_section_by_h2(s, "Metrics")
    s = _remove_section_by_h2(s, "Test Metrics")
    s = re.sub(r"(?m)^\s*-\s*-\s+", "- ", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"(?m)^\s*#\s*Run Report.*\n?", "", s)
    s = re.sub(r"(?mi)^\s*(Agent ID|Dataset|Model|Task|Validation metric|Val Metric|Optimized Metric|Status|Generated)\s*:\s*.*\n?","",s)
    s = re.sub(r"(?m)^\s*---\s*$\n?", "", s)
    return s.strip()

def extract_steps(md: str) -> Tuple[List[str], List[Step]]:
    lines = md.replace("\r\n", "\n").split("\n")
    lines = [ln for ln in lines if not re.match(r"^\s*#\s+", ln)]
    first_h2 = next((i for i, ln in enumerate(lines) if re.match(r"^\s*##\s+", ln)), None)
    if first_h2 is None:
        return [ln.strip() for ln in lines if ln.strip()], []

    run_info = []
    drop_prefixes = (
        "agent id:",
        "dataset:",
        "model:",
        "task:",
        "validation metric:",
        "val metric:",
        "optimized metric:",
        "status:",
        "generated:",
    )
    for ln in lines[:first_h2]:
        s = ln.strip()
        if not s or s == "---":
            continue
        if s.lower().startswith(drop_prefixes):
            continue
        run_info.append(s)

    steps: List[Step] = []
    cur_title: Optional[str] = None
    cur_body: List[str] = []

    def flush():
        nonlocal cur_title, cur_body
        if cur_title is None:
            return
        steps.append(Step(title=cur_title.strip(), body="\n".join(cur_body).strip()))
        cur_title, cur_body = None, []

    for ln in lines[first_h2:]:
        m = re.match(r"^\s*##\s+(.*)$", ln)
        if m:
            flush()
            cur_title = m.group(1)
            continue
        cur_body.append(ln)

    flush()
    return run_info, steps

def prettify_step_title(t: str) -> str:
    # "Dataexploration" -> "Data exploration"
    t = re.sub(r"(?<!^)([A-Z])", r" \1", t).strip()
    t = re.sub(r"^Data(?=[A-Z])", "Data ", t)
    return t[:1].upper() + t[1:] if t else t

# Rendering helpers
def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def plots_compare_splits_page_flowables(
    iteration: int,
    task_type: str,
    plot_groups: Dict[str, List[Path]],
    split_order: List[str],
    styles,
) -> List:
    present_splits = [s for s in split_order if plot_groups.get(s)]
    if not present_splits:
        return []
    if len(present_splits) > 3:
        flows = []
        for start in range(0, len(present_splits), 3):
            if flows:
                flows.append(PageBreak())
            flows.extend(
                plots_compare_splits_page_flowables(
                    iteration,
                    task_type,
                    plot_groups,
                    present_splits[start : start + 3],
                    styles,
                )
            )
        return flows

    task_type = (task_type or "").strip().lower()
    if task_type == TaskTypes.CLASSIFICATION:
        row_suffixes = ["_roc", "_pr"]
    else:
        row_suffixes = ["_pred_vs_actual", "_residuals_vs_pred", "_residuals_hist"]

    # Build lookup: split -> suffix -> Path
    split_to_suffix: Dict[str, Dict[str, Path]] = {}
    for split in present_splits:
        m: Dict[str, Path] = {}
        for p in plot_groups.get(split, []):
            if not p or not p.exists():
                continue
            lname = p.name.lower()
            for suf in row_suffixes:
                if suf in lname:
                    m[suf] = p
        split_to_suffix[split] = m

    content_w = A4[0] - 4 * cm
    ncols = len(present_splits)
    gap = 0.25 * cm
    max_col_w = 8 * cm
    col_w = min((content_w - gap * (ncols - 1)) / ncols, max_col_w)

    aspect_ratio = 4.0 / 6.2  # matches matplotlib figure.figsize
    img_h = col_w * aspect_ratio

    flows: List = []
    # NOTE: removed forced PageBreak so plots can appear right after metrics if desired.
    flows.append(Paragraph(f"Plots comparison (Iteration {iteration})", styles["H1"]))
    flows.append(
        Paragraph(
            "Columns are dataset splits (train / validation / test if exists). "
            "Rows correspond to the same plot type across splits.",
            styles["Muted"],
        )
    )
    flows.append(Spacer(1, 8))
    table_data = [[s.title() for s in present_splits]]
    for suf in row_suffixes:
        row = []
        for split in present_splits:
            p = split_to_suffix.get(split, {}).get(suf)
            if p and p.exists():
                row.append(RLImage(str(p), width=col_w, height=img_h))
            else:
                row.append(Paragraph("—", styles["Muted"]))
        table_data.append(row)

    tbl = Table(
        table_data,
        colWidths=[col_w] * ncols,
        hAlign="LEFT",
    )
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 0), (-1, 0), _pdf_color("navy")),
        ("TEXTCOLOR", (0, 0), (-1, 0), _pdf_color("white")),
        ("GRID", (0, 0), (-1, -1), 0.35, _pdf_color("grid")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    flows.append(tbl)
    return flows

def run_meta_flowables(meta: RunMeta):
    rows = [
        ("Agent ID", meta.agent_id),
        ("Model", meta.model_name),
        ("Dataset", meta.dataset),
        ("Task", meta.task_type),
        ("Optimized Metric", meta.val_metric or "—"),
        ("Split Allowed Iterations", meta.split_allowed_iterations if meta.split_allowed_iterations is not None else "—"),
        ("Exploration Iterations", meta.exploration_iterations if meta.exploration_iterations is not None else "—"),
    ]
    data = [[k, str(v)] for k, v in rows]

    table = Table(data, colWidths=[5 * cm, 10 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (0, -1), _pdf_color("pale_blue")),
                ("TEXTCOLOR", (0, 0), (0, -1), _pdf_color("navy")),
                ("BACKGROUND", (1, 0), (1, -1), _pdf_color("white")),
                ("GRID", (0, 0), (-1, -1), 0.35, _pdf_color("grid")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table

def step_body_to_flowables(text: str, styles):
    flows = []
    buf: List[str] = []

    def flush_paragraph():
        nonlocal buf
        if not buf:
            return
        ptxt = " ".join(x.strip() for x in buf).strip()
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
            flows.append(Paragraph(_esc(line.lstrip()[1:].lstrip()), styles["Quote"]))
            continue
        if re.match(r"^\s*[-*]\s+\S+", line):
            flush_paragraph()
            flows.append(Paragraph(_esc(line.strip()), styles["Body"]))
            continue
        s = line.strip()
        if s.endswith(":") and len(s) <= 40 and " " in s:
            flush_paragraph()
            flows.append(Paragraph(_esc(s), styles["MiniHeader"]))
            continue
        if re.match(r"^(\/workspace\/|run\/|Path To |Train Path:|Val Path:|Test Path:)", line):
            flush_paragraph()
            flows.append(Paragraph(_esc(line), styles["CodeBlock"]))
            continue
        buf.append(line)

    flush_paragraph()
    return flows

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

    if len(present) > 3:
        data = [["Split", "Metric", "Value"]]
        for split in present:
            data.extend(
                [split, metric, _fmt_metric(value)]
                for metric, value in sorted(metrics_by_split[split].items())
            )
    else:
        data = [["Metric"] + [s.title() for s in present]]
        for k in keys:
            row = [k] + [_fmt_metric(metrics_by_split[s].get(k)) for s in present]
            data.append(row)

    tbl = Table(data, hAlign="LEFT")
    ts = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), _pdf_color("navy")),
            ("TEXTCOLOR", (0, 0), (-1, 0), _pdf_color("white")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.35, _pdf_color("grid")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_pdf_color("white"), _pdf_color("pale_gray")]),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]
    )

    if val_metric and len(present) > 3:
        for row_index, row in enumerate(data[1:], start=1):
            if row[1] == val_metric:
                ts.add(
                    "BACKGROUND",
                    (0, row_index),
                    (-1, row_index),
                    _pdf_color("pale_gold"),
                )
    elif val_metric and val_metric in keys:
        row_index = 1 + keys.index(val_metric)
        ts.add(
            "BACKGROUND",
            (0, row_index),
            (-1, row_index),
            _pdf_color("pale_gold"),
        )

    tbl.setStyle(ts)
    return tbl

def write_iteration_pdf(
    out_pdf: Path,
    iteration: int,
    run_meta: RunMeta,
    report_text_raw: Optional[str],
    metrics_by_split: Dict[str, Dict[str, float]],
    plot_groups: Dict[str, List[Path]],
    split_order: List[str],
) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=1.6 * cm,
    )

    story = []
    story.append(Paragraph(f"Run Report | Iteration {iteration}", styles["H1"]))
    story.append(Spacer(1, 6))
    story.append(run_meta_flowables(run_meta))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Metrics", styles["H2"]))
    story.append(metrics_table_flowable(metrics_by_split, split_order, run_meta.val_metric, styles))
    story.append(Spacer(1, 10))

    plot_section = plots_compare_splits_page_flowables(
        iteration=iteration,
        task_type=run_meta.task_type,
        plot_groups=plot_groups,
        split_order=split_order,
        styles=styles,
    )
    if plot_section:
        story.append(PageBreak())
        story.extend(plot_section)
    if report_text_raw:
        story.append(PageBreak())
        cleaned = clean_report_md(report_text_raw)
        run_info, steps = extract_steps(cleaned)

        if run_info:
            story.extend(step_body_to_flowables("\n".join(run_info), styles))
            story.append(Spacer(1, 8))

        for i, st in enumerate(steps, start=1):
            story.append(Paragraph(f"{i}. {prettify_step_title(st.title)}", styles["H2"]))
            story.extend(step_body_to_flowables(st.body, styles))
            story.append(Spacer(1, 6))

    def on_page(c, d):
        c.setFont("Helvetica", 8)
        c.setFillColor(_pdf_color("slate"))
        c.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Page {d.page}")
        c.setFillColor(_pdf_color("navy"))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def _iteration_approach(config: Config, iteration: int) -> str:
    output = load_step_output(config, "model_architecture", config.iteration_dir(iteration))
    architecture = getattr(output, "architecture", None)
    if not isinstance(architecture, str) or not architecture.strip():
        return "Not recorded"

    compact = " ".join(architecture.split())
    if len(compact) > 100:
        compact = compact[:97].rsplit(" ", 1)[0] + "…"
    return compact


def _metric_value(metrics: Dict[str, float], metric_name: str | None) -> float | None:
    if not metric_name:
        return None
    normalized_name = metric_name.casefold()
    for name, value in metrics.items():
        if name.casefold() == normalized_name:
            return value
    return None


def _all_iteration_summaries(
    config: Config,
    iterations: List[int],
    best_iteration: int | None,
    val_metric: str | None,
) -> List[IterationSummary]:
    rows: List[IterationSummary] = []
    previous_split_version: int | None = None
    previous_value: float | None = None
    for iteration in iterations:
        inputs = gather_iteration_inputs(config, iteration)
        validation = next(
            (split for split in inputs.splits if split.split_name == VALIDATION_SPLIT),
            None,
        )
        split_output = load_step_output(config, "data_split", config.iteration_dir(iteration))
        split_version = getattr(split_output, "split_version", None)
        validation_value = _metric_value(
            validation.metrics if validation else {}, val_metric
        )
        split_changed = (
            bool(rows)
            and split_version is not None
            and previous_split_version is not None
            and split_version != previous_split_version
        )
        split_comparable = (
            bool(rows)
            and split_version is not None
            and previous_split_version is not None
            and split_version == previous_split_version
        )
        change_from_previous = None
        if (
            split_comparable
            and validation_value is not None
            and previous_value is not None
        ):
            change_from_previous = validation_value - previous_value

        rows.append(
            IterationSummary(
                iteration=iteration,
                approach=_iteration_approach(config, iteration),
                split_version=split_version,
                validation_value=validation_value,
                change_from_previous=change_from_previous,
                split_changed=split_changed,
                is_best=iteration == best_iteration,
            )
        )
        previous_split_version = split_version
        previous_value = validation_value
    return rows


def _best_iteration_test_metrics(
    config: Config,
    best_iteration: int | None,
) -> List[Tuple[str, str, float]]:
    if best_iteration is None:
        return []
    inputs = gather_iteration_inputs(config, best_iteration)
    return [
        (split.split_name, metric_name, value)
        for split in inputs.splits
        if split.split_name.casefold().startswith("test")
        for metric_name, value in sorted(split.metrics.items())
    ]


def _pivot_test_metrics(
    test_metric_rows: List[Tuple[str, str, float]],
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], float]]:
    split_names = list(dict.fromkeys(split for split, _, _ in test_metric_rows))
    metric_names = sorted({metric for _, metric, _ in test_metric_rows})
    values = {
        (split, metric): value
        for split, metric, value in test_metric_rows
    }
    return split_names, metric_names, values


def _format_metric_change(summary: IterationSummary) -> str:
    if summary.split_changed:
        return "— (split changed)"
    if summary.change_from_previous is None:
        return "—"
    return f"{summary.change_from_previous:+.6g}"


def write_all_iterations_markdown(
    path: Path,
    run_meta: RunMeta,
    best_iteration: int | None,
    summaries: List[IterationSummary],
    test_metric_rows: List[Tuple[str, str, float]],
) -> None:
    best_summary = next((row for row in summaries if row.is_best), None)
    best_score = (
        _fmt_metric(best_summary.validation_value)
        if best_summary and best_summary.validation_value is not None
        else "not recorded"
    )
    lines = [
        "# All Iterations Summary",
        "",
        f"**Agent ID:** `{run_meta.agent_id}`  ",
        f"**Dataset:** `{run_meta.dataset}`  ",
        f"**Model:** `{run_meta.model_name}`  ",
        f"**Task:** `{run_meta.task_type}`  ",
        f"**Optimized metric:** `{run_meta.val_metric or 'Not specified'}`  ",
        f"**Best validation {run_meta.val_metric or 'score'}:** {best_score} "
        f"(iteration {best_iteration if best_iteration is not None else 'not identified'})  ",
        f"**Archived iterations:** {len(summaries)}",
        "",
        "Detailed reports are available for each iteration in `markdown/` and `pdf/`.",
        "",
        f"| Iteration | Model / approach | Split | Validation {run_meta.val_metric or 'metric'} | Change vs previous | Best |",
        "| ---: | :--- | :---: | ---: | ---: | :---: |",
    ]
    if summaries:
        lines.extend(
            "| {iteration} | {approach} | {split} | {value} | {change} | {best} |".format(
                iteration=summary.iteration,
                approach=summary.approach.replace("|", "\\|"),
                split=(f"split_{summary.split_version}" if summary.split_version is not None else "—"),
                value=_fmt_metric(summary.validation_value),
                change=_format_metric_change(summary),
                best="Yes" if summary.is_best else "",
            )
            for summary in summaries
        )
    else:
        lines.append("| — | — | — | No iteration data available | — | — |")
    lines.extend(
        [
            "",
            "Change is the current validation score minus the previous archived iteration's score. "
            "It is omitted when split versions differ, a split version is missing, or a score is missing.",
            "Other train and validation metrics are included in each detailed iteration report.",
            "",
            "## Held-out test results for the selected best iteration",
            "",
        ]
    )
    if test_metric_rows:
        test_splits, test_metrics, test_values = _pivot_test_metrics(test_metric_rows)
        lines.extend(
            [
                "These results are for the selected best iteration only.",
                "",
                "| Metric | " + " | ".join(test_splits) + " |",
                "| :--- | " + " | ".join("---:" for _ in test_splits) + " |",
            ]
        )
        lines.extend(
            "| " + metric_name + " | " + " | ".join(
                _fmt_metric(test_values.get((split, metric_name)))
                for split in test_splits
            ) + " |"
            for metric_name in test_metrics
        )
    else:
        lines.append("_No held-out test metrics were recorded for this run._")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_all_iterations_pdf(
    path: Path,
    run_meta: RunMeta,
    best_iteration: int | None,
    summaries: List[IterationSummary],
    test_metric_rows: List[Tuple[str, str, float]],
) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=1.6 * cm,
    )
    best_summary = next((row for row in summaries if row.is_best), None)
    best_score = (
        _fmt_metric(best_summary.validation_value)
        if best_summary and best_summary.validation_value is not None
        else "not recorded"
    )
    story = [
        Paragraph("All Iterations Summary", styles["H1"]),
        run_meta_flowables(run_meta),
        Spacer(1, 8),
        Paragraph(
            f"Best validation { _esc(run_meta.val_metric or 'score') }: "
            f"{_esc(best_score)} (iteration "
            f"{best_iteration if best_iteration is not None else 'not identified'}). "
            f"Archived iterations: {len(summaries)}.",
            styles["Body"],
        ),
        Paragraph(
            "Change is the current validation score minus the previous archived iteration's score. "
            "It is omitted when split versions differ, a split version is missing, or a score is missing. "
            "Detailed reports in markdown/ and pdf/ contain the other metrics and step details.",
            styles["Muted"],
        ),
        Spacer(1, 8),
    ]
    table_data = [["Iteration", "Model / approach", "Split", "Validation", "Change", "Best"]]
    if summaries:
        table_data.extend(
            [
                str(summary.iteration),
                Paragraph(_esc(summary.approach), styles["Muted"]),
                f"split_{summary.split_version}" if summary.split_version is not None else "—",
                _fmt_metric(summary.validation_value),
                _esc(_format_metric_change(summary)),
                "Yes" if summary.is_best else "",
            ]
            for summary in summaries
        )
    else:
        table_data.append(["—", "No iteration data available", "—", "—", "—", "—"])

    table = LongTable(
        table_data,
        colWidths=[1.4 * cm, 6.6 * cm, 2.0 * cm, 2.7 * cm, 2.8 * cm, 1.2 * cm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), _pdf_color("navy")),
        ("TEXTCOLOR", (0, 0), (-1, 0), _pdf_color("white")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, _pdf_color("grid")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_pdf_color("white"), _pdf_color("pale_gray")]),
        ("ALIGN", (0, 0), (2, -1), "LEFT"),
        ("ALIGN", (3, 1), (4, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index, summary in enumerate(summaries, start=1):
        if summary.is_best:
            table_style.extend(
                [
                    ("BACKGROUND", (0, row_index), (-1, row_index), _pdf_color("pale_teal")),
                    ("FONTNAME", (0, row_index), (0, row_index), "Helvetica-Bold"),
                    ("TEXTCOLOR", (0, row_index), (0, row_index), _pdf_color("navy")),
                ]
            )
    table.setStyle(TableStyle(table_style))
    story.append(table)
    if test_metric_rows:
        story.extend(
            [
                Spacer(1, 12),
                Paragraph("Held-out test results for the selected best iteration", styles["H2"]),
                Paragraph(
                    "Test results below are for the selected best iteration only.",
                    styles["Muted"],
                ),
            ]
        )
        test_splits, test_metrics, test_values = _pivot_test_metrics(test_metric_rows)
        test_table_data = [["Metric"] + [split.title() for split in test_splits]]
        test_table_data.extend(
            [metric_name]
            + [
                _fmt_metric(test_values.get((split, metric_name)))
                for split in test_splits
            ]
            for metric_name in test_metrics
        )
        available_width = A4[0] - 4 * cm
        metric_column_width = min(3.5 * cm, available_width * 0.3)
        split_column_width = (available_width - metric_column_width) / max(1, len(test_splits))
        test_table = LongTable(
            test_table_data,
            colWidths=[metric_column_width] + [split_column_width] * len(test_splits),
            repeatRows=1,
            hAlign="LEFT",
        )
        test_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _pdf_color("navy")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _pdf_color("white")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.35, _pdf_color("grid")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_pdf_color("white"), _pdf_color("pale_gray")]),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(test_table)
    else:
        story.extend(
            [
                Spacer(1, 12),
                Paragraph("Held-out test results", styles["H2"]),
                Paragraph("No held-out test metrics were recorded for this run.", styles["Muted"]),
            ]
        )
    doc.build(story)


def write_all_iterations_summary(
    config: Config,
    run_meta: RunMeta,
    iterations: List[int],
    best_iteration: int | None,
) -> None:
    summaries = _all_iteration_summaries(
        config, iterations, best_iteration, run_meta.val_metric
    )
    test_metric_rows = _best_iteration_test_metrics(config, best_iteration)
    write_all_iterations_markdown(
        config.reports_dir / "all_iterations.md",
        run_meta,
        best_iteration,
        summaries,
        test_metric_rows,
    )
    write_all_iterations_pdf(
        config.reports_dir / "all_iterations.pdf",
        run_meta,
        best_iteration,
        summaries,
        test_metric_rows,
    )

def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", type=Path, required=True, help="Path to outputs/<agent_id>")
    args = ap.parse_args()

    agent_dir: Path = args.agent_dir.resolve()
    if not agent_dir.exists():
        raise SystemExit(f"Agent dir not found: {agent_dir}")

    config = load_config_from_run_dir_and_reroot(agent_dir / Config.RUN_DIRNAME)
    run_meta = load_run_meta(config)
    dataset_meta = load_dataset_meta_from_config(config)

    iterations = get_archived_iterations(config)
    if not iterations:
        raise SystemExit("No iteration folders found.")

    out_dir = config.pdf_reports_dir
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)
    config.markdown_reports_dir.mkdir(parents=True, exist_ok=True)
    for it in iterations:
        inp = gather_iteration_inputs(config, it)
        split_order = [split.split_name for split in inp.splits]
        report_path = inp.report_md
        if report_path is None:
            report_metrics = {
                f"{split.split_name}/{metric_name}": metric_value
                for split in inp.splits
                if split.split_name != "test"
                for metric_name, metric_value in split.metrics.items()
            }
            test_metrics = next((split.metrics for split in inp.splits if split.split_name == "test"), None)
            report_path = write_iteration_report(
                config,
                iteration=it,
                iteration_dir=config.iteration_dir(it),
                report_path=config.markdown_reports_dir / f"run_report_iter_{it}.md",
                metrics=report_metrics,
                test_metrics=test_metrics,
            )
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else None

        metrics_by_split: Dict[str, Dict[str, float]] = {
            s.split_name: s.metrics for s in inp.splits if s.metrics
        }

        plot_groups: Dict[str, List[Path]] = {}
        for s in inp.splits:
            plots = build_plots_for_split(
                meta=dataset_meta,
                split=s,
                plots_dir=plots_dir,
                iteration=it,
            )
            if plots:
                plot_groups[s.split_name] = plots

        out_pdf = out_dir / f"iteration_{it}.pdf"
        write_iteration_pdf(
            out_pdf=out_pdf,
            iteration=it,
            run_meta=run_meta,
            report_text_raw=report_text,
            metrics_by_split=metrics_by_split,
            plot_groups=plot_groups,
            split_order=split_order,
        )

    best_iteration = load_best_iteration_snapshot_iteration(config)
    write_all_iterations_summary(config, run_meta, iterations, best_iteration)
    if best_iteration in iterations:
        shutil.copy2(
            config.markdown_reports_dir / f"run_report_iter_{best_iteration}.md",
            config.reports_dir / "best_iteration.md",
        )
        shutil.copy2(
            config.pdf_reports_dir / f"iteration_{best_iteration}.pdf",
            config.reports_dir / "best_iteration.pdf",
        )

if __name__ == "__main__":
    main()
