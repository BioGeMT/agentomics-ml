from __future__ import annotations

import argparse
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
from agentomics.runtime.all_iterations_summary import write_all_iterations_summary
from agentomics.runtime.report_data import (
    DatasetMeta,
    RunMeta,
    SplitArtifacts,
    gather_iteration_inputs,
    load_dataset_meta_from_config,
    load_run_meta,
)
from agentomics.runtime.report_styles import (
    REPORT_PALETTE,
    build_styles,
    escape_report_text as _esc,
    format_metric as _fmt_metric,
    pdf_color as _pdf_color,
    run_meta_flowables,
)
from agentomics.utils.config import Config
from agentomics.utils.task_types import TaskTypes
from agentomics.runtime.iteration_reports import write_iteration_report
from agentomics.runtime.read_write_utils import (
    get_archived_iterations,
    load_best_iteration_snapshot_iteration,
    load_config_from_run_dir_and_reroot,
)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage,
)

@dataclass
class Step:
    title: str
    body: str

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
