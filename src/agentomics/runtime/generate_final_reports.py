from __future__ import annotations

import argparse
import json
import os
import re
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from agentomics.agents.steps.validation_evaluation import ValidationEvaluationStep
from agentomics.runtime.step_outputs import load_step_output
from agentomics.runtime.plots_helpers import render_plots
from agentomics.utils.config import Config
from agentomics.utils.ml_tasks import ClassificationTask, get_task
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
    TableStyle,
    PageBreak,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from agentomics.datasets.data_contract import LABELS_FILE_NAME, NUMERIC_LABEL_COLUMN_NAME, TRAIN_SPLIT, VALIDATION_SPLIT

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
            textColor=colors.black,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=colors.black,
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
            textColor=colors.black,
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
            textColor=colors.Color(0.25, 0.25, 0.25),
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
            textColor=colors.Color(0.15, 0.15, 0.15),
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
            textColor=colors.black,
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
            textColor=colors.black,
            backColor=colors.Color(0.96, 0.96, 0.96),
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles

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
    if config.task_type == ClassificationTask.name and not config.label_to_scalar:
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

def build_plots_for_split(
    meta: DatasetMeta,
    split: SplitArtifacts,
    plots_dir: Path,
    iteration: int,
) -> Dict[str, Path]:
    labeled = safe_read_csv(split.labeled_csv)
    preds = safe_read_csv(split.preds_csv)
    if labeled is None or preds is None:
        return {}

    try:
        plot_data = get_task(meta.task_type).get_plot_data(
            labeled, preds,
            {"numeric_label_col": meta.numeric_label_col, "label_to_scalar": meta.label_to_scalar},
        )
    except Exception as e:
        print(f"[PLOTS SKIP] iter={iteration} split={split.split_name} :: {e}")
        return {}

    prefix = plots_dir / f"iter_{iteration}_{split.split_name}"
    return render_plots(plot_data, prefix, split.split_name.title())

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
    plot_groups: Dict[str, Dict[str, Path]],
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
                    plot_groups,
                    present_splits[start : start + 3],
                    styles,
                )
            )
        return flows

    plot_names = list(dict.fromkeys(
        name for split in present_splits for name in plot_groups[split]
    ))

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
    for plot_name in plot_names:
        row = []
        for split in present_splits:
            p = plot_groups[split].get(plot_name)
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
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
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
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
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
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
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
                    colors.Color(1.0, 0.98, 0.90),
                )
    elif val_metric and val_metric in keys:
        row_index = 1 + keys.index(val_metric)
        ts.add(
            "BACKGROUND",
            (0, row_index),
            (-1, row_index),
            colors.Color(1.0, 0.98, 0.90),
        )

    tbl.setStyle(ts)
    return tbl

def write_iteration_pdf(
    out_pdf: Path,
    iteration: int,
    run_meta: RunMeta,
    report_text_raw: Optional[str],
    metrics_by_split: Dict[str, Dict[str, float]],
    plot_groups: Dict[str, Dict[str, Path]],
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
        c.setFillColor(colors.grey)
        c.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Page {d.page}")
        c.setFillColor(colors.black)

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

        plot_groups: Dict[str, Dict[str, Path]] = {}
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
