from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
    Image as RLImage,
)

from agentomics.datasets.data_contract import TRAIN_SPLIT, VALIDATION_SPLIT
from agentomics.runtime.report_data import RunMeta, gather_iteration_inputs
from agentomics.runtime.report_styles import (
    REPORT_PALETTE,
    build_styles,
    escape_report_text,
    format_metric,
    pdf_color,
    run_meta_flowables,
)
from agentomics.runtime.step_outputs import load_step_output
from agentomics.utils.config import Config
from agentomics.utils.metrics import get_higher_is_better_map


@dataclass(frozen=True)
class IterationSummary:
    iteration: int
    approach: str
    split_version: int | None
    validation_value: float | None
    validation_sample_count: int | None
    change_from_previous: float | None
    split_changed: bool
    is_best: bool


@dataclass(frozen=True)
class IterationMetricPoint:
    iteration: int
    split_version: int | None
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


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


def _evaluation_sample_count(split) -> int | None:
    if not split.labeled_csv or not split.preds_csv:
        return None
    if not split.labeled_csv.exists() or not split.preds_csv.exists():
        return None
    try:
        labels = pd.read_csv(
            split.labeled_csv,
            usecols=["id"],
            dtype={"id": str},
            keep_default_na=False,
        )
        predictions = pd.read_csv(
            split.preds_csv,
            usecols=["id"],
            dtype={"id": str},
            keep_default_na=False,
        )
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError):
        return None
    return len(predictions.merge(labels, on="id", how="inner"))


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
        validation_value = _metric_value(validation.metrics if validation else {}, val_metric)
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
        if split_comparable and validation_value is not None and previous_value is not None:
            change_from_previous = validation_value - previous_value

        rows.append(
            IterationSummary(
                iteration=iteration,
                approach=_iteration_approach(config, iteration),
                split_version=split_version,
                validation_value=validation_value,
                validation_sample_count=_evaluation_sample_count(validation),
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
) -> Tuple[List[Tuple[str, str, float]], Dict[str, int | None]]:
    if best_iteration is None:
        return [], {}
    inputs = gather_iteration_inputs(config, best_iteration)
    test_splits = [
        split for split in inputs.splits if split.split_name.casefold().startswith("test")
    ]
    test_sample_counts = {
        split.split_name: _evaluation_sample_count(split) for split in test_splits
    }
    test_metric_rows = [
        (split.split_name, metric_name, value)
        for split in test_splits
        for metric_name, value in sorted(split.metrics.items())
    ]
    return test_metric_rows, test_sample_counts


def _iteration_metric_points(
    config: Config,
    iterations: List[int],
    best_iteration: int | None,
) -> List[IterationMetricPoint]:
    points = []
    for iteration in iterations:
        inputs = gather_iteration_inputs(config, iteration)
        split_output = load_step_output(config, "data_split", config.iteration_dir(iteration))
        split_version = getattr(split_output, "split_version", None)
        split_metrics = {split.split_name: split.metrics for split in inputs.splits}
        points.append(
            IterationMetricPoint(
                iteration=iteration,
                split_version=split_version,
                train_metrics=split_metrics.get(TRAIN_SPLIT, {}),
                validation_metrics=split_metrics.get(VALIDATION_SPLIT, {}),
                test_metrics=(
                    {
                        metric_name: value
                        for split in inputs.splits
                        if split.split_name.casefold().startswith("test")
                        for metric_name, value in split.metrics.items()
                    }
                    if iteration == best_iteration
                    else {}
                ),
            )
        )
    return points


def _find_metric_value(metrics: Dict[str, float], metric_name: str) -> float | None:
    normalized_name = metric_name.casefold()
    for name, value in metrics.items():
        if name.casefold() == normalized_name:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def _plot_metric_segment(ax, segment, color: str, label: str) -> None:
    if segment:
        ax.plot(
            [point[0] for point in segment],
            [point[1] for point in segment],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=label,
        )


def _plot_split_metric_series(
    ax,
    points: List[IterationMetricPoint],
    metric_name: str,
    attribute: str,
    color: str,
    label: str,
) -> None:
    segment = []
    previous_point = None
    label_used = False
    for point in points:
        metrics = getattr(point, attribute)
        value = _find_metric_value(metrics, metric_name)
        if value is None:
            _plot_metric_segment(ax, segment, color, label if not label_used else "_nolegend_")
            label_used = label_used or bool(segment)
            segment = []
            previous_point = None
            continue
        if segment and (
            point.split_version != previous_point.split_version
            or point.iteration != previous_point.iteration + 1
        ):
            _plot_metric_segment(ax, segment, color, label if not label_used else "_nolegend_")
            label_used = True
            segment = []
        segment.append((point.iteration, value))
        previous_point = point
    _plot_metric_segment(ax, segment, color, label if not label_used else "_nolegend_")


def _plot_iteration_metric_trends(
    path: Path,
    points: List[IterationMetricPoint],
    best_iteration: int | None,
) -> Path | None:
    metric_names = sorted(
        {
            metric_name
            for point in points
            for metrics in (point.train_metrics, point.validation_metrics, point.test_metrics)
            for metric_name, value in metrics.items()
            if _find_metric_value({metric_name: value}, metric_name) is not None
        },
        key=str.casefold,
    )
    if not metric_names:
        return None

    columns = 2 if len(metric_names) > 1 else 1
    rows = math.ceil(len(metric_names) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11, max(3.0, rows * 2.6)),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for ax, metric_name in zip(flat_axes, metric_names):
        _plot_split_metric_series(
            ax, points, metric_name, "train_metrics", REPORT_PALETTE["teal"], "Train"
        )
        _plot_split_metric_series(
            ax,
            points,
            metric_name,
            "validation_metrics",
            REPORT_PALETTE["orange"],
            "Validation",
        )
        test_x = []
        test_y = []
        for point in points:
            value = _find_metric_value(point.test_metrics, metric_name)
            if value is not None:
                test_x.append(point.iteration)
                test_y.append(value)
        if test_x:
            ax.scatter(
                test_x,
                test_y,
                color=REPORT_PALETTE["vermilion"],
                marker="D",
                s=38,
                zorder=4,
                label="Test (best only)",
            )
        if best_iteration is not None and any(
            point.iteration == best_iteration for point in points
        ):
            ax.axvline(
                best_iteration,
                color=REPORT_PALETTE["slate"],
                linestyle=":",
                linewidth=1,
                label="Best iteration",
            )
        ax.set_title(metric_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="best")

    for ax in flat_axes[len(metric_names):]:
        ax.set_visible(False)

    figure.suptitle("Train and validation metrics across iterations", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _pivot_test_metrics(
    test_metric_rows: List[Tuple[str, str, float]],
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], float]]:
    split_names = list(dict.fromkeys(split for split, _, _ in test_metric_rows))
    metric_names = sorted({metric for _, metric, _ in test_metric_rows})
    values = {(split, metric): value for split, metric, value in test_metric_rows}
    return split_names, metric_names, values


def _format_metric_change(
    summary: IterationSummary,
    higher_is_better: bool | None,
) -> str:
    if summary.split_changed:
        return "— (split changed)"
    if summary.change_from_previous is None:
        return "—"
    change = summary.change_from_previous
    if change == 0:
        return "0 (no change)"
    if higher_is_better is None:
        return f"{change:+.6g} (direction unknown)"
    improved = (change > 0) == higher_is_better
    return f"{change:+.6g} ({'better' if improved else 'worse'})"


def _test_split_title(split: str, sample_count: int | None) -> str:
    count = str(sample_count) if sample_count is not None else "—"
    return f"{split} (n={count})"


def _display_split_version(split_version: int | None) -> str:
    if split_version is None:
        return "—"
    if split_version == 0:
        return "Initial split"
    return f"Revised split {split_version}"


def write_all_iterations_markdown(
    path: Path,
    run_meta: RunMeta,
    best_iteration: int | None,
    summaries: List[IterationSummary],
    test_metric_rows: List[Tuple[str, str, float]],
    test_sample_counts: Dict[str, int | None],
    metric_plot_path: Path | None = None,
) -> None:
    best_summary = next((row for row in summaries if row.is_best), None)
    best_score = (
        format_metric(best_summary.validation_value)
        if best_summary and best_summary.validation_value is not None
        else "not recorded"
    )
    higher_is_better = get_higher_is_better_map().get(run_meta.val_metric)
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
        f"| Iteration | Model / approach | Split | Validation {run_meta.val_metric or 'metric'} | Val. n | Change vs previous | Best |",
        "| ---: | :--- | :---: | ---: | ---: | ---: | :---: |",
    ]
    if summaries:
        lines.extend(
            "| {iteration} | {approach} | {split} | {value} | {sample_count} | {change} | {best} |".format(
                iteration=summary.iteration,
                approach=summary.approach.replace("|", "\\|"),
                split=_display_split_version(summary.split_version),
                value=format_metric(summary.validation_value),
                sample_count=(summary.validation_sample_count if summary.validation_sample_count is not None else "—"),
                change=_format_metric_change(summary, higher_is_better),
                best="Yes" if summary.is_best else "",
            )
            for summary in summaries
        )
    else:
        lines.append("| — | — | — | No iteration data available | — | — | — |")

    lines.extend(
        [
            "",
            "Change is current minus previous; ‘better’ or ‘worse’ follows the metric's optimization direction. "
            "It is omitted when split versions or scores are missing or splits differ.",
            "‘Initial split’ is the first train/validation partition. ‘Revised split N’ marks a later partition change.",
            "Other train and validation metrics are included in each detailed iteration report.",
            "",
        ]
    )
    if metric_plot_path is not None:
        lines.extend(
            [
                "## Train and validation metrics across iterations",
                "",
                "Lines stop when the data split changes. Test results are shown only for the selected best iteration.",
                "",
                f"![Train and validation metrics across iterations]({metric_plot_path.as_posix()})",
                "",
            ]
        )

    lines.extend(
        [
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
                "| Metric | "
                + " | ".join(
                    _test_split_title(split, test_sample_counts.get(split))
                    for split in test_splits
                )
                + " |",
                "| :--- | " + " | ".join("---:" for _ in test_splits) + " |",
            ]
        )
        lines.extend(
            "| " + metric_name + " | " + " | ".join(
                format_metric(test_values.get((split, metric_name))) for split in test_splits
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
    test_sample_counts: Dict[str, int | None],
    metric_plot_path: Path | None = None,
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
        format_metric(best_summary.validation_value)
        if best_summary and best_summary.validation_value is not None
        else "not recorded"
    )
    higher_is_better = get_higher_is_better_map().get(run_meta.val_metric)
    story = [
        Paragraph("All Iterations Summary", styles["H1"]),
        run_meta_flowables(run_meta),
        Spacer(1, 8),
        Paragraph(
            f"Best validation {escape_report_text(run_meta.val_metric or 'score')}: "
            f"{escape_report_text(best_score)} (iteration "
            f"{best_iteration if best_iteration is not None else 'not identified'}). "
            f"Archived iterations: {len(summaries)}.",
            styles["Body"],
        ),
        Paragraph(
            "Changes show whether the score improved or worsened according to the metric's direction. "
            "They are omitted when split versions differ or a score is missing. "
            "Detailed reports in markdown/ and pdf/ contain the other metrics and step details.",
            styles["Muted"],
        ),
        Paragraph(
            "‘Initial split’ is the first train/validation partition. ‘Revised split N’ marks a later partition change.",
            styles["Muted"],
        ),
        Spacer(1, 8),
    ]
    table_data = [
        ["Iteration", "Model / approach", "Split", "Validation", "Val. n", "Change", "Best"]
    ]
    if summaries:
        table_data.extend(
            [
                str(summary.iteration),
                Paragraph(escape_report_text(summary.approach), styles["Muted"]),
                _display_split_version(summary.split_version),
                format_metric(summary.validation_value),
                str(summary.validation_sample_count)
                if summary.validation_sample_count is not None
                else "—",
                escape_report_text(_format_metric_change(summary, higher_is_better)),
                "Yes" if summary.is_best else "",
            ]
            for summary in summaries
        )
    else:
        table_data.append(["—", "No iteration data available", "—", "—", "—", "—", "—"])

    table = LongTable(
        table_data,
        colWidths=[1.3 * cm, 5.1 * cm, 2.4 * cm, 2.1 * cm, 1.6 * cm, 2.9 * cm, 1.2 * cm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), pdf_color("navy")),
        ("TEXTCOLOR", (0, 0), (-1, 0), pdf_color("white")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, pdf_color("grid")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [pdf_color("white"), pdf_color("pale_gray")]),
        ("ALIGN", (0, 0), (2, -1), "LEFT"),
        ("ALIGN", (3, 1), (5, -1), "RIGHT"),
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
                    ("BACKGROUND", (0, row_index), (-1, row_index), pdf_color("pale_teal")),
                    ("FONTNAME", (0, row_index), (0, row_index), "Helvetica-Bold"),
                    ("TEXTCOLOR", (0, row_index), (0, row_index), pdf_color("navy")),
                ]
            )
    table.setStyle(TableStyle(table_style))
    story.append(table)

    if metric_plot_path is not None:
        image = RLImage(str(metric_plot_path))
        max_width = A4[0] - 4 * cm
        max_height = A4[1] - 4 * cm
        scale = min(max_width / image.drawWidth, max_height / image.drawHeight, 1.0)
        image.drawWidth *= scale
        image.drawHeight *= scale
        story.extend(
            [
                PageBreak(),
                Paragraph("Train and validation metrics across iterations", styles["H2"]),
                Paragraph(
                    "Lines stop when the data split changes. Test results are shown only for the selected best iteration.",
                    styles["Muted"],
                ),
                image,
            ]
        )

    if test_metric_rows:
        story.extend(
            [
                Spacer(1, 12),
                Paragraph("Held-out test results for the selected best iteration", styles["H2"]),
                Paragraph("Test results below are for the selected best iteration only.", styles["Muted"]),
            ]
        )
        test_splits, test_metrics, test_values = _pivot_test_metrics(test_metric_rows)
        test_table_data = [
            ["Metric"]
            + [
                _test_split_title(split, test_sample_counts.get(split))
                for split in test_splits
            ]
        ]
        test_table_data.extend(
            [metric_name]
            + [format_metric(test_values.get((split, metric_name))) for split in test_splits]
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
                    ("BACKGROUND", (0, 0), (-1, 0), pdf_color("navy")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), pdf_color("white")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.35, pdf_color("grid")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [pdf_color("white"), pdf_color("pale_gray")]),
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
    summaries = _all_iteration_summaries(config, iterations, best_iteration, run_meta.val_metric)
    test_metric_rows, test_sample_counts = _best_iteration_test_metrics(config, best_iteration)
    metric_points = _iteration_metric_points(config, iterations, best_iteration)
    metric_plot_path = _plot_iteration_metric_trends(
        config.pdf_reports_dir / "plots" / "all_iterations_metrics.png",
        metric_points,
        best_iteration,
    )
    metric_plot_link = (
        Path("pdf") / "plots" / metric_plot_path.name
        if metric_plot_path is not None
        else None
    )
    write_all_iterations_markdown(
        config.reports_dir / "all_iterations.md",
        run_meta,
        best_iteration,
        summaries,
        test_metric_rows,
        test_sample_counts,
        metric_plot_link,
    )
    write_all_iterations_pdf(
        config.reports_dir / "all_iterations.pdf",
        run_meta,
        best_iteration,
        summaries,
        test_metric_rows,
        test_sample_counts,
        metric_plot_path,
    )
