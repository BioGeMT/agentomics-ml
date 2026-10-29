from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

from agentomics.datasets.data_contract import VALIDATION_SPLIT
from agentomics.runtime.report_data import RunMeta, gather_iteration_inputs
from agentomics.runtime.report_styles import (
    build_styles,
    escape_report_text,
    format_metric,
    pdf_color,
    run_meta_flowables,
)
from agentomics.runtime.step_outputs import load_step_output
from agentomics.utils.config import Config


@dataclass(frozen=True)
class IterationSummary:
    iteration: int
    approach: str
    split_version: int | None
    validation_value: float | None
    change_from_previous: float | None
    split_changed: bool
    is_best: bool


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
    values = {(split, metric): value for split, metric, value in test_metric_rows}
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
        format_metric(best_summary.validation_value)
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
                split=f"split_{summary.split_version}" if summary.split_version is not None else "—",
                value=format_metric(summary.validation_value),
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
                Paragraph(escape_report_text(summary.approach), styles["Muted"]),
                f"split_{summary.split_version}" if summary.split_version is not None else "—",
                format_metric(summary.validation_value),
                escape_report_text(_format_metric_change(summary)),
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
        ("BACKGROUND", (0, 0), (-1, 0), pdf_color("navy")),
        ("TEXTCOLOR", (0, 0), (-1, 0), pdf_color("white")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, pdf_color("grid")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [pdf_color("white"), pdf_color("pale_gray")]),
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
                    ("BACKGROUND", (0, row_index), (-1, row_index), pdf_color("pale_teal")),
                    ("FONTNAME", (0, row_index), (0, row_index), "Helvetica-Bold"),
                    ("TEXTCOLOR", (0, row_index), (0, row_index), pdf_color("navy")),
                ]
            )
    table.setStyle(TableStyle(table_style))
    story.append(table)

    if test_metric_rows:
        story.extend(
            [
                Spacer(1, 12),
                Paragraph("Held-out test results for the selected best iteration", styles["H2"]),
                Paragraph("Test results below are for the selected best iteration only.", styles["Muted"]),
            ]
        )
        test_splits, test_metrics, test_values = _pivot_test_metrics(test_metric_rows)
        test_table_data = [["Metric"] + [split.title() for split in test_splits]]
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
