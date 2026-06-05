from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from runtime.read_write_utils import get_archived_iterations, load_best_iteration_snapshot_iteration, load_config_from_run_dir_and_reroot
from runtime.step_outputs import load_step_output
from utils.config import Config


def write_iteration_report(
    config: Config,
    iteration: int,
    iteration_dir: Path | None = None,
    report_path: Path | None = None,
    metrics: dict[str, float] | None = None,
) -> Path:
    source_dir = iteration_dir or config.current_iteration_dir
    target_path = report_path or config.markdown_reports_dir / f"run_report_iter_{iteration}.md"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        render_iteration_report(
            config=config,
            iteration=iteration,
            iteration_dir=source_dir,
            metrics=metrics,
        ),
        encoding="utf-8",
    )
    return target_path


def write_exported_run_reports(agent_dir: Path) -> list[Path]:
    config = load_config_from_run_dir_and_reroot(agent_dir / Config.RUN_DIRNAME)
    config.markdown_reports_dir.mkdir(parents=True, exist_ok=True)

    report_paths: list[Path] = []
    for iteration in get_archived_iterations(config):
        iteration_dir = config.iteration_dir(iteration)
        report_paths.append(
            write_iteration_report(
                config=config,
                iteration=iteration,
                iteration_dir=iteration_dir,
                report_path=config.markdown_reports_dir / f"run_report_iter_{iteration}.md",
                metrics=_load_validation_metrics(config, iteration_dir),
            )
        )
    return report_paths


def render_iteration_report(
    config: Config,
    iteration: int,
    iteration_dir: Path,
    metrics: dict[str, float] | None = None,
) -> str:
    lines = [
        f"# Run Report - Iteration {iteration}",
        "",
        f"**Agent ID:** `{config.agent_id}`  ",
        f"**Dataset:** `{config.dataset}`  ",
        f"**Model:** `{config.model_name}`  ",
        f"**Task:** `{config.task_type}`  ",
        f"**Validation metric:** `{config.val_metric}`  ",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    step_sections_added = 0
    for step_id in [str(step_id) for step_id in config.step_sequence]:
        output = load_step_output(config, step_id, iteration_dir)
        if output is None:
            continue
        step_sections_added += 1
        lines.extend(
            [
                f"## {_humanize_step_title(step_id)}",
                "",
                _render_section_body(output),
                "",
            ]
        )

    if step_sections_added == 0:
        lines.extend(
            [
                "## Iteration",
                "",
                "_No step outputs available._",
                "",
            ]
        )

    if metrics:
        lines.extend(["## Metrics", "", _render_metrics(metrics), ""])

    return "\n".join(lines).rstrip() + "\n"


def _render_section_body(output: Any) -> str:
    payload = output.model_dump() if hasattr(output, "model_dump") else output
    if not isinstance(payload, dict):
        return _render_value(payload)

    lines: list[str] = []
    files_created = payload.get("files_created")
    for key, value in payload.items():
        if key in {"files_created", "status", "unresolved_issues"}:
            continue
        lines.extend(
            [
                f"**{key.replace('_', ' ').title()}:**",
                "",
                _render_value(value),
                "",
            ]
        )

    if files_created:
        lines.extend(["**Files created:**", ""])
        lines.extend(f"- `{Path(str(file_path)).name}`" for file_path in files_created)
        lines.append("")

    rendered = "\n".join(lines).strip()
    return rendered or "_No content._"


def _render_value(value: Any) -> str:
    if value is None:
        return "_None_"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "_Empty_"
        if "\n" in stripped or len(stripped) > 120:
            return _quote_block(stripped)
        return stripped
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n".join(f"- `{item}`" for item in value) or "_Empty list_"
    return _quote_block(json.dumps(value, indent=2, sort_keys=True, default=str))


def _quote_block(text: str) -> str:
    return "\n".join(
        f"> {line}" if line.strip() else ">"
        for line in text.splitlines()
    )


def _humanize_step_title(step_name: str) -> str:
    normalized = step_name.replace("_", " ").strip()
    special_cases = {
        "dataexploration": "Data Exploration",
        "predictionexploration": "Prediction Exploration",
        "modelarchitecture": "Model Architecture",
        "modeltraining": "Model Training",
        "modelinference": "Model Inference",
        "datarepresentation": "Data Representation",
        "datasplit": "Data Split",
    }
    return special_cases.get(normalized.replace(" ", "").lower(), normalized.title())


def _render_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(
        f"- **{metric_name}**: {metric_value}"
        for metric_name, metric_value in metrics.items()
    ) or "_No metrics._"


def _load_validation_metrics(config: Config, iteration_dir: Path) -> dict[str, float]:
    validation_output = load_step_output(config, "validation_evaluation", iteration_dir)
    if validation_output is None:
        return {}
    metrics = getattr(validation_output, "metrics", None)
    if not isinstance(metrics, dict):
        return {}
    return {str(metric_name): float(metric_value) for metric_name, metric_value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown iteration reports from archived step outputs.")
    parser.add_argument("--agent-dir", type=Path, required=True, help="Path to outputs/<agent_id>")
    args = parser.parse_args()
    write_exported_run_reports(args.agent_dir.resolve())


if __name__ == "__main__":
    main()
