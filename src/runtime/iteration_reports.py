from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from runtime.read_write_utils import get_archived_iterations, load_config_from_run_dir
from runtime.read_write_utils import load_iteration_state
from runtime.step_outputs import load_step_output
from utils.config import Config


def write_iteration_report(
    config: Config,
    iteration: int,
    iteration_dir: Path | None = None,
    report_path: Path | None = None,
) -> Path:
    source_dir = iteration_dir or config.current_iteration_dir
    target_path = report_path or config.agent_reports_dir / f"run_report_iter_{iteration}.md"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        render_iteration_report(
            config=config,
            iteration=iteration,
            iteration_dir=source_dir,
        ),
        encoding="utf-8",
    )
    return target_path


def write_exported_run_reports(agent_dir: Path) -> list[Path]:
    config = load_config_from_run_dir(agent_dir / "run_files")
    reports_dir = agent_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_paths: list[Path] = []
    for iteration in get_archived_iterations(
        config,
        search_root_dir=agent_dir / "run_files",
    ):
        report_paths.append(
            write_iteration_report(
                config=config,
                iteration=iteration,
                iteration_dir=agent_dir / "run_files" / f"{Config.ITERATION_DIR_PREFIX}{iteration}",
                report_path=reports_dir / f"run_report_iter_{iteration}.md",
            )
        )
    return report_paths


def render_iteration_report(
    config: Config,
    iteration: int,
    iteration_dir: Path,
) -> str:
    iteration_state = load_iteration_state(iteration_dir)
    lines = [
        f"# Agentomics Run Report - Iteration {iteration}",
        "",
        f"Agent ID: {config.agent_id}",
        f"Dataset: {config.dataset}",
        f"Model: {config.model_name}",
        f"Task: {config.task_type}",
        f"Validation metric: {config.val_metric}",
    ]
    status = iteration_state.get("status")
    if status is not None:
        lines.append(f"Status: {status}")
    lines.extend(
        [
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]
    )

    step_sections_added = 0
    for step_id in [str(step_id) for step_id in config.step_sequence]:
        output = load_step_output(config, step_id, iteration_dir)
        if output is None:
            continue
        step_sections_added += 1
        lines.extend(
            [
                f"## {step_id.replace('_', ' ').title()}",
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

    return "\n".join(lines).rstrip() + "\n"


def _render_section_body(output: Any) -> str:
    payload = output.model_dump() if hasattr(output, "model_dump") else output
    if not isinstance(payload, dict):
        return _render_value(payload)

    lines: list[str] = []
    files_created = payload.get("files_created")
    for key, value in payload.items():
        if key == "files_created":
            continue
        lines.extend(
            [
                f"{key.replace('_', ' ').title()}:",
                "",
                _render_value(value),
                "",
            ]
        )

    if files_created:
        lines.extend(["Files created:", ""])
        lines.extend(f"- `{file_path}`" for file_path in files_created)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown iteration reports from archived step outputs.")
    parser.add_argument("--agent-dir", type=Path, required=True, help="Path to outputs/<agent_id>")
    args = parser.parse_args()
    write_exported_run_reports(args.agent_dir.resolve())


if __name__ == "__main__":
    main()
