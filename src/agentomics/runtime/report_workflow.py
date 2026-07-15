from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agentomics.runtime.filesystem import restore_host_ownership
from agentomics.runtime.iteration_reports import write_exported_run_reports
from agentomics.utils.config import Config


def _write_outputs_readme(workspace_directory: Path, agent_id: str) -> None:
    metadata_path = (
        workspace_directory
        / Config.BEST_ITERATION_SNAPSHOT_DIRNAME
        / Config.RUNTIME_INFO_DIRNAME
        / Config.ITERATION_METADATA_FILENAME
    )
    best_iteration = "No best iteration found"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        best_iteration = str(metadata.get("iteration", best_iteration))

    content = f"""# Agentomics Run

**Agent ID:** `{agent_id}`

**Best iteration:** `{best_iteration}`

- Best model and code: `best_iteration_snapshot/`
- Iteration archives: `run/iteration_N/`
- Reports: `reports/`
- Logs and diagnostics: `extras/`

## Inference

```bash
agentomics-inference --agent-dir {workspace_directory} --input <input> --output <predictions.csv>
```
"""
    (workspace_directory / "README.md").write_text(content, encoding="utf-8")


def _generate_reports(workspace_directory: Path) -> None:
    try:
        write_exported_run_reports(workspace_directory)
    except Exception as error:
        print(f"Warning: Failed to generate iteration reports: {error}", file=sys.stderr)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "agentomics.runtime.generate_final_reports",
                "--agent-dir",
                str(workspace_directory),
            ],
            check=True,
        )
    except Exception as error:
        print(f"Warning: Failed to generate PDF reports: {error}", file=sys.stderr)


def run_report_workflow(workspace_directory: Path, agent_id: str) -> None:
    try:
        _generate_reports(workspace_directory)
        _write_outputs_readme(workspace_directory, agent_id)
    finally:
        restore_host_ownership(workspace_directory)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final Agentomics reports.")
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--agent-id", required=True)
    arguments = parser.parse_args()
    run_report_workflow(arguments.workspace_dir, arguments.agent_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
