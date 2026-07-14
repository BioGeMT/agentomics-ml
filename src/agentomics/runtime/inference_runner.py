from __future__ import annotations

from pathlib import Path

from agentomics.runtime.evaluate_result import get_metrics
from agentomics.runtime.conda_utils import run_python_in_environment
from agentomics.runtime.filesystem import remove_path
from agentomics.utils.exceptions import AgentScriptFailed
from agentomics.datasets.data_contract import INPUT_DIR_NAME


def run_inference_script(
    env_path: Path,
    script_path: Path,
    input_path: Path,
    output_path: Path,
    artifacts_dir: Path,
    check: bool = False,
    capture_output: bool = True,
    environment: dict[str, str] | None = None,
):
    return run_python_in_environment(
        env_path,
        script_path,
        [
            "--input", str(input_path),
            "--output", str(output_path),
            "--artifacts-dir", str(artifacts_dir),
        ],
        capture_output=capture_output,
        check=check,
        environment=environment,
    )
def run_inference_on_split(
    split_path: Path,
    output_path: Path,
    conda_env_path: Path,
    inference_script_path: Path,
    training_artifacts_dir: Path,
):
    input_path = split_path / INPUT_DIR_NAME
    if not input_path.is_dir():
        raise FileNotFoundError(f"Split input folder not found at {input_path}")
    remove_path(output_path)
    return run_inference_script(
        env_path=conda_env_path,
        script_path=inference_script_path,
        input_path=input_path,
        output_path=output_path,
        artifacts_dir=training_artifacts_dir,
    )

def compute_metrics(results_file: Path, labels_path: Path, numeric_label_col: str, task_type: str, evaluation_stage: str) -> dict | None:
    """Compute metrics from inference output. Returns None if labels_path does not exist."""
    if not labels_path.exists():
        return None
    try:
        return get_metrics(
            results_file=results_file,
            test_file=labels_path,
            numeric_label_col=numeric_label_col,
            task_type=task_type,
        )
    except Exception as error:
        #TODO needs a prediction-specific exception?
        raise AgentScriptFailed(f"Metrics computation failed for {evaluation_stage}.") from error
