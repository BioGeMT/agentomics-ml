from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from eval.evaluate_result import get_metrics
from runtime.filesystem import remove_path
from utils.exceptions import AgentScriptFailed


def run_inference_script(
    env_path: Path,
    script_path: Path,
    input_path: Path,
    output_path: Path,
    artifacts_dir: Path,
    check: bool = False,
):
    return subprocess.run(
        [
            "conda", "run", "-p", str(env_path),
            "python", str(script_path),
            "--input", str(input_path),
            "--output", str(output_path),
            "--artifacts-dir", str(artifacts_dir),
        ],
        capture_output=True,
        cwd=script_path.parent,
        check=check,
    )
#TODO should the temp files be in a specific folder?
#TODO is this writing in the split folder? That would fail due to permissions
# Only for runs that need labelless files generated on the fly.
def run_inference_on_labeled_data(evaluation_stage: str, labeled_input_path: Path, output_path: Path, conda_env_path: Path, inference_script_path: Path, training_artifacts_dir: Path, label_col: str):
    """Run inference script after creating a temporary labelless input."""
    inference_input_path = output_path.parent / f"{evaluation_stage}.no_label.csv"
    try:
        remove_path(inference_input_path)
        _create_labelless_file(
            labeled_input_path=labeled_input_path,
            target_path=inference_input_path,
            label_col=label_col,
        )
        remove_path(output_path)
        return run_inference_script(
            env_path=conda_env_path,
            script_path=inference_script_path,
            input_path=inference_input_path,
            output_path=output_path,
            artifacts_dir=training_artifacts_dir,
        )
    finally:
        remove_path(inference_input_path)

def compute_metrics(results_file: Path, labeled_input_path: Path, numeric_label_col: str, task_type: str, evaluation_stage: str) -> dict | None:
    """Compute metrics from inference output. Returns None if labeled_input does not exist."""
    if not labeled_input_path.exists():
        return None
    try:
        return get_metrics(
            results_file=results_file,
            test_file=labeled_input_path,
            numeric_label_col=numeric_label_col,
            task_type=task_type,
        )
    except Exception as error:
        #TODO needs a prediction-specific exception?
        raise AgentScriptFailed(f"Metrics computation failed for {evaluation_stage}.") from error

def _create_labelless_file(labeled_input_path: Path, target_path: Path, label_col: str):
    if not labeled_input_path.exists():
        raise FileNotFoundError(f"Labeled input file not found at {labeled_input_path} during the inference stage")
    labeled_df = pd.read_csv(labeled_input_path)
    if label_col not in labeled_df.columns:
        raise ValueError(f"No {label_col} column found in {labeled_input_path}.")
    labeled_df.drop(columns=[label_col]).to_csv(target_path, index=False)
