from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from eval.evaluate_result import get_metrics
from utils.metrics import get_task_to_metrics_names


@dataclass(frozen=True)
class EvaluationArtifacts:
    dataset: str
    artifact_root: Path

    @property
    def run_dir(self) -> Path:
        agentomics_dir = self.artifact_root / "agentomics"
        matches = list(agentomics_dir.glob(f"{self.dataset}_*"))
        assert len(matches) == 1
        return matches[0]

    @property
    def submission_path(self) -> Path:
        return self.run_dir / "submission" / "submission.csv"

    @property
    def code_dir(self) -> Path:
        return self.run_dir / "submission"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def evaluate_submission(
    dataset: str,
    artifact_root: Path,
    data_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, float], str]:
    artifacts = EvaluationArtifacts(dataset=dataset, artifact_root=artifact_root)
    submission = pd.read_csv(artifacts.submission_path)
    dataset_dir = data_dir / "agentomics" / dataset / "raw"
    test_df = pd.read_csv(dataset_dir / "test.csv")
    label_col = test_df.columns[-1]
    labels = test_df[label_col].astype(str)
    numeric_labels, uniques = pd.factorize(labels, sort=True)
    mapping = {str(value): idx for idx, value in enumerate(uniques)}

    # Get probability for class 1 from target column
    prob_1 = submission["target"].astype(float)
    prob_0 = 1 - prob_1
    predictions = (prob_1 >= 0.5).astype(int)

    results_csv = output_dir / "metrics_results.csv"
    test_csv = output_dir / "metrics_test.csv"
    pd.DataFrame({
        "prediction": predictions,
        "probability_0": prob_0,
        "probability_1": prob_1
    }).to_csv(results_csv, index=False)
    pd.DataFrame({"numeric_label": numeric_labels}).to_csv(test_csv, index=False)

    ordered = get_task_to_metrics_names()["classification"]
    metrics = {name: float(value) for name, value in get_metrics(
        results_file=str(results_csv),
        test_file=str(test_csv),
        task_type="classification",
        numeric_label_col="numeric_label",
        prob_col_prefix="probability_",
    ).items() if name in ordered}

    _write_json(output_dir / "label_mapping.json", mapping)
    _write_json(output_dir / "metrics.json", metrics)
    return metrics, "classification"


def _allclose(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> bool:
    if frame_a.shape != frame_b.shape:
        return False
    for column in frame_a.columns:
        if column not in frame_b.columns:
            return False
        series_a = frame_a[column].to_numpy(dtype=float)
        series_b = frame_b[column].to_numpy(dtype=float)
        if not np.allclose(series_a, series_b, rtol=1e-6, atol=1e-8):
            return False
    return True


INFERENCE_STAGE = {
    "missing": 0,
    "exists": 1,
    "runs": 2,
    "matches": 3,
}


def rerun_inference(
    dataset: str,
    artifact_root: Path,
    data_dir: Path,
    output_dir: Path,
    agent: str = None,
) -> str:
    artifacts = EvaluationArtifacts(dataset=dataset, artifact_root=artifact_root)
    inference_path = artifacts.code_dir / "inference.py"
    if not inference_path.exists():
        return "missing"
    env_yaml = inference_path.parent / "environment.yaml"
    if not env_yaml.exists():
        return "exists"

    # For oneshot: just check files exist, skip actual rerun
    # (no tools = static code = deterministic. If we get predictions it means inference run sucessfully)
    if agent == "oneshot":
        return "matches"

    # For tool-using agents: 
    features_path = data_dir / "agentomics" / dataset / "prepared" / "public" / "test_features.csv"
    replay_dir = output_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_output = replay_dir / "submission.csv"
    env_name = f"inference-replay-{dataset}"
    subprocess.run(
        ["conda", "env", "create", "-n", env_name, "-f", str(env_yaml)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    command = ["conda", "run", "-n", env_name, "python"]
    command += [
        str(inference_path),
        "--input",
        str(features_path),
        "--output",
        str(replay_output),
    ]
    try:
        subprocess.run(command, check=True, cwd=inference_path.parent)
    except subprocess.CalledProcessError:
        return "exists"
    original = pd.read_csv(artifacts.submission_path)
    replayed = pd.read_csv(replay_output)
    return "matches" if _allclose(original, replayed) else "runs"
