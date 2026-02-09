from __future__ import annotations

import json
import re
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
from utils.biomlbench_target_utils import get_target_col_from_description


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


def get_submission_dir_from_artifacts(artifact_root: Path) -> Path:
    submission_csv_paths = sorted(artifact_root.glob("**/submission/submission.csv"))
    assert len(submission_csv_paths) == 1, (
        "Expected exactly one submission.csv under run artifacts. "
        f"Found {len(submission_csv_paths)} at: {[str(p) for p in submission_csv_paths]}"
    )
    return submission_csv_paths[0].parent


def evaluate_classification_submission(
    dataset: str,
    artifact_root: Path,
    data_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, float], str]:
    """Evaluate binary classification submission (genomic benchmarks only)."""
    artifacts = EvaluationArtifacts(dataset=dataset, artifact_root=artifact_root)
    submission = pd.read_csv(artifacts.submission_path)
    dataset_dir = data_dir / "agentomics" / dataset / "raw"
    test_df = pd.read_csv(dataset_dir / "test.csv")
    label_col = test_df.columns[-1]
    labels = test_df[label_col].astype(str)
    numeric_labels, uniques = pd.factorize(labels, sort=True)
    mapping = {str(value): idx for idx, value in enumerate(uniques)}

    # Get probability for class 1 from numeric_label column
    prob_1 = submission["numeric_label"].astype(float)
    prob_0 = 1 - prob_1
    predictions = (prob_1 >= 0.5).astype(int)

    results_csv = output_dir / "metrics_results.csv"
    test_csv = output_dir / "metrics_test.csv"
    pd.DataFrame({
        "id": submission["id"],
        "prediction": predictions,
        "probability_0": prob_0,
        "probability_1": prob_1
    }).to_csv(results_csv, index=False)
    pd.DataFrame({
        "id": test_df["id"],
        "numeric_label": numeric_labels
    }).to_csv(test_csv, index=False)

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


def evaluate_proteingym_submission(
    task_id: str,
    artifact_root: Path,
    output_dir: Path,
    data_dir: Path,
) -> tuple[dict[str, float], str]:
    submission_dir = get_submission_dir_from_artifacts(artifact_root)
    predictions_path = submission_dir / "submission_extended.csv"
    assert predictions_path.is_file(), (
        f"Missing ProteinGym fold predictions file: {predictions_path}. "
        "Run ProteinGym post-processing first."
    )

    description_path = data_dir / task_id / "prepared" / "public" / "description.md"
    answers_path = data_dir / task_id / "prepared" / "private" / "answers.csv"
    assert description_path.is_file(), f"Missing BioMLBench description file: {description_path}"
    assert answers_path.is_file(), f"Missing BioMLBench answers file: {answers_path}"

    label_col = get_target_col_from_description(description_path)
    preds_df = pd.read_csv(predictions_path)
    fold_pred_cols = [col for col in preds_df.columns if col.startswith("fitness_score_fold_")]
    assert len(fold_pred_cols) > 0, (
        f"Expected fold prediction columns in {predictions_path}, found: {preds_df.columns.tolist()}"
    )

    per_fold_metrics = {}
    metric_names = set()
    for fold_pred_col in fold_pred_cols:
        fold_metrics = get_metrics(
            pred_col=fold_pred_col,
            results_file=predictions_path,
            test_file=answers_path,
            output_file=output_dir / f"metrics_{fold_pred_col}.txt",
            numeric_label_col=label_col,
            delete_preds=False,
            task_type="regression",
        )
        for metric_name, value in fold_metrics.items():
            metric_names.add(metric_name)
            per_fold_metrics[f"{metric_name}_{fold_pred_col}"] = float(value)

    averaged_metrics = {}
    for metric_name in metric_names:
        values = [per_fold_metrics[f"{metric_name}_{fold_pred_col}"] for fold_pred_col in fold_pred_cols]
        averaged_metrics[metric_name] = float(np.mean(values))

    output_payload = {}
    output_payload.update(per_fold_metrics)
    output_payload.update(averaged_metrics)
    _write_json(output_dir / "metrics.json", output_payload)
    return averaged_metrics, "regression"


def _get_biomlbench_task_type(description_path: Path) -> str:
    description_text = description_path.read_text().lower()
    if (
        "task description: binary classification" in description_text
        or "task description: classification" in description_text
    ):
        return "classification"
    if "task description: regression" in description_text:
        return "regression"

    metric_match = re.search(r"main metric:\s*\**\s*([a-zA-Z0-9_]+)", description_text)
    assert metric_match is not None, f"Could not infer task type from description: {description_path}"
    main_metric = metric_match.group(1).lower()

    classification_metrics = {"roc_auc", "auroc", "pr_auc", "auprc", "f1", "accuracy", "acc"}
    regression_metrics = {"pearsonr", "spearmanr", "rmse", "mse", "mae", "mean_absolute_error", "r2"}

    if main_metric in classification_metrics:
        return "classification"
    if main_metric in regression_metrics:
        return "regression"

    raise ValueError(f"Could not infer task type from description: {description_path}")


def evaluate_biomlbench_submission(
    task_id: str,
    artifact_root: Path,
    output_dir: Path,
    data_dir: Path,
) -> tuple[dict[str, float], str]:
    submission_dir = get_submission_dir_from_artifacts(artifact_root)
    submission_path = submission_dir / "submission.csv"
    assert submission_path.is_file(), f"Missing submission file: {submission_path}"

    description_path = data_dir / task_id / "prepared" / "public" / "description.md"
    answers_path = data_dir / task_id / "prepared" / "private" / "answers.csv"
    assert description_path.is_file(), f"Missing BioMLBench description file: {description_path}"
    assert answers_path.is_file(), f"Missing BioMLBench answers file: {answers_path}"

    task_type = _get_biomlbench_task_type(description_path)
    label_col = get_target_col_from_description(description_path)
    answers_df = pd.read_csv(answers_path)
    assert label_col in answers_df.columns, (
        f"Target column '{label_col}' from {description_path} "
        f"not found in answers columns: {answers_df.columns.tolist()}"
    )

    submission_df = pd.read_csv(submission_path)
    assert "id" in submission_df.columns, f"Submission must contain 'id': {submission_path}"
    if "numeric_label" in submission_df.columns:
        pred_col = "numeric_label"
    elif "prediction" in submission_df.columns:
        pred_col = "prediction"
    else:
        assert len(submission_df.columns) >= 2, (
            f"Submission must include a prediction column: {submission_path}"
        )
        pred_col = submission_df.columns[1]

    if task_type == "classification":
        prob_1 = submission_df[pred_col].astype(float).clip(0.0, 1.0)
        eval_results = pd.DataFrame(
            {
                "id": submission_df["id"],
                "prediction": (prob_1 >= 0.5).astype(int),
                "probability_0": 1.0 - prob_1,
                "probability_1": prob_1,
            }
        )
        eval_results_path = output_dir / "metrics_results.csv"
        eval_results.to_csv(eval_results_path, index=False)
        ordered = get_task_to_metrics_names()["classification"]
        raw_metrics = get_metrics(
            results_file=eval_results_path,
            test_file=answers_path,
            task_type="classification",
            numeric_label_col=label_col,
            pred_col="prediction",
            prob_col_prefix="probability_",
        )
    else:
        eval_results = pd.DataFrame(
            {
                "id": submission_df["id"],
                "prediction": submission_df[pred_col].astype(float),
            }
        )
        eval_results_path = output_dir / "metrics_results.csv"
        eval_results.to_csv(eval_results_path, index=False)
        ordered = get_task_to_metrics_names()["regression"]
        raw_metrics = get_metrics(
            results_file=eval_results_path,
            test_file=answers_path,
            task_type="regression",
            pred_col="prediction",
            numeric_label_col=label_col,
        )

    metrics = {name: float(value) for name, value in raw_metrics.items() if name in ordered}
    _write_json(output_dir / "metrics.json", metrics)
    return metrics, task_type


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

    # For zeroshot: just check files exist, skip actual rerun
    # (no tools = static code = deterministic. If we get predictions it means inference run sucessfully)
    if agent == "zeroshot":
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
