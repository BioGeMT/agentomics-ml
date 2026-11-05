from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from agentomics.agents.steps.validation_evaluation import ValidationEvaluationStep
from agentomics.datasets.data_contract import (
    LABELS_FILE_NAME,
    NUMERIC_LABEL_COLUMN_NAME,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from agentomics.runtime.read_write_utils import load_best_iteration_snapshot_iteration
from agentomics.runtime.step_outputs import load_step_output
from agentomics.utils.config import Config
from agentomics.utils.task_types import TaskTypes


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
    split_name: str
    labeled_csv: Optional[Path]
    preds_csv: Optional[Path]
    metrics: Dict[str, float]


@dataclass
class IterationInputs:
    iteration: int
    report_md: Optional[Path]
    splits: List[SplitArtifacts]


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
    if config.task_type == TaskTypes.CLASSIFICATION and not config.label_to_scalar:
        raise SystemExit("Classification run config must include non-empty 'label_to_scalar'")
    return DatasetMeta(
        task_type=config.task_type,
        numeric_label_col=NUMERIC_LABEL_COLUMN_NAME,
        label_to_scalar=config.label_to_scalar,
    )


def _find_versioned_split_labels(
    config: Config,
    split_name: str,
    split_version: int | None,
) -> Optional[Path]:
    if split_version is None:
        return None
    candidate = config.splits_dir / f"split_{split_version}" / split_name / LABELS_FILE_NAME
    return candidate if candidate.exists() else None


def gather_iteration_inputs(config: Config, iteration: int) -> IterationInputs:
    iter_dir = config.iteration_dir(iteration)
    validation_metrics = load_validation_metrics(config, iter_dir)

    report_md = config.markdown_reports_dir / f"run_report_iter_{iteration}.md"
    if not report_md.exists():
        report_md = None

    data_split_output = load_step_output(config, "data_split", iter_dir)
    split_version = getattr(data_split_output, "split_version", None)
    train_csv = _find_versioned_split_labels(config, TRAIN_SPLIT, split_version)
    val_csv = _find_versioned_split_labels(config, VALIDATION_SPLIT, split_version)

    validation_evaluation_dir = iter_dir / ValidationEvaluationStep.step_id
    train_preds = validation_evaluation_dir / "eval_predictions_train.csv"
    val_preds = validation_evaluation_dir / "eval_predictions_validation.csv"

    splits = [
        SplitArtifacts("train", train_csv, train_preds, get_split_metrics(validation_metrics, "train")),
        SplitArtifacts("validation", val_csv, val_preds, get_split_metrics(validation_metrics, "validation")),
    ]

    best_iter = load_best_iteration_snapshot_iteration(config)
    if best_iter is not None and iteration == best_iter:
        for test_metrics_path in sorted(
            config.best_iteration_snapshot_dir.glob("eval_predictions_test*.metrics.json")
        ):
            predictions_stem = test_metrics_path.name.removesuffix(".metrics.json")
            split_name = predictions_stem.removeprefix("eval_predictions_")
            test_preds = config.best_iteration_snapshot_dir / f"{predictions_stem}.csv"
            test_labeled = config.best_iteration_snapshot_dir / f"{predictions_stem}.numeric_labels.csv"
            splits.append(
                SplitArtifacts(
                    split_name,
                    test_labeled,
                    test_preds,
                    load_saved_metrics(test_metrics_path),
                )
            )

    return IterationInputs(iteration=iteration, report_md=report_md, splits=splits)
