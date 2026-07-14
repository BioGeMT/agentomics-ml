from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agentomics.datasets.data_contract import (
    LABEL_COLUMN_NAME,
    NUMERIC_LABEL_COLUMN_NAME,
    validate_and_read_labels,
)
from agentomics.datasets.label_processing import convert_classification_labels, convert_regression_labels
from agentomics.runtime.evaluate_result import get_metrics
from agentomics.runtime.read_write_utils import load_config_from_run_dir
from agentomics.utils.config import Config
from agentomics.utils.task_types import TaskTypes


def _read_numeric_labels(labels_path: Path, config: Config) -> pd.DataFrame:
    columns = list(pd.read_csv(labels_path, nrows=0).columns)
    if NUMERIC_LABEL_COLUMN_NAME in columns:
        return validate_and_read_labels(
            labels_path, NUMERIC_LABEL_COLUMN_NAME, require_numeric_values=True
        )
    labels = validate_and_read_labels(labels_path, LABEL_COLUMN_NAME)
    if config.task_type == TaskTypes.CLASSIFICATION:
        return convert_classification_labels(labels, config.label_to_scalar, "eval")
    return convert_regression_labels(labels, "eval")


def evaluate_predictions(
    agent_dir: Path,
    predictions_path: Path,
    labels_path: Path,
    output_path: Path,
) -> dict:
    config = load_config_from_run_dir(agent_dir / Config.RUN_DIRNAME)
    if config is None:
        raise FileNotFoundError(f"Could not load run config under {agent_dir / Config.RUN_DIRNAME}")

    numeric_labels = _read_numeric_labels(labels_path, config)
    numeric_labels_path = predictions_path.parent / f"{predictions_path.stem}.numeric_labels.csv"
    numeric_labels.to_csv(numeric_labels_path, index=False)
    metrics = get_metrics(
        results_file=predictions_path,
        test_file=numeric_labels_path,
        task_type=config.task_type,
        numeric_label_col=NUMERIC_LABEL_COLUMN_NAME,
    )

    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics written to {output_path}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute evaluation metrics for a trained model's predictions.")
    ap.add_argument("--agent-dir", type=Path, required=True, help="Path to the run output dir (e.g. outputs/<agent_id>)")
    ap.add_argument("--predictions", type=Path, required=True, help="Inference output CSV (id + prediction + probability_* columns)")
    ap.add_argument("--labels", type=Path, required=True, help="Contract labels.csv (id,label or id,numeric_label), aligned to predictions by id")
    ap.add_argument("--output", type=Path, default=None, help="Where to write the metrics JSON (default: <predictions dir>/metrics.json)")
    args = ap.parse_args()

    output_path = args.output or (args.predictions.parent / "metrics.json")
    evaluate_predictions(args.agent_dir, args.predictions, args.labels, output_path)


if __name__ == "__main__":
    main()
