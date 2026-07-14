from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from datasets.data_contract import (
    LABEL_COLUMN_NAME,
    NUMERIC_LABEL_COLUMN_NAME,
    validate_and_read_labels,
)
from datasets.label_processing import convert_classification_labels, convert_regression_labels
from runtime.evaluate_result import get_metrics
from runtime.read_write_utils import load_config_from_run_dir
from utils.config import Config
from utils.task_types import TaskTypes


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute evaluation metrics for a trained model's predictions.")
    ap.add_argument("--agent-dir", type=Path, required=True, help="Path to the run output dir (e.g. outputs/<agent_id>)")
    ap.add_argument("--predictions", type=Path, required=True, help="Inference output CSV (id + prediction + probability_* columns)")
    ap.add_argument("--labels", type=Path, required=True, help="Contract labels.csv (id,label or id,numeric_label), aligned to predictions by id")
    ap.add_argument("--output", type=Path, default=None, help="Where to write the metrics JSON (default: <predictions dir>/metrics.json)")
    args = ap.parse_args()

    config = load_config_from_run_dir(args.agent_dir / Config.RUN_DIRNAME)
    if config is None:
        sys.exit(f"Could not load run config under {args.agent_dir / Config.RUN_DIRNAME}")

    numeric_labels = _read_numeric_labels(args.labels, config)
    labeled_path = args.predictions.parent / f"{args.predictions.stem}.numeric_labels.csv"
    numeric_labels.to_csv(labeled_path, index=False)
    metrics = get_metrics(
        results_file=args.predictions,
        test_file=labeled_path,
        task_type=config.task_type,
        numeric_label_col=NUMERIC_LABEL_COLUMN_NAME,
    )

    print(json.dumps(metrics, indent=2))
    output_path = args.output or (args.predictions.parent / "metrics.json")
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
