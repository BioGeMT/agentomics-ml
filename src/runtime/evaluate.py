from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from datasets.normalize_dataset import normalize_input_dataset
from runtime.evaluate_result import get_metrics
from runtime.read_write_utils import load_config_from_run_dir
from utils.config import Config

NUMERIC_LABEL_COL = "numeric_label" #TODO get this from config?


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute evaluation metrics for a trained model's predictions.")
    ap.add_argument("--agent-dir", type=Path, required=True, help="Path to the run output dir (e.g. outputs/<agent_id>)")
    ap.add_argument("--predictions", type=Path, required=True, help="Inference output CSV (id + prediction + probability_* columns)")
    ap.add_argument("--labeled-input", type=Path, required=True, help="CSV with the true label column and matching ids")
    ap.add_argument("--label-col", required=True, help="Name of the true label column in --labeled-input")
    ap.add_argument("--output", type=Path, default=None, help="Where to write the metrics JSON (default: <predictions dir>/metrics.json)")
    args = ap.parse_args()

    config = load_config_from_run_dir(args.agent_dir / Config.RUN_DIRNAME)
    if config is None:
        sys.exit(f"Could not load run config under {args.agent_dir / Config.RUN_DIRNAME}")

    with open(args.labeled_input, newline="") as f:
        header = next(csv.reader(f), [])
    if "id" not in header:
        sys.exit(f"'id' column required in {args.labeled_input} to align with predictions")
    if args.label_col not in header:
        sys.exit(f"Label column '{args.label_col}' not found in {args.labeled_input}")

    tmp_labeled = args.predictions.parent / "._eval_labeled.csv"
    try:
        normalize_input_dataset(
            args.labeled_input,
            tmp_labeled,
            label_col=args.label_col,
            label_to_scalar=config.label_to_scalar,
        )
        metrics = get_metrics(
            results_file=args.predictions,
            test_file=tmp_labeled,
            task_type=config.task_type,
            numeric_label_col=NUMERIC_LABEL_COL,
        )
    finally:
        tmp_labeled.unlink(missing_ok=True)

    print(json.dumps(metrics, indent=2))
    output_path = args.output or (args.predictions.parent / "metrics.json")
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
