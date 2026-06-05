from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

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

    labeled = pd.read_csv(args.labeled_input)
    if "id" not in labeled.columns:
        sys.exit(f"'id' column required in {args.labeled_input} to align with predictions")
    if args.label_col not in labeled.columns:
        sys.exit(f"Label column '{args.label_col}' not found in {args.labeled_input}")

    if config.label_to_scalar:
        # Classification
        mapping = {str(key): value for key, value in config.label_to_scalar.items()}
        labeled[NUMERIC_LABEL_COL] = labeled[args.label_col].astype(str).map(mapping)
        if labeled[NUMERIC_LABEL_COL].isna().any():
            unknown = sorted(set(labeled.loc[labeled[NUMERIC_LABEL_COL].isna(), args.label_col].astype(str)))
            sys.exit(f"Labels not present in the training mapping {mapping}: {unknown}")
    else:
        # Regression
        labeled[NUMERIC_LABEL_COL] = labeled[args.label_col]

    tmp_labeled = args.predictions.parent / "._eval_labeled.csv"
    labeled.to_csv(tmp_labeled, index=False)
    try:
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
