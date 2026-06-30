import argparse
from pathlib import Path

from datasets.dataset_preparation import prepare_dataset
from runtime.read_write_utils import load_config_from_run_dir
from utils.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare a re-training dataset dir into the contract format, reusing the run's trained task_type and label mapping."
    )
    ap.add_argument("--dataset-dir", type=Path, required=True, help="Dataset folder with train/validation splits (folders, or train.csv/validation.csv + metadata.json)")
    ap.add_argument("--output-dir", type=Path, required=True, help="Destination directory for the prepared dataset")
    ap.add_argument("--agent-dir", type=Path, required=True, help="Run output dir, used to load the trained config (task_type, label_to_scalar)")
    ap.add_argument("--label-col", default=None, help="Label column name for CSV-form splits (overrides metadata.json). Not needed for folder-form splits or when metadata.json declares label_column")
    args = ap.parse_args()

    config = load_config_from_run_dir(args.agent_dir / Config.RUN_DIRNAME)
    if config is None:
        raise SystemExit(f"Could not load run config under {args.agent_dir / Config.RUN_DIRNAME}")

    prepare_dataset(
        source_dir=args.dataset_dir,
        destination_dir=args.output_dir,
        task_type=config.task_type,
        label_to_scalar=config.label_to_scalar,
        label_column=args.label_col,
    )


if __name__ == "__main__":
    main()
