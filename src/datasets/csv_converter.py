"""Convert tabular CSV files into the folder-based dataset format expected by Agentomics.

Usage as CLI:
    PYTHONPATH=src python src/datasets/csv_converter.py \
        --train-csv data/train.csv \
        --test-csv data/test.csv \
        --label-column target \
        --output-dir datasets/my_dataset

Usage as library:
    from datasets.csv_converter import convert_csv_dataset

    convert_csv_dataset(
        output_dir=Path("datasets/my_dataset"),
        label_column="target",
        splits={"train": train_df, "test": test_df},
    )

Each split produces:
    output_dir/split_name/input/data.csv   (features + id)
    output_dir/split_name/labels.csv       (id + label)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from datasets.data_contract import (
    ID_COLUMN_NAME,
    INPUT_DIR_NAME,
    LABEL_COLUMN_NAME,
    LABELS_FILE_NAME,
    METADATA_FILE_NAME,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

TABULAR_INPUT_FILE_NAME = "data.csv"

def convert_csv_dataset(
    output_dir: Path,
    label_column: str,
    splits: dict[str, pd.DataFrame],
    id_column: str | None = None,
    task_type: str | None = None,
) -> None:
    """Convert tabular DataFrames into the folder-based dataset format.

    Args:
        output_dir: Target dataset directory.
        label_column: Name of the column containing labels.
        splits: Mapping of split name ("train", "validation", "test") to DataFrame.
        id_column: Column to use as sample ID. If None, IDs are auto-generated.
        task_type: If provided, writes metadata.json with this task type.
    """
    if TRAIN_SPLIT not in splits:
        raise ValueError(f"A '{TRAIN_SPLIT}' split is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in splits.items():
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {split_name} split.")
        _write_split(output_dir, split_name, df, label_column, id_column)

    if task_type:
        metadata = {"task_type": task_type}
        (output_dir / METADATA_FILE_NAME).write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    print(f"Dataset created at {output_dir}")

def _write_split(
    output_dir: Path,
    split_name: str,
    df: pd.DataFrame,
    label_column: str,
    id_column: str | None,
) -> None:
    split_dir = output_dir / split_name
    input_dir = split_dir / INPUT_DIR_NAME
    input_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    if id_column and id_column not in df.columns:
        raise ValueError(f"--id-column '{id_column}' not found in CSV. Available columns: {list(df.columns)}")
    if id_column:
        df[ID_COLUMN_NAME] = df[id_column].astype(str)
        if id_column != ID_COLUMN_NAME:
            df = df.drop(columns=[id_column])
    elif ID_COLUMN_NAME not in df.columns:
        df.insert(0, ID_COLUMN_NAME, [f"{split_name}-{i}" for i in range(len(df))])
    else:
        df[ID_COLUMN_NAME] = df[ID_COLUMN_NAME].astype(str)

    labels = pd.DataFrame({
        ID_COLUMN_NAME: df[ID_COLUMN_NAME],
        LABEL_COLUMN_NAME: df[label_column].astype(str),
    })
    labels.to_csv(split_dir / LABELS_FILE_NAME, index=False)

    df.drop(columns=[label_column]).to_csv(input_dir / TABULAR_INPUT_FILE_NAME, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV files into the folder-based dataset format expected by Agentomics.",
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to training CSV file")
    parser.add_argument("--validation-csv", type=Path, default=None, help="Path to validation CSV file")
    parser.add_argument("--test-csv", type=Path, default=None, help="Path to test CSV file")
    parser.add_argument("--label-column", required=True, help="Name of the column containing labels")
    parser.add_argument("--id-column", default=None, help="Name of the ID column (auto-generated if not provided)")
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None, help="Task type to write in metadata.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory")
    args = parser.parse_args()

    splits = {TRAIN_SPLIT: pd.read_csv(args.train_csv)}
    if args.validation_csv:
        splits[VALIDATION_SPLIT] = pd.read_csv(args.validation_csv)
    if args.test_csv:
        splits[TEST_SPLIT] = pd.read_csv(args.test_csv)

    convert_csv_dataset(
        output_dir=args.output_dir,
        label_column=args.label_column,
        splits=splits,
        id_column=args.id_column,
        task_type=args.task_type,
    )

if __name__ == "__main__":
    main()
