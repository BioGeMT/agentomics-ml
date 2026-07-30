"""Internal helpers for converting flat CSV datasets to Agentomics' raw folder contract."""
import json
import sys
from pathlib import Path

import pandas as pd

from agentomics.datasets.data_contract import (
    DATASET_DESCRIPTION_FILE_NAME,
    ID_COLUMN_NAME,
    INPUT_DIR_NAME,
    LABEL_COLUMN_NAME,
    LABELS_FILE_NAME,
    METADATA_FILE_NAME,
    SUPPLEMENTARY_DIR_NAME,
    TEST_SPLIT_PREFIX,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from agentomics.runtime.filesystem import create_absolute_symlink, remove_path

TABULAR_INPUT_FILE_NAME = "data.csv"
CSV_LABEL_COLUMN_METADATA_KEY = "label_column"
CSV_ID_COLUMN_METADATA_KEY = "id_column"
TRAIN_CSV_FILE_NAME = f"{TRAIN_SPLIT}.csv"
VALIDATION_CSV_FILE_NAME = f"{VALIDATION_SPLIT}.csv"
TEST_CSV_FILE_NAME = f"{TEST_SPLIT_PREFIX}.csv"
CSV_CONVERTED_SOURCE_DIR_NAME = "_csv_converted_source"
ALLOWED_PUBLIC_CSV_DATASET_ENTRIES = {
    TRAIN_CSV_FILE_NAME,
    VALIDATION_CSV_FILE_NAME,
    TEST_CSV_FILE_NAME,
    SUPPLEMENTARY_DIR_NAME,
    METADATA_FILE_NAME,
    DATASET_DESCRIPTION_FILE_NAME,
}
ALLOWED_TEST_CSV_DATASET_ENTRIES = (
    ALLOWED_PUBLIC_CSV_DATASET_ENTRIES | {TEST_CSV_FILE_NAME}
)


def is_public_csv_dataset(source_dir: Path) -> bool:
    source_dir = Path(source_dir)
    return not (source_dir / TRAIN_SPLIT).exists() and (source_dir / TRAIN_CSV_FILE_NAME).is_file()


def is_test_csv_dataset(source_dir: Path) -> bool:
    source_dir = Path(source_dir)
    return not (source_dir / TEST_SPLIT_PREFIX).exists() and (source_dir / TEST_CSV_FILE_NAME).is_file()

def convert_csv_dataset_to_standard_raw_dataset(
    source_dir: Path,
    destination_dir: Path,
    task_type: str | None = None,
    interactive: bool = False,
    label_column: str | None = None,
) -> Path:
    source_dir = Path(source_dir)
    _validate_raw_csv_dataset_entries(source_dir, ALLOWED_PUBLIC_CSV_DATASET_ENTRIES)

    source_metadata = _load_metadata(source_dir)
    train_df = pd.read_csv(source_dir / TRAIN_CSV_FILE_NAME)
    label_column, id_column = _resolve_csv_columns(source_metadata, train_df, interactive, label_column_override=label_column)

    splits = {TRAIN_SPLIT: train_df}
    validation_csv = source_dir / VALIDATION_CSV_FILE_NAME
    if validation_csv.is_file():
        splits[VALIDATION_SPLIT] = pd.read_csv(validation_csv)

    converted_source_dir = _converted_source_dir(destination_dir)
    remove_path(converted_source_dir)
    convert_csv_dataset(
        output_dir=converted_source_dir,
        label_column=label_column,
        splits=splits,
        id_column=id_column,
    )
    _write_preserved_metadata(
        converted_source_dir,
        source_metadata,
        task_type,
        label_column,
        id_column,
    )
    _link_optional_dataset_files(source_dir, converted_source_dir)
    return converted_source_dir


def convert_csv_test_dataset_to_standard_raw_dataset(
    source_dir: Path,
    destination_dir: Path,
    label_column: str | None = None,
    id_column: str | None = None,
) -> Path:
    source_dir = Path(source_dir)
    _validate_raw_csv_dataset_entries(source_dir, ALLOWED_TEST_CSV_DATASET_ENTRIES)

    source_metadata = _load_metadata(source_dir)
    label_column = label_column or source_metadata.get(
        CSV_LABEL_COLUMN_METADATA_KEY
    )
    if label_column is None:
        raise ValueError("CSV test dataset requires 'label_column' in metadata.json.")
    if id_column is None:
        id_column = source_metadata.get(CSV_ID_COLUMN_METADATA_KEY)

    converted_source_dir = _converted_source_dir(destination_dir)
    remove_path(converted_source_dir)
    _write_input_and_labels(
        split_dir=converted_source_dir / TEST_SPLIT_PREFIX,
        df=pd.read_csv(source_dir / TEST_CSV_FILE_NAME),
        label_column=str(label_column),
        id_prefix=TEST_SPLIT_PREFIX,
        id_column=str(id_column) if id_column is not None else None,
    )
    return converted_source_dir


def convert_inference_csv(
    csv_path: Path,
    output_split_dir: Path,
    label_column: str | None = None,
    id_column: str | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    _write_input_and_labels(
        split_dir=Path(output_split_dir),
        df=df,
        label_column=label_column,
        id_prefix="input",
        id_column=id_column,
    )


def convert_csv_dataset(
    output_dir: Path,
    label_column: str,
    splits: dict[str, pd.DataFrame],
    id_column: str | None = None,
    task_type: str | None = None,
) -> None:
    """Convert tabular DataFrames into the folder-based dataset format."""
    if TRAIN_SPLIT not in splits:
        raise ValueError(f"A '{TRAIN_SPLIT}' split is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in splits.items():
        _write_input_and_labels(
            split_dir=output_dir / split_name,
            df=df,
            label_column=label_column,
            id_prefix=split_name,
            id_column=id_column,
        )

    if task_type:
        metadata = {"task_type": task_type}
        (output_dir / METADATA_FILE_NAME).write_text(json.dumps(metadata, indent=4), encoding="utf-8")


def _converted_source_dir(destination_dir: Path) -> Path:
    destination_dir = Path(destination_dir)
    return destination_dir.parent / CSV_CONVERTED_SOURCE_DIR_NAME / destination_dir.name


def _load_metadata(source_dir: Path) -> dict:
    metadata_path = source_dir / METADATA_FILE_NAME
    return json.loads(metadata_path.read_text()) if metadata_path.exists() else {}


def _validate_raw_csv_dataset_entries(source_dir: Path, allowed_entries: set[str]) -> None:
    unsupported_entries = sorted(item.name for item in source_dir.iterdir() if item.name not in allowed_entries)
    if unsupported_entries:
        raise ValueError(
            f"CSV dataset {source_dir.name} has unsupported top-level entries: {unsupported_entries}. "
            f"Allowed: {sorted(allowed_entries)}."
        )


def _resolve_csv_columns(source_metadata: dict, train_df: pd.DataFrame, interactive: bool, label_column_override: str | None = None) -> tuple[str, str | None]:
    label_column = label_column_override or source_metadata.get(CSV_LABEL_COLUMN_METADATA_KEY)
    if label_column is None:
        if interactive and sys.stdin.isatty():
            from agentomics.datasets.datasets_interactive import select_csv_label_column

            label_column = select_csv_label_column(train_df.columns)
        else:
            raise ValueError("CSV dataset requires 'label_column' in metadata.json when running non-interactively.")
    id_column = source_metadata.get(CSV_ID_COLUMN_METADATA_KEY)
    return str(label_column), str(id_column) if id_column is not None else None


def _write_preserved_metadata(
    converted_source_dir: Path,
    source_metadata: dict,
    task_type: str | None,
    label_column: str,
    id_column: str | None,
) -> None:
    converted_metadata = {
        **source_metadata,
        CSV_LABEL_COLUMN_METADATA_KEY: label_column,
    }
    if id_column is not None:
        converted_metadata[CSV_ID_COLUMN_METADATA_KEY] = id_column
    if task_type is not None:
        converted_metadata["task_type"] = task_type
    if converted_metadata:
        (converted_source_dir / METADATA_FILE_NAME).write_text(
            json.dumps(converted_metadata, indent=4),
            encoding="utf-8",
        )


def _link_optional_dataset_files(source_dir: Path, converted_source_dir: Path) -> None:
    source_desc = source_dir / DATASET_DESCRIPTION_FILE_NAME
    if source_desc.exists():
        create_absolute_symlink(source_desc, converted_source_dir / DATASET_DESCRIPTION_FILE_NAME)
    source_supp = source_dir / SUPPLEMENTARY_DIR_NAME
    if source_supp.exists():
        create_absolute_symlink(source_supp, converted_source_dir / SUPPLEMENTARY_DIR_NAME)


def _ensure_id_column(df: pd.DataFrame, id_column: str | None, id_prefix: str) -> pd.DataFrame:
    df = df.copy()

    if id_column and id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' not found in CSV. Available columns: {list(df.columns)}")
    if id_column:
        df[ID_COLUMN_NAME] = df[id_column].astype(str)
        if id_column != ID_COLUMN_NAME:
            df = df.drop(columns=[id_column])
    elif ID_COLUMN_NAME not in df.columns:
        df.insert(0, ID_COLUMN_NAME, [f"{id_prefix}-{i}" for i in range(len(df))])
    else:
        df[ID_COLUMN_NAME] = df[ID_COLUMN_NAME].astype(str)
    return df


def _write_input_and_labels(
    split_dir: Path,
    df: pd.DataFrame,
    label_column: str | None,
    id_prefix: str,
    id_column: str | None,
) -> None:
    if label_column is not None and label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available columns: {list(df.columns)}")

    split_dir = Path(split_dir)
    input_dir = split_dir / INPUT_DIR_NAME
    input_dir.mkdir(parents=True, exist_ok=True)

    df = _ensure_id_column(df, id_column, id_prefix)

    if label_column is not None:
        pd.DataFrame({
            ID_COLUMN_NAME: df[ID_COLUMN_NAME],
            LABEL_COLUMN_NAME: df[label_column].astype(str),
        }).to_csv(split_dir / LABELS_FILE_NAME, index=False)
        df = df.drop(columns=[label_column])

    df.to_csv(input_dir / TABULAR_INPUT_FILE_NAME, index=False)
