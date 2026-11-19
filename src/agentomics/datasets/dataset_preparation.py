import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rich.console import Console

from agentomics.datasets.data_contract import (
    DATASET_DESCRIPTION_FILE_NAME,
    INPUT_DIR_NAME,
    LABELS_FILE_NAME,
    METADATA_FILE_NAME,
    NON_TEST_SPLIT_NAMES,
    NUMERIC_LABEL_COLUMN_NAME,
    SUPPLEMENTARY_DIR_NAME,
    TEST_SPLIT_PREFIX,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    record_input_dir_structure,
    validate_public_dataset_entries,
    validate_split_entries,
    validate_splits,
)
from agentomics.datasets.csv_converter import (
    convert_csv_dataset_to_standard_raw_dataset,
    convert_csv_test_dataset_to_standard_raw_dataset,
    is_public_csv_dataset,
    is_test_csv_dataset,
)
from agentomics.datasets.datasets_interactive import (
    select_positive_class,
    select_task_type,
)
from agentomics.datasets.label_processing import (
    load_split_label_dfs,
)
from agentomics.runtime.filesystem import create_absolute_symlink, find_symlinks_in_dir
from agentomics.utils.ml_tasks import MLTask, get_task

console = Console()

def _reject_symlinks_in_dir(directory: Path) -> None:
    symlinks = find_symlinks_in_dir(directory, include_root=True)
    if symlinks:
        raise ValueError(
            f"Symlinks in {directory} are not allowed. "
            f"{len(symlinks)} symlink(s) found (e.g. {symlinks[0]})."
        )

def _resolve_task(
    source_metadata: dict, cli_task_type: str | None, train_labels: pd.DataFrame, interactive: bool,
) -> MLTask:
    resolved = cli_task_type or source_metadata.get("task_type")
    if resolved is None:
        resolved = select_task_type(train_labels, interactive=interactive)
    return get_task(resolved)

def _build_output_metadata(
    source_metadata: dict,
    resolved_task_type: str,
    split_labels: dict[str, pd.DataFrame],
    input_structure: list[str],
) -> dict:
    metadata = {**source_metadata}
    metadata["task_type"] = resolved_task_type
    metadata["splits"] = {
        "train_rows": len(split_labels[TRAIN_SPLIT]),
        "validation_rows": len(split_labels[VALIDATION_SPLIT]) if VALIDATION_SPLIT in split_labels else None,
    }
    metadata["numeric_label_col"] = NUMERIC_LABEL_COLUMN_NAME
    metadata["input_structure"] = input_structure
    metadata["prepared"] = True
    return metadata

def prepare_dataset(
    source_dir: Path,
    destination_dir: Path,
    task_type=None,
    positive_class=None,
    negative_class=None,
    interactive=False,
    label_to_scalar=None,
    label_column=None,
) -> dict:
    """Prepare a dataset from source_dir into destination_dir.

    Reads from source_dir (never modified), writes the prepared result to
    destination_dir (input dirs are symlinked, labels are converted, metadata
    is written). Returns the output metadata dict.
    """
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError(
            f"source_dir and destination_dir must be different. "
            f"Both resolve to: {source_dir.resolve()}"
        )

    _reject_symlinks_in_dir(source_dir)
    if is_public_csv_dataset(source_dir):
        source_dir = convert_csv_dataset_to_standard_raw_dataset(
            source_dir=source_dir,
            destination_dir=destination_dir,
            task_type=task_type,
            interactive=interactive,
            label_column=label_column,
        )
    metadata_path = source_dir / METADATA_FILE_NAME
    source_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    validate_public_dataset_entries(source_dir)

    source_splits = {
        split_name: source_dir / split_name
        for split_name in NON_TEST_SPLIT_NAMES
        if (source_dir / split_name).is_dir()
    }
    if TRAIN_SPLIT not in source_splits:
        raise FileNotFoundError(f"Required train/ split is missing: {source_dir / TRAIN_SPLIT}")

    for split_name, split_path in source_splits.items():
        validate_split_entries(split_path, split_name)

    input_structure = record_input_dir_structure(source_splits[TRAIN_SPLIT] / INPUT_DIR_NAME)
    split_labels = load_split_label_dfs(source_splits)

    task = _resolve_task(source_metadata, task_type, split_labels[TRAIN_SPLIT], interactive)
    output_metadata = _build_output_metadata(
        source_metadata, task.name, split_labels, input_structure,
    )

    numeric_split_labels, label_metadata = task.validate_and_numericalize_labels(
        split_labels, source_metadata, fit=True,
        options={
            "label_to_scalar": label_to_scalar,
            "positive_class": positive_class,
            "negative_class": negative_class,
            "positive_class_selector": select_positive_class if interactive and sys.stdin.isatty() else None,
        },
    )
    output_metadata.update(label_metadata)

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_splits = {}
    for split_name, source_split in source_splits.items():
        dest_split = destination_dir / split_name
        dest_split.mkdir(exist_ok=True)
        create_absolute_symlink(source_split / INPUT_DIR_NAME, dest_split / INPUT_DIR_NAME)
        numeric_split_labels[split_name].to_csv(dest_split / LABELS_FILE_NAME, index=False)
        destination_splits[split_name] = dest_split

    validate_splits(destination_splits, input_structure)

    source_desc = source_dir / DATASET_DESCRIPTION_FILE_NAME
    if source_desc.exists():
        create_absolute_symlink(source_desc, destination_dir / DATASET_DESCRIPTION_FILE_NAME)
    else:
        console.print("INFO: No dataset description provided.")
        (destination_dir / DATASET_DESCRIPTION_FILE_NAME).write_text(
            "No dataset description available.",
            encoding="utf-8",
        )

    source_supp = source_dir / SUPPLEMENTARY_DIR_NAME
    if source_supp.exists():
        create_absolute_symlink(source_supp, destination_dir / SUPPLEMENTARY_DIR_NAME)

    (destination_dir / METADATA_FILE_NAME).write_text(json.dumps(output_metadata, indent=4), encoding="utf-8")

    return output_metadata


def check_dataset(source_dir: Path, interactive: bool = False) -> dict:
    """Validate a dataset against the run contract without executing a run.

    Mirrors the preparation a run performs (train/validation, plus the test split
    when present) into a temporary directory that is discarded, so the user's
    dataset is never modified. Returns the prepared train/validation metadata,
    augmented with ``test_rows`` (the validated test-split row count, or None when
    no test split is present). Raises ValueError / FileNotFoundError with a
    descriptive message when the dataset does not satisfy the contract.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {source_dir}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        metadata = prepare_dataset(
            source_dir=source_dir,
            destination_dir=temporary_root / "prepared",
            interactive=interactive,
        )
        test_rows = None
        has_test_split = (source_dir / TEST_SPLIT_PREFIX).is_dir() or is_test_csv_dataset(source_dir)
        if has_test_split:
            test_metadata = prepare_test_dataset(
                source_dir=source_dir,
                destination_dir=temporary_root / "prepared_test",
                task_type=metadata["task_type"],
                input_structure=metadata["input_structure"],
                label_to_scalar=metadata.get("label_to_scalar"),
                label_column=metadata.get("label_column"),
                id_column=metadata.get("id_column"),
            )
            test_rows = test_metadata["splits"]["test_rows"]

    summary = dict(metadata)
    summary["test_rows"] = test_rows
    return summary


def check_dataset_prepared(dataset_dir: Path) -> bool:
    metadata_path = Path(dataset_dir) / METADATA_FILE_NAME
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("prepared") is True


def prepare_test_dataset(
    source_dir: Path,
    destination_dir: Path,
    task_type: str,
    input_structure: list[str],
    label_to_scalar: dict[str, int] | None = None,
    label_column: str | None = None,
    id_column: str | None = None,
) -> dict:
    """Prepare test dataset from source_dir into destination_dir using run metadata.

    Reads from source_dir (never modified), writes the prepared result to
    destination_dir (input dir is symlinked, labels are converted, metadata
    is written). Returns the output metadata dict.
    """
    task = get_task(task_type)
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    _reject_symlinks_in_dir(source_dir)
    csv_test_dataset = is_test_csv_dataset(source_dir)
    if csv_test_dataset:
        source_dir = convert_csv_test_dataset_to_standard_raw_dataset(
            source_dir,
            destination_dir,
            label_column=label_column,
            id_column=id_column,
        )

    test_source = source_dir / TEST_SPLIT_PREFIX
    if not test_source.is_dir():
        raise FileNotFoundError(f"Required test/ split is missing: {test_source}")

    validate_split_entries(test_source, TEST_SPLIT_PREFIX)

    test_labels = load_split_label_dfs({TEST_SPLIT_PREFIX: test_source})

    numeric_test_labels, label_metadata = task.validate_and_numericalize_labels(
        test_labels, {"label_to_scalar": label_to_scalar},
    )
    numeric_labels = numeric_test_labels[TEST_SPLIT_PREFIX]

    dest_test = destination_dir / TEST_SPLIT_PREFIX
    dest_test.mkdir(parents=True, exist_ok=True)
    if csv_test_dataset:
        shutil.copytree(test_source / INPUT_DIR_NAME, dest_test / INPUT_DIR_NAME)
    else:
        create_absolute_symlink(
            test_source / INPUT_DIR_NAME,
            dest_test / INPUT_DIR_NAME,
        )
    numeric_labels.to_csv(dest_test / LABELS_FILE_NAME, index=False)

    validate_splits({TEST_SPLIT_PREFIX: dest_test}, input_structure)

    test_metadata = {
        "task_type": task_type,
        "splits": {"test_rows": len(test_labels[TEST_SPLIT_PREFIX])},
        "numeric_label_col": NUMERIC_LABEL_COLUMN_NAME,
        "input_structure": input_structure,
        "prepared": True,
    }
    test_metadata.update(label_metadata)
    (destination_dir / METADATA_FILE_NAME).write_text(json.dumps(test_metadata, indent=4), encoding="utf-8")

    return test_metadata
