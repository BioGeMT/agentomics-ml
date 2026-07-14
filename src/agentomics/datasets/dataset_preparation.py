import json
import math
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

from agentomics.datasets.data_contract import (
    DATASET_DESCRIPTION_FILE_NAME,
    INPUT_DIR_NAME,
    LABEL_COLUMN_NAME,
    LABELS_FILE_NAME,
    METADATA_FILE_NAME,
    NON_TEST_SPLIT_NAMES,
    NUMERIC_LABEL_COLUMN_NAME,
    SUPPLEMENTARY_DIR_NAME,
    TEST_SPLIT,
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
    convert_classification_labels,
    convert_regression_labels,
    load_split_label_dfs,
    validate_single_label_classification,
)
from agentomics.runtime.filesystem import create_absolute_symlink, find_symlinks_in_dir
from agentomics.utils.task_types import TaskTypes

console = Console()

def _reject_symlinks_in_dir(directory: Path) -> None:
    symlinks = find_symlinks_in_dir(directory, include_root=True)
    if symlinks:
        raise ValueError(
            f"Symlinks in {directory} are not allowed. "
            f"{len(symlinks)} symlink(s) found (e.g. {symlinks[0]})."
        )

def _sort_class_labels(labels) -> list[str]:
    string_labels = [str(label) for label in labels]

    # If every label is numeric-looking, sort by value: "1", "2", "10".
    # If any label is non-numeric, keep the whole set lexicographic.
    try:
        numeric_values = [float(label.strip()) for label in string_labels]
        all_labels_are_numeric = all(math.isfinite(value) for value in numeric_values)
    except ValueError:
        all_labels_are_numeric = False

    if all_labels_are_numeric:
        return sorted(string_labels, key=lambda label: float(label.strip()))
    return sorted(string_labels)

def _resolve_task_type(source_metadata: dict, cli_task_type: str | None, train_labels: pd.DataFrame, interactive: bool) -> str:
    resolved = cli_task_type or source_metadata.get("task_type")
    if resolved is None:
        resolved = select_task_type(train_labels, interactive=interactive)
    if resolved not in TaskTypes:
        raise ValueError(f"Invalid task_type '{resolved}'. Must be one of: {TaskTypes}")
    return resolved

def _build_binary_label_mapping(
    source_metadata: dict,
    cli_positive_class: str | None,
    cli_negative_class: str | None,
    unique_labels,
    interactive: bool,
) -> dict[str, int]:
    resolved_positive = cli_positive_class if cli_positive_class is not None else source_metadata.get("positive_class")
    resolved_negative = cli_negative_class if cli_negative_class is not None else source_metadata.get("negative_class")
    has_positive = resolved_positive is not None
    has_negative = resolved_negative is not None
    if has_positive != has_negative:
        raise ValueError("Provide both positive_class and negative_class, or neither.")
    if not has_positive:
        if interactive and sys.stdin.isatty():
            resolved_positive = select_positive_class(unique_labels)
            resolved_negative = next(str(label) for label in unique_labels if str(label) != str(resolved_positive))
        else:
            sorted_labels = _sort_class_labels(unique_labels)
            resolved_negative = sorted_labels[0]
            resolved_positive = sorted_labels[1]
    label_set = {str(label) for label in unique_labels}
    resolved_set = {str(resolved_positive), str(resolved_negative)}
    if resolved_set != label_set:
        raise ValueError(
            f"positive_class and negative_class must match the observed labels. "
            f"Observed: {sorted(label_set)}; requested: {sorted(resolved_set)}"
        )
    return {str(resolved_negative): 0, str(resolved_positive): 1}

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

    resolved_task_type = _resolve_task_type(source_metadata, task_type, split_labels[TRAIN_SPLIT], interactive)
    output_metadata = _build_output_metadata(
        source_metadata, resolved_task_type, split_labels, input_structure,
    )

    if resolved_task_type == TaskTypes.CLASSIFICATION:
        validate_single_label_classification(split_labels)
        has_cli_class_override = positive_class is not None or negative_class is not None
        if label_to_scalar is not None:
            label_to_scalar = {str(k): int(v) for k, v in label_to_scalar.items()}
        elif "label_to_scalar" in source_metadata and not has_cli_class_override:
            label_to_scalar = {str(k): int(v) for k, v in source_metadata["label_to_scalar"].items()}
        else:
            unique_labels = pd.concat(
                [frame[LABEL_COLUMN_NAME] for frame in split_labels.values()],
                ignore_index=True,
            ).dropna().unique()
            if len(unique_labels) == 2 or has_cli_class_override:
                label_to_scalar = _build_binary_label_mapping(
                    source_metadata, positive_class, negative_class, unique_labels, interactive,
                )
            else:
                sorted_labels = _sort_class_labels(unique_labels)
                label_to_scalar = {str(label): index for index, label in enumerate(sorted_labels)}
            console.print(f"INFO: Label to number mapping: {label_to_scalar}. If this is wrong, please provide --positive-class and --negative-class parameters to the script.")
        output_metadata["label_to_scalar"] = label_to_scalar
        numeric_split_labels = {
            split_name: convert_classification_labels(labels, label_to_scalar, split_name)
            for split_name, labels in split_labels.items()
        }
    elif resolved_task_type == TaskTypes.REGRESSION:
        numeric_split_labels = {
            split_name: convert_regression_labels(labels, split_name)
            for split_name, labels in split_labels.items()
        }
    else:
        raise ValueError(f"Unknown task_type: {resolved_task_type}. Expected one of {TaskTypes}.")

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
) -> dict:
    """Prepare test dataset from source_dir into destination_dir using run metadata.

    Reads from source_dir (never modified), writes the prepared result to
    destination_dir (input dir is symlinked, labels are converted, metadata
    is written). Returns the output metadata dict.
    """
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    _reject_symlinks_in_dir(source_dir)
    if is_test_csv_dataset(source_dir):
        source_dir = convert_csv_test_dataset_to_standard_raw_dataset(source_dir, destination_dir)

    test_source = source_dir / TEST_SPLIT
    if not test_source.is_dir():
        raise FileNotFoundError(f"Required test/ split is missing: {test_source}")

    validate_split_entries(test_source, TEST_SPLIT)

    test_labels = load_split_label_dfs({TEST_SPLIT: test_source})

    if task_type == TaskTypes.CLASSIFICATION:
        if not label_to_scalar:
            raise ValueError("label_to_scalar is required for classification test data preparation")
        validate_single_label_classification(test_labels)
        label_to_scalar = {str(k): int(v) for k, v in label_to_scalar.items()}
        actual_labels = set(test_labels[TEST_SPLIT][LABEL_COLUMN_NAME].astype(str).unique())
        unknown_labels = sorted(actual_labels - set(label_to_scalar))
        if unknown_labels:
            raise ValueError(
                f"Test split contains labels not present in train: {unknown_labels}. "
                f"Known labels: {sorted(label_to_scalar)}"
            )
        numeric_labels = convert_classification_labels(test_labels[TEST_SPLIT], label_to_scalar, TEST_SPLIT)
    elif task_type == TaskTypes.REGRESSION:
        numeric_labels = convert_regression_labels(test_labels[TEST_SPLIT], TEST_SPLIT)
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Expected one of {TaskTypes}.")

    dest_test = destination_dir / TEST_SPLIT
    dest_test.mkdir(parents=True, exist_ok=True)
    create_absolute_symlink(test_source / INPUT_DIR_NAME, dest_test / INPUT_DIR_NAME)
    numeric_labels.to_csv(dest_test / LABELS_FILE_NAME, index=False)

    validate_splits({TEST_SPLIT: dest_test}, input_structure)

    test_metadata = {
        "task_type": task_type,
        "splits": {"test_rows": len(test_labels[TEST_SPLIT])},
        "numeric_label_col": NUMERIC_LABEL_COLUMN_NAME,
        "input_structure": input_structure,
        "prepared": True,
    }
    if task_type == TaskTypes.CLASSIFICATION:
        test_metadata["label_to_scalar"] = label_to_scalar
    (destination_dir / METADATA_FILE_NAME).write_text(json.dumps(test_metadata, indent=4), encoding="utf-8")

    return test_metadata
