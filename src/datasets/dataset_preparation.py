import json
import math
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

from runtime.filesystem import copy_path_overwriting_target, remove_path
from datasets.data_contract import (
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
    validate_splits,
)
from datasets.datasets_interactive import (
    select_positive_class,
    select_task_type,
)
from datasets.label_processing import (
    convert_classification_labels,
    convert_regression_labels,
    load_split_label_dfs,
    validate_single_label_classification,
)
from utils.task_types import TaskTypes

console = Console()

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
        "test_rows": len(split_labels[TEST_SPLIT]) if TEST_SPLIT in split_labels else None,
    }
    metadata["numeric_label_col"] = NUMERIC_LABEL_COLUMN_NAME
    metadata["input_structure"] = input_structure
    return metadata

def prepare_dataset(
    dataset_dir,
    output_dir,
    test_sets_output_dir,
    task_type=None,
    positive_class=None,
    negative_class=None,
    interactive=False,
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    test_sets_output_dir = Path(test_sets_output_dir)

    dataset_name = dataset_dir.name
    prepared_dir = output_dir / dataset_name
    prepared_test_dir = test_sets_output_dir / dataset_name
    if dataset_dir.resolve() == prepared_dir.resolve():
        raise ValueError("Input dataset directory must be different from the prepared output directory.")

    source_splits = {
        split_name: dataset_dir / split_name
        for split_name in [*NON_TEST_SPLIT_NAMES, TEST_SPLIT]
        if (dataset_dir / split_name).is_dir()
    }

    if TRAIN_SPLIT not in source_splits:
        raise FileNotFoundError(f"Required train/ split is missing: {dataset_dir / TRAIN_SPLIT}")

    input_structure = record_input_dir_structure(source_splits[TRAIN_SPLIT] / INPUT_DIR_NAME)
    split_labels = load_split_label_dfs(source_splits)

    metadata_path = dataset_dir / METADATA_FILE_NAME
    source_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    resolved_task_type = _resolve_task_type(source_metadata, task_type, split_labels[TRAIN_SPLIT], interactive)

    output_metadata = _build_output_metadata(
        source_metadata, resolved_task_type, split_labels, input_structure,
    )

    if resolved_task_type == TaskTypes.CLASSIFICATION:
        validate_single_label_classification(split_labels)
        has_cli_class_override = positive_class is not None or negative_class is not None
        if "label_to_scalar" in source_metadata and not has_cli_class_override:
            label_to_scalar = {str(k): int(v) for k, v in source_metadata["label_to_scalar"].items()}
        else:
            non_test_labels = [
                split_labels[split_name]
                for split_name in NON_TEST_SPLIT_NAMES
                if split_name in split_labels
            ]
            unique_labels = pd.concat(
                [frame[LABEL_COLUMN_NAME] for frame in non_test_labels],
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
        if TEST_SPLIT in split_labels:
            test_labels = set(split_labels[TEST_SPLIT][LABEL_COLUMN_NAME].astype(str).unique())
            unknown_labels = sorted(test_labels - set(label_to_scalar))
            if unknown_labels:
                raise ValueError(
                    f"Test split contains labels not present in train: {unknown_labels}. "
                    f"Known labels: {sorted(label_to_scalar)}"
                )
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

    remove_path(prepared_dir)
    prepared_dir.mkdir(parents=True)
    remove_path(prepared_test_dir)

    for split_name in NON_TEST_SPLIT_NAMES:
        if split_name in source_splits:
            copy_path_overwriting_target(source_splits[split_name], prepared_dir / split_name)
            numeric_split_labels[split_name].to_csv(prepared_dir / split_name / LABELS_FILE_NAME, index=False)

    if TEST_SPLIT in source_splits:
        prepared_test_dir.mkdir(parents=True)
        copy_path_overwriting_target(source_splits[TEST_SPLIT], prepared_test_dir / TEST_SPLIT)
        numeric_split_labels[TEST_SPLIT].to_csv(prepared_test_dir / TEST_SPLIT / LABELS_FILE_NAME, index=False)

    prepared_splits = {
        split_name: prepared_dir / split_name
        for split_name in NON_TEST_SPLIT_NAMES
        if split_name in source_splits
    }
    validate_splits(prepared_splits, input_structure)
    if TEST_SPLIT in source_splits:
        validate_splits({TEST_SPLIT: prepared_test_dir / TEST_SPLIT}, input_structure)

    supplementary_source = dataset_dir / SUPPLEMENTARY_DIR_NAME
    if supplementary_source.is_dir():
        copy_path_overwriting_target(supplementary_source, prepared_dir / SUPPLEMENTARY_DIR_NAME)

    description_source = dataset_dir / DATASET_DESCRIPTION_FILE_NAME
    if description_source.exists():
        copy_path_overwriting_target(description_source, prepared_dir / DATASET_DESCRIPTION_FILE_NAME)
    else:
        console.print("INFO: No dataset description provided.")
        (prepared_dir / DATASET_DESCRIPTION_FILE_NAME).write_text(
            "No dataset description available.",
            encoding="utf-8",
        )

    # Written last - check_dataset_prepared uses metadata.json existence as a succesful preparation proof
    (prepared_dir / METADATA_FILE_NAME).write_text(json.dumps(output_metadata, indent=4), encoding="utf-8")

def check_dataset_prepared(dataset_dir: Path, prepared_datasets_dir: Path) -> bool:
    return (prepared_datasets_dir / dataset_dir.name / METADATA_FILE_NAME).is_file()

def prepare_all_datasets(
    datasets_dir: str,
    prepared_datasets_dir: str,
    prepared_test_sets_dir: str,
) -> None:
    datasets_path = Path(datasets_dir)

    if not datasets_path.exists():
        console.print(f"[red]No datasets found in {datasets_dir}[/red]")
        sys.exit(1)

    dataset_dirs = sorted(
        d for d in datasets_path.iterdir() if d.is_dir()
    )

    if not dataset_dirs:
        console.print(f"[red]No datasets found in {datasets_dir}[/red]")
        sys.exit(1)

    skipped = 0
    prepared = 0
    failed = 0

    for dataset_dir in dataset_dirs:
        if check_dataset_prepared(dataset_dir, Path(prepared_datasets_dir)):
            skipped += 1
            continue
        console.print(f"[cyan]Preparing {dataset_dir.name}[/cyan]")
        try:
            prepare_dataset(
                dataset_dir=dataset_dir,
                output_dir=prepared_datasets_dir,
                test_sets_output_dir=prepared_test_sets_dir,
            )
            console.print(f"[green]{dataset_dir.name} prepared successfully[/green]")
            prepared += 1
        except Exception as e:
            console.print(f"[red]{dataset_dir.name} failed: {e}[/red]")
            failed += 1

    console.print("")
    console.print(f"Prepared: {prepared}, skipped (already prepared): {skipped}, failed: {failed}")
    if failed > 0:
        sys.exit(1)

def setup_nonsensitive_dataset_files_for_agent(prepared_datasets_dir: Path, agent_datasets_dir: Path, dataset_name: str):
    source_dataset_dir = prepared_datasets_dir / dataset_name
    target_dataset_dir = agent_datasets_dir / dataset_name
    target_dataset_dir.mkdir(parents=True, exist_ok=True)

    target_paths = [
        DATASET_DESCRIPTION_FILE_NAME,
        METADATA_FILE_NAME,
        SUPPLEMENTARY_DIR_NAME,
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
    ]
    for relative_path in target_paths:
        source_path = source_dataset_dir / relative_path
        target_path = target_dataset_dir / relative_path

        if source_path.exists():
            copy_path_overwriting_target(source_path, target_path)
