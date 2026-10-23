from pathlib import Path

import pandas as pd

from agentomics.utils.task_types import TaskTypes

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT_PREFIX = "test"
MINI_TRAIN_SPLIT = "mini_train"

NON_TEST_SPLIT_NAMES = (TRAIN_SPLIT, VALIDATION_SPLIT)

INPUT_DIR_NAME = "input"
SUPPLEMENTARY_DIR_NAME = "supplementary"
PREPARED_DATASETS_DIR_NAME = "prepared_datasets"
DATASET_METADATA_FILE_NAME = "dataset_metadata.json"
LABELS_FILE_NAME = "labels.csv"
GENERATION_LABELS_FILE_NAME = "labels.h5ad"
ID_COLUMN_NAME = "id"
LABEL_COLUMN_NAME = "label"
NUMERIC_LABEL_COLUMN_NAME = "numeric_label"
PREDICTION_COLUMN_NAME = "prediction"
METADATA_FILE_NAME = "metadata.json"
DATASET_DESCRIPTION_FILE_NAME = "dataset_description.md"

ALLOWED_PUBLIC_DATASET_ENTRIES = {
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT_PREFIX,
    SUPPLEMENTARY_DIR_NAME,
    METADATA_FILE_NAME,
    DATASET_DESCRIPTION_FILE_NAME,
}
ALLOWED_SPLIT_ENTRIES = {INPUT_DIR_NAME, LABELS_FILE_NAME}
GENERATION_ALLOWED_SPLIT_ENTRIES = {INPUT_DIR_NAME, GENERATION_LABELS_FILE_NAME}

def labels_file_name(task_type: str | None) -> str:
    """Return the measured-response filename for a task."""
    return GENERATION_LABELS_FILE_NAME if task_type == TaskTypes.CONDITIONAL_GENERATION else LABELS_FILE_NAME


def read_split_samples(split_path: Path, task_type: str | None, value_column: str = NUMERIC_LABEL_COLUMN_NAME) -> pd.DataFrame:
    """Read condition requests for generation, or sample labels for other tasks."""
    if task_type == TaskTypes.CONDITIONAL_GENERATION:
        from agentomics.datasets.conditional_generation_contract import PERTURBATIONS_FILE_NAME, read_perturbations

        return read_perturbations(Path(split_path) / INPUT_DIR_NAME / PERTURBATIONS_FILE_NAME)
    return validate_and_read_labels(Path(split_path) / LABELS_FILE_NAME, value_column)


def predictions_file_suffix(task_type: str) -> str:
    """Conditional generation predicts cells (h5ad); the other task types predict one value per id (csv)."""
    return ".h5ad" if task_type == TaskTypes.CONDITIONAL_GENERATION else ".csv"

def record_input_dir_structure(input_dir: Path) -> list[str]:
    """
    Returns sorted top-level entries in input_dir. Directories are suffixed with '/'.

    Only the top-level interface is recorded: required top-level files like data.zip
    must be present in every split, while sample files inside matching top-level
    directories may differ between train/validation/test.
    """
    return sorted(
        f"{item.name}/" if item.is_dir() else item.name for item in Path(input_dir).iterdir()
    )

def validate_split_entries(split_path: Path, split_name: str, task_type: str | None = None) -> None:
    allowed_entries = (
        GENERATION_ALLOWED_SPLIT_ENTRIES
        if task_type == TaskTypes.CONDITIONAL_GENERATION
        else ALLOWED_SPLIT_ENTRIES
    )
    unsupported_entries = sorted(
        item.name for item in split_path.iterdir() if item.name not in allowed_entries
    )
    if unsupported_entries:
        raise ValueError(
            f"{split_name} split folder has unsupported top-level entries: {unsupported_entries}. "
            f"Allowed: {sorted(allowed_entries)}."
        )

def validate_public_dataset_entries(dataset_dir: Path) -> None:
    unsupported_entries = sorted(
        item.name for item in dataset_dir.iterdir()
        if (
            item.name not in ALLOWED_PUBLIC_DATASET_ENTRIES
            and not (item.is_dir() and item.name.startswith(TEST_SPLIT_PREFIX))
        )
    )
    if unsupported_entries:
        raise ValueError(
            f"Public dataset {dataset_dir.name} has unsupported top-level entries: {unsupported_entries}. "
            f"Allowed: {sorted(ALLOWED_PUBLIC_DATASET_ENTRIES)} and directories "
            f"whose names start with '{TEST_SPLIT_PREFIX}'."
        )

def validate_splits(
    split_paths: dict[str, Path],
    expected_input_structure: list[str],
    task_type: str | None = None,
) -> None:
    """Validates selected split folders against the full dataset contract."""
    for split_name, split_path in split_paths.items():
        _validate_split_folder(split_path, split_name, expected_input_structure, task_type)

def _validate_split_folder(
    split_path: Path,
    split_name: str,
    expected_input_structure: list[str],
    task_type: str | None = None,
) -> None:
    """
    Validates one split folder against the dataset contract.

    Each split must contain input/ and labels.csv, or labels.h5ad for conditional
    generation. The input/ interface must match the expected structure (recorded from
    train/input/ at preparation time).
    """
    split_path = Path(split_path)
    if not split_path.is_dir():
        raise ValueError(f"Split folder must exist and be a directory: {split_path}")

    input_path = split_path / INPUT_DIR_NAME
    labels_path = split_path / labels_file_name(task_type)
    if not input_path.is_dir() or not labels_path.is_file():
        raise ValueError(
            f"{split_name} split folder must contain input/ and {labels_path.name}: {split_path}"
        )

    validate_split_entries(split_path, split_name, task_type)
    _validate_input_structure(input_path, expected_input_structure)
    if task_type == TaskTypes.CONDITIONAL_GENERATION:
        # Imported here: the generation contract module imports this one.
        from agentomics.datasets.conditional_generation_contract import validate_conditional_generation_split

        validate_conditional_generation_split(split_path)
    else:
        validate_and_read_labels(labels_path, NUMERIC_LABEL_COLUMN_NAME, require_numeric_values=True)

def _format_input_structure_mismatch(input_dir: Path, entries: list[str]) -> list[str]:
    formatted_entries = []
    for entry in entries[:10]:
        path = input_dir / entry.rstrip("/")
        if path.is_symlink():
            formatted_entries.append(f"{entry} (symlink -> {path.readlink()})")
        else:
            formatted_entries.append(entry)
    return formatted_entries


def _validate_input_structure(input_dir: Path, expected_structure: list[str]) -> None:
    actual = record_input_dir_structure(input_dir)
    if actual == expected_structure:
        return
    missing = sorted(set(expected_structure) - set(actual))
    extra = sorted(set(actual) - set(expected_structure))

    raise ValueError(
        f"Input structure doesn't match the recorded train/input/ structure: {input_dir}. "
        f"Missing: {_format_input_structure_mismatch(input_dir, missing)}; "
        f"Extra: {_format_input_structure_mismatch(input_dir, extra)} "
        "(showing up to 10 missing/extra entries)"
    )

def validate_and_read_labels(
    labels_path: Path, value_column: str, require_numeric_values: bool = False,
) -> pd.DataFrame:
    """Validate a labels CSV and return its contents as a DataFrame.

    Checks file existence, parseability, column schema, empty rows, empty IDs,
    duplicate IDs, and empty values in the value column. When require_numeric_values
    is True, also validates that values are numeric and finite.
    """
    labels_path = Path(labels_path)
    expected_columns = [ID_COLUMN_NAME, value_column]

    # labels.csv must exist
    if not labels_path.is_file():
        raise ValueError(f"Labels file does not exist: {labels_path}")

    # labels.csv must be parseable as CSV
    try:
        df = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"{labels_path} must be a valid CSV with columns {expected_columns}"
        ) from exc

    # labels.csv must expose only the contract columns
    if set(df.columns) != set(expected_columns):
        raise ValueError(f"{labels_path} must contain exactly columns {sorted(expected_columns)}")

    # labels.csv must contain some labeled examples
    if df.empty:
        raise ValueError(f"{labels_path} contains no label rows")

    # all ids must be non-empty
    ids = df[ID_COLUMN_NAME].str.strip()
    empty_id_mask = ids == ""
    if empty_id_mask.any():
        raise ValueError(f"{labels_path} has {empty_id_mask.sum()} empty id(s)")

    # ids must be unique
    dup_id_mask = ids.duplicated()
    if dup_id_mask.any():
        duplicate_ids = ids[dup_id_mask].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{labels_path} contains duplicate ids: {duplicate_ids} "
            "(showing up to 10)"
        )

    # all values must be non-empty
    values = df[value_column].str.strip()
    empty_value_mask = values == ""
    if empty_value_mask.any():
        raise ValueError(f"{labels_path} has {empty_value_mask.sum()} empty {value_column}(s)")

    if require_numeric_values:
        # values must be numeric
        numeric_values = pd.to_numeric(values, errors="coerce")
        non_numeric_mask = numeric_values.isna()
        if non_numeric_mask.any():
            non_numeric = values[non_numeric_mask].drop_duplicates().head(10).tolist()
            raise ValueError(
                f"{labels_path} has non-numeric {value_column} values: {non_numeric} "
                "(showing up to 10)"
            )

        # values must be finite
        non_finite_mask = numeric_values.isin([float("inf"), float("-inf")])
        if non_finite_mask.any():
            non_finite = values[non_finite_mask].drop_duplicates().head(10).tolist()
            raise ValueError(
                f"{labels_path} has non-finite {value_column} values: {non_finite} "
                "(showing up to 10)"
            )

    return df
