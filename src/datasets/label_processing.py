from pathlib import Path

import pandas as pd

from datasets.data_contract import (
    ID_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    LABELS_FILE_NAME,
    NUMERIC_LABEL_COLUMN_NAME,
    validate_and_read_labels,
)


def load_split_label_dfs(source_splits: dict[str, Path]) -> dict[str, pd.DataFrame]:
    split_labels = {}
    for split_name, split_path in source_splits.items():
        labels_path = split_path / LABELS_FILE_NAME
        split_labels[split_name] = validate_and_read_labels(labels_path, LABEL_COLUMN_NAME)
    return split_labels

def validate_single_label_classification(split_labels: dict[str, pd.DataFrame]) -> None:
    for split_name, labels in split_labels.items():
        label_values = labels[LABEL_COLUMN_NAME].astype(str).str.strip()
        # Bracket-wrapped labels commonly indicate unsupported multi-label targets.
        multi_label_mask = label_values.str.startswith("[") & label_values.str.endswith("]")
        if multi_label_mask.any():
            examples = label_values[multi_label_mask].drop_duplicates().head(10).tolist()
            raise ValueError(
                f"{split_name}/labels.csv appears to contain multi-label classification values: {examples}. "
                "Only single-label classification is supported."
            )

def convert_classification_labels(labels: pd.DataFrame, label_to_scalar: dict[str, int], split_name: str) -> pd.DataFrame:
    mapped_labels = labels[LABEL_COLUMN_NAME].astype(str).map(label_to_scalar)
    if mapped_labels.isna().any():
        missing_labels = sorted(
            labels.loc[mapped_labels.isna(), LABEL_COLUMN_NAME].astype(str).unique()
        )
        raise ValueError(f"{split_name}/labels.csv contains labels absent from label_to_scalar: {missing_labels}")
    return pd.DataFrame({
        ID_COLUMN_NAME: labels[ID_COLUMN_NAME].astype(str),
        NUMERIC_LABEL_COLUMN_NAME: mapped_labels.astype(int),
    })

def convert_regression_labels(labels: pd.DataFrame, split_name: str) -> pd.DataFrame:
    numeric_labels = pd.to_numeric(labels[LABEL_COLUMN_NAME], errors="coerce")
    if numeric_labels.isna().any():
        invalid_labels = (
            labels.loc[numeric_labels.isna(), LABEL_COLUMN_NAME]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{split_name}/labels.csv has non-numeric label values: {invalid_labels} "
            "(showing up to 10)"
        )
    return pd.DataFrame({
        ID_COLUMN_NAME: labels[ID_COLUMN_NAME].astype(str),
        NUMERIC_LABEL_COLUMN_NAME: numeric_labels,
    })
