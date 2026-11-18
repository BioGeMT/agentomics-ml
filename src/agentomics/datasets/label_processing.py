from pathlib import Path

import pandas as pd

from agentomics.datasets.data_contract import (
    LABEL_COLUMN_NAME,
    LABELS_FILE_NAME,
    validate_and_read_labels,
)


def load_split_label_dfs(source_splits: dict[str, Path]) -> dict[str, pd.DataFrame]:
    split_labels = {}
    for split_name, split_path in source_splits.items():
        labels_path = split_path / LABELS_FILE_NAME
        split_labels[split_name] = validate_and_read_labels(labels_path, LABEL_COLUMN_NAME)
    return split_labels

def merge_labels_and_preds(
    labeled: pd.DataFrame, preds: pd.DataFrame, numeric_label_col: str, pred_col: str,
) -> tuple[pd.Series, pd.Series]:
    if "id" not in labeled.columns or "id" not in preds.columns:
        raise ValueError("Both labeled and predictions CSV must include an 'id' column.")
    if numeric_label_col not in labeled.columns:
        raise ValueError(f"Label column '{numeric_label_col}' not found in labeled CSV. Available: {list(labeled.columns)}")
    if pred_col not in preds.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in predictions. Available: {list(preds.columns)}")
    merged = labeled[["id", numeric_label_col]].merge(preds[["id", pred_col]], on="id", how="inner")
    if merged.empty:
        raise ValueError("After merging on 'id', no rows matched between labeled and predictions CSV.")
    return merged[numeric_label_col], merged[pred_col]
