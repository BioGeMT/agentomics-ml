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
