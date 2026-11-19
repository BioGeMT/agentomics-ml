import math

from sklearn.metrics import (
    average_precision_score, f1_score, log_loss, matthews_corrcoef, roc_auc_score, accuracy_score
)
import numpy as np
import pandas as pd
from rich.console import Console

from agentomics.datasets.data_contract import (
    ID_COLUMN_NAME, LABEL_COLUMN_NAME, NUMERIC_LABEL_COLUMN_NAME, TEST_SPLIT_PREFIX,
)
from agentomics.datasets.label_processing import merge_labels_and_preds
from agentomics.utils.metrics import Metric
from agentomics.utils.ml_tasks.base import MLTask


class ClassificationTask(MLTask):
    name = "classification"
    MINI_TRAIN_SAMPLES_PER_CLASS = 100

    def metrics(self):
        return {
            "ACC": Metric(
                function=lambda y_true, y_pred: accuracy_score(y_true, y_pred),
                needs_probabilities=False,
                higher_is_better=True
            ),
            "AUPRC": Metric(
                function=self._auprc_metric,
                needs_probabilities=True,
                higher_is_better=True
            ),
            "AUROC": Metric(
                function=self._auroc_metric,
                needs_probabilities=True,
                higher_is_better=True
            ),
            "F1": Metric(
                function=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
                needs_probabilities=False,
                higher_is_better=True
            ),
            "LOG_LOSS": Metric(
                function=lambda y_true, y_prob: log_loss(y_true, np.clip(y_prob, 1e-15, 1-1e-15)),
                needs_probabilities=True,
                higher_is_better=False
            ),
            "MCC": Metric(
                function=lambda y_true, y_pred: matthews_corrcoef(y_true, y_pred),
                needs_probabilities=False,
                higher_is_better=True
            ),
        }

    @staticmethod
    def _auroc_metric(y_true, y_prob):
        """Handle AUROC for both binary and multiclass cases."""
        if y_prob.shape[1] == 2:
            # Binary classification
            return roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multiclass classification - use 'ovr' (one-vs-rest) strategy
            return roc_auc_score(y_true, y_prob, multi_class='ovr')

    @staticmethod
    def _auprc_metric(y_true, y_prob):
        """Handle AUPRC for both binary and multiclass cases."""
        if y_prob.shape[1] == 2:
            # Binary classification
            return average_precision_score(y_true, y_prob[:, 1])
        else:
            # Multiclass classification - use macro average
            return average_precision_score(y_true, y_prob, average='macro')

    def default_validation_metric(self) -> str:
        return "AUROC"

    def score(
        self, predictions_with_labels: pd.DataFrame, prediction_columns: list[str],
        *, numeric_label_col: str, pred_col: str, prob_col_prefix: str,
    ) -> dict[str, float]:
        predictions_with_labels[pred_col] = predictions_with_labels[pred_col].astype(int)
        predictions_with_labels[numeric_label_col] = predictions_with_labels[numeric_label_col].astype(int)

        prob_cols = [col for col in prediction_columns if col.startswith(prob_col_prefix)]
        if not prob_cols:
            raise ValueError(f"No probability columns found with prefix '{prob_col_prefix}'.")
        prob_cols_sorted = sorted(prob_cols, key=lambda column_name: int(column_name.split('_')[1]))

        metrics = {}
        metric_to_fn = self.metrics()
        for metric_name, metric_fn in metric_to_fn.items():
            if metric_fn.needs_probabilities:
                metrics[metric_name] = metric_fn(
                    predictions_with_labels[numeric_label_col],
                    predictions_with_labels[prob_cols_sorted].astype(float).values,
                )
            else:
                metrics[metric_name] = metric_fn(predictions_with_labels[numeric_label_col], predictions_with_labels[pred_col])
        return metrics

    def validate_and_numericalize_labels(
        self, split_labels: dict[str, pd.DataFrame], metadata: dict,
        *, fit: bool = False, options: dict | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        _validate_labels(split_labels)
        options = options or {}
        mapping = options.get("label_to_scalar")
        if not fit:
            mapping = metadata.get("label_to_scalar")
            if not mapping:
                raise ValueError("label_to_scalar is required when reusing classification labels")
        elif mapping is None:
            has_override = options.get("positive_class") is not None or options.get("negative_class") is not None
            if "label_to_scalar" in metadata and not has_override:
                mapping = metadata["label_to_scalar"]
            else:
                unique_labels = pd.concat(
                    [frame[LABEL_COLUMN_NAME] for frame in split_labels.values()],
                    ignore_index=True,
                ).dropna().unique()
                if len(unique_labels) == 2 or has_override:
                    mapping = _build_binary_label_mapping(
                        metadata, options.get("positive_class"), options.get("negative_class"),
                        unique_labels, options.get("positive_class_selector"),
                    )
                else:
                    mapping = {label: index for index, label in enumerate(_sort_class_labels(unique_labels))}
                Console().print(f"INFO: Label to number mapping: {mapping}. If this is wrong, please provide --positive-class and --negative-class parameters to the script.")
        mapping = {str(key): int(value) for key, value in mapping.items()}
        if not fit and TEST_SPLIT_PREFIX in split_labels:
            actual = set(split_labels[TEST_SPLIT_PREFIX][LABEL_COLUMN_NAME].astype(str).unique())
            unknown = sorted(actual - set(mapping))
            if unknown:
                raise ValueError(
                    f"Test split contains labels not present in train: {unknown}. "
                    f"Known labels: {sorted(mapping)}"
                )
        return (
            {name: _convert_labels(frame, mapping, name) for name, frame in split_labels.items()},
            {"label_to_scalar": mapping},
        )

    def get_task_knowledge(self, metadata: dict) -> str:
        return f"Label mapping: {metadata.get('label_to_scalar', {})}"

    def split_instructions(self) -> tuple[str, str]:
        count = self.MINI_TRAIN_SAMPLES_PER_CLASS
        return (
            "Ensure that the validation split contains representative samples from ALL classes.",
            f"Mini-train split must contain exactly {count} samples per class "
            f"(or all available if the training split has fewer than {count} for a class)",
        )

    def validate_mini_train(self, train: pd.DataFrame, mini_train: pd.DataFrame) -> None:
        train_counts = train[NUMERIC_LABEL_COLUMN_NAME].value_counts()
        mini_counts = mini_train[NUMERIC_LABEL_COLUMN_NAME].value_counts()
        missing = set(train_counts.index) - set(mini_counts.index)
        if missing:
            raise ValueError(f"mini_train must contain samples from all classes. Missing classes: {sorted(missing)}")
        for label, count in mini_counts.items():
            if label not in train_counts:
                raise ValueError(f"mini_train contains class {label} absent from train")
            expected = min(self.MINI_TRAIN_SAMPLES_PER_CLASS, train_counts[label])
            if count != expected:
                raise ValueError(
                    f"mini_train must have exactly {expected} samples for class {label} "
                    f"(min({self.MINI_TRAIN_SAMPLES_PER_CLASS}, {train_counts[label]}) available in train), "
                    f"got {count}."
                )

    def inference_output_description(self, metadata: dict) -> str:
        probability_columns = "\n".join(
            f"- 'probability_{class_id}': probability for class {class_id} (float)"
            for class_id in sorted(metadata["label_to_scalar"].values())
        )
        return ("A csv with columns:\n- 'id': the input sample id\n"
                f"- 'prediction': the predicted class (int)\n{probability_columns}")

    def get_plot_data(
        self, labeled: pd.DataFrame, predictions: pd.DataFrame, metadata: dict,
    ) -> dict[str, tuple[pd.Series, pd.Series]]:
        mapping = metadata.get("label_to_scalar")
        if not mapping:
            raise ValueError("Classification run config must include non-empty 'label_to_scalar'")
        pred_col = f"probability_{max(mapping.values())}"
        values = merge_labels_and_preds(labeled, predictions, metadata["numeric_label_col"], pred_col)
        return {"roc": values, "pr": values}

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

def _build_binary_label_mapping(
    source_metadata: dict,
    cli_positive_class: str | None,
    cli_negative_class: str | None,
    unique_labels,
    positive_class_selector,
) -> dict[str, int]:
    resolved_positive = cli_positive_class if cli_positive_class is not None else source_metadata.get("positive_class")
    resolved_negative = cli_negative_class if cli_negative_class is not None else source_metadata.get("negative_class")
    has_positive = resolved_positive is not None
    has_negative = resolved_negative is not None
    if has_positive != has_negative:
        raise ValueError("Provide both positive_class and negative_class, or neither.")
    if not has_positive:
        if positive_class_selector is not None:
            resolved_positive = positive_class_selector(unique_labels)
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

def _validate_labels(split_labels: dict[str, pd.DataFrame]) -> None:
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

def _convert_labels(labels: pd.DataFrame, label_to_scalar: dict[str, int], split_name: str) -> pd.DataFrame:
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
