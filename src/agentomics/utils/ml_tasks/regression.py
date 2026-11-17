import pandas as pd

from agentomics.datasets.data_contract import ID_COLUMN_NAME, LABEL_COLUMN_NAME, NUMERIC_LABEL_COLUMN_NAME
from agentomics.utils.metrics import Metric, get_regression_metrics_functions
from agentomics.utils.ml_tasks.base import MLTask


class RegressionTask(MLTask):
    name = "regression"
    MINI_TRAIN_SAMPLE_COUNT = 100

    def metrics(self) -> dict[str, Metric]:
        return get_regression_metrics_functions()

    def default_validation_metric(self) -> str:
        return "MAE"

    def score(
        self, merged: pd.DataFrame, prediction_columns: list[str],
        *, numeric_label_col: str, pred_col: str, prob_col_prefix: str,
    ) -> dict[str, float]:
        merged[numeric_label_col] = merged[numeric_label_col].astype(float)
        merged[pred_col] = merged[pred_col].astype(float)

        metrics = {}
        metric_to_fn = self.metrics()
        for metric_name, metric_fn in metric_to_fn.items():
            metrics[metric_name] = metric_fn(merged[numeric_label_col], merged[pred_col])
        return metrics

    def prepare_labels(
        self, split_labels: dict[str, pd.DataFrame], metadata: dict,
        *, fit: bool = False, options: dict | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        return {name: _convert_labels(frame, name) for name, frame in split_labels.items()}, {}

    def get_task_knowledge(self, metadata: dict) -> str:
        return ""

    def split_instructions(self) -> tuple[str, str]:
        count = self.MINI_TRAIN_SAMPLE_COUNT
        return "", f"exactly {count} samples (or all available if the training split has fewer than {count})"

    def validate_mini_train(self, train: pd.DataFrame, mini_train: pd.DataFrame) -> None:
        expected = min(self.MINI_TRAIN_SAMPLE_COUNT, len(train))
        if len(mini_train) != expected:
            raise ValueError(
                f"mini_train must have exactly {expected} samples "
                f"(min({self.MINI_TRAIN_SAMPLE_COUNT}, {len(train)}) available in train), "
                f"got {len(mini_train)}."
            )

    def inference_output_description(self, metadata: dict) -> str:
        return "A csv with columns 'id' and 'prediction', where 'prediction' contains the predicted continuous value."


def _convert_labels(labels: pd.DataFrame, split_name: str) -> pd.DataFrame:
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
