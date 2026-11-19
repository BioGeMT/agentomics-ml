import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
import numpy as np

from agentomics.datasets.data_contract import (
    ID_COLUMN_NAME, LABEL_COLUMN_NAME, NUMERIC_LABEL_COLUMN_NAME, PREDICTION_COLUMN_NAME,
)
from agentomics.datasets.label_processing import merge_labels_and_preds
from agentomics.utils.metrics import Metric
from agentomics.utils.ml_tasks.base import MLTask


class RegressionTask(MLTask):
    name = "regression"
    MINI_TRAIN_SAMPLE_COUNT = 100

    def metrics(self):
        return {
            "MSE": Metric(
                function=lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                needs_probabilities=False,  # Regression always uses predictions
                higher_is_better=False
            ),
            "RMSE": Metric(
                function=lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                needs_probabilities=False,
                higher_is_better=False
            ),
            "MAE": Metric(
                function=lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
                needs_probabilities=False,
                higher_is_better=False
            ),
            "MAPE": Metric(
                function=lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred),
                needs_probabilities=False,
                higher_is_better=False
            ),
            "PEARSON": Metric(
                function=self._pcc,
                needs_probabilities=False,
                higher_is_better=True
            ),
            "SPEARMAN": Metric(
                function=self._scc,
                needs_probabilities=False,
                higher_is_better=True
            ),
            "R2": Metric(
                function=lambda y_true, y_pred: r2_score(y_true, y_pred),
                needs_probabilities=False,
                higher_is_better=True
            ),
        }

    @staticmethod
    def _pcc(y_true, y_pred):
        r = pearsonr(np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel())[0]
        return float(r) if np.isfinite(r) else 0.0

    @staticmethod
    def _scc(y_true, y_pred):
        """Spearman correlation coefficient."""
        r = spearmanr(np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel())[0]
        return float(r) if np.isfinite(r) else 0.0

    def default_validation_metric(self) -> str:
        return "MAE"

    def score(self, labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
        y_true = labels[NUMERIC_LABEL_COLUMN_NAME].astype(float)
        y_pred = predictions[PREDICTION_COLUMN_NAME].astype(float)
        return {name: metric(y_true, y_pred) for name, metric in self.metrics().items()}

    def validate_and_numericalize_labels(
        self, split_labels: dict[str, pd.DataFrame], metadata: dict,
        *, fit: bool = False, options: dict | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        return {name: _convert_labels(frame, name) for name, frame in split_labels.items()}, {}

    def validate_collection_classes(self, classes: list[str] | None) -> None:
        if classes:
            raise ValueError("regression takes no classes")

    def split_instructions(self) -> tuple[str, str]:
        count = self.MINI_TRAIN_SAMPLE_COUNT
        return "", f"Mini-train split must contain exactly {count} samples (or all available if the training split has fewer than {count})"

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

    def get_plot_data(
        self, labeled: pd.DataFrame, predictions: pd.DataFrame, metadata: dict,
    ) -> dict[str, tuple[pd.Series, pd.Series]]:
        values = merge_labels_and_preds(labeled, predictions, NUMERIC_LABEL_COLUMN_NAME, "prediction")
        return {"pred_vs_actual": values, "residuals_vs_pred": values, "residuals_hist": values}

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
