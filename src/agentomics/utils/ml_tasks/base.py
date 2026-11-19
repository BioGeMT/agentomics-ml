from abc import ABC, abstractmethod

import pandas as pd

from agentomics.utils.metrics import Metric


class MLTask(ABC):
    name: str

    @abstractmethod
    def metrics(self) -> dict[str, Metric]:
        """Return supported metrics, including their optimization directions."""
        raise NotImplementedError

    @abstractmethod
    def default_validation_metric(self) -> str:
        """Task-specific default metric used to select the best model."""
        raise NotImplementedError

    @abstractmethod
    def score(
        self, predictions_with_labels: pd.DataFrame, prediction_columns: list[str],
        *, numeric_label_col: str, pred_col: str, prob_col_prefix: str,
    ) -> dict[str, float]:
        """Compute task specific metrics by comparing predictions and reference labels.

        The caller joins predictions and reference labels on the sample ID.
        Each row in 'predictions_with_labels' contains a prediction and a reference label for the same sample.

        'prediction_columns' lists the columns from the prediction DataFrame before the join. 'numeric_label_col'
        identifies the reference label column, and 'pred_col' identifies the predicted value or class.
        For tasks that use class probabilities, 'prob_col_prefix' identifies the probability columns among 'prediction_columns'.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_labels(
        self, split_labels: dict[str, pd.DataFrame], metadata: dict,
        *, fit: bool = False, options: dict | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        """Validate and transform labels according to the task's requirements.

        fit=True may derive task metadata from training/validation labels.
        The default reuses existing metadata, including for held-out test labels.
        options contains preparation overrides and optional interactive callbacks.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_mini_train(self, train: pd.DataFrame, mini_train: pd.DataFrame) -> None:
        """Task-specific mini-train validation. Raise ValueError if mini_train is invalid."""
        raise NotImplementedError

    @abstractmethod
    def inference_output_description(self, metadata: dict) -> str:
        """Describe the task's prediction format for the inference agent."""
        raise NotImplementedError

    def get_task_knowledge(self, metadata: dict) -> str:
        """Optional task-specific context appended to the dataset description."""
        return ""

    def split_instructions(self) -> tuple[str, str]:
        """Optional task-specific instructions for the split agent.
        Returns a tuple of (validation instructions, mini-train instructions).
        """
        return ("", "")

    def get_plot_data(
        self, labeled: pd.DataFrame, predictions: pd.DataFrame, metadata: dict,
    ) -> dict[str, tuple[pd.Series, pd.Series]]:
        """Prepare optional task-specific plot data.
        Subclasses may omit this method; only the plot section is skipped.
        """
        return {}
