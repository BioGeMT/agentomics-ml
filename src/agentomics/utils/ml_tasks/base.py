from abc import ABC, abstractmethod

import pandas as pd

from agentomics.utils.metrics import Metric


class MLTask(ABC):
    """Define label preparation, scoring, and workflow hooks for an ML task.

    Label DataFrames contain one row per sample, keyed by `id`. Raw label
    tables use `label`, and prepared label tables use `numeric_label`.
    Prediction tables also use `id`, with prediction columns defined by the task.
    """

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
    def score(self, labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
        """Compute task metrics from prepared labels and predictions.

        The caller matches the two tables on `id` so rows at the same
        position refer to the same sample. Read the prediction columns
        required by the task from `predictions`.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_and_numericalize_labels(
        self, split_labels: dict[str, pd.DataFrame], metadata: dict,
        *, fit: bool = False, options: dict | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        """Validate and convert labels to numeric values for each split.

        `split_labels` maps split names to DataFrames with `id` and `label`
        columns. Return DataFrames keyed by the same split names, preserving
        `id` and replacing `label` with `numeric_label`.

        With `fit=True`, the task may derive an encoding or use one supplied
        through `metadata` or `options`. With `fit=False`, reuse the encoding
        from `metadata` if needed.

        Also return a JSON-serializable dictionary saved under `task_metadata`
        in the prepared dataset metadata. Classification returns `label_to_scalar`
        so test preparation and evaluation can reuse the training class IDs.
        Return `{}` when there is no label metadata to save.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_mini_train(self, train: pd.DataFrame, mini_train: pd.DataFrame) -> None:
        """Check task-specific mini-train rules using the prepared label tables.

        Both `train` and `mini_train` contain `id` and `numeric_label` columns.
        Raise `ValueError` if mini-train violates the task's rules.
        """
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
        """Prepare plot data for one split in the final report.

        Return a dictionary keyed by registered plot name, such as `roc` or
        `pred_vs_actual`. Each value is a pair of pandas Series containing
        reference labels first and predictions or scores second.

        The final report passes each pair to its plot renderer and includes
        the resulting image. Return an empty dictionary to skip plots for
        this split. Metrics and other report content are still included.
        """
        return {}
