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
    def get_task_knowledge(self, metadata: dict) -> str:
        """Task-specific context appended to the dataset description."""
        raise NotImplementedError

    @abstractmethod
    def split_instructions(self) -> tuple[str, str]:
        """Task-specific instructions for the split agent."""
        raise NotImplementedError

    @abstractmethod
    def validate_mini_train(self, train: pd.DataFrame, mini_train: pd.DataFrame) -> None:
        """Task-specific mini-train validation. Raise ValueError if mini_train is invalid."""
        raise NotImplementedError

    @abstractmethod
    def inference_output_description(self, metadata: dict) -> str:
        """Describe the task's prediction format for the inference agent."""
        raise NotImplementedError
