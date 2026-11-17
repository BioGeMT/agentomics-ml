from abc import ABC, abstractmethod

from agentomics.utils.metrics import Metric


class MLTask(ABC):
    name: str

    @abstractmethod
    def metrics(self) -> dict[str, Metric]:
        """Return supported metrics, including their optimization directions."""
        raise NotImplementedError

    @abstractmethod
    def default_validation_metric(self) -> str:
        """Return a metric name present in metrics()."""
        raise NotImplementedError
