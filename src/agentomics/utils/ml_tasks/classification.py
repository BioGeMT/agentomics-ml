from agentomics.utils.metrics import Metric, get_classification_metrics_functions
from agentomics.utils.ml_tasks.base import MLTask


class ClassificationTask(MLTask):
    name = "classification"

    def metrics(self) -> dict[str, Metric]:
        return get_classification_metrics_functions()

    def default_validation_metric(self) -> str:
        return "AUROC"