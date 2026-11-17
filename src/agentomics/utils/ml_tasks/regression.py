from agentomics.utils.metrics import Metric, get_regression_metrics_functions
from agentomics.utils.ml_tasks.base import MLTask


class RegressionTask(MLTask):
    name = "regression"

    def metrics(self) -> dict[str, Metric]:
        return get_regression_metrics_functions()

    def default_validation_metric(self) -> str:
        return "MAE"