"""Machine learning tasks and their shared lookup interface."""
from agentomics.utils.ml_tasks.base import MLTask
from agentomics.utils.ml_tasks.classification import ClassificationTask
from agentomics.utils.ml_tasks.regression import RegressionTask
from agentomics.utils.ml_tasks.registry import (
    get_task,
    get_task_names,
    get_task_to_metrics_names,
    get_tasks,
    resolve_val_metric,
)

__all__ = [
    "MLTask",
    "ClassificationTask",
    "RegressionTask",
    "get_task",
    "get_task_names",
    "get_task_to_metrics_names",
    "get_tasks",
    "resolve_val_metric",
]
