from agentomics.utils.ml_tasks.base import MLTask
from agentomics.utils.ml_tasks.classification import ClassificationTask
from agentomics.utils.ml_tasks.regression import RegressionTask


_TASKS: dict[str, MLTask] = {
    task.name: task for task in (ClassificationTask(), RegressionTask())
}


def get_tasks() -> tuple[MLTask, ...]:
    return tuple(_TASKS.values())

def get_task_names() -> tuple[str, ...]:
    return tuple(_TASKS)

def get_task(task_type: str) -> MLTask:
    try:
        return _TASKS[task_type]
    except KeyError:
        raise ValueError(
            f"Unknown task_type: {task_type}. Expected one of {tuple(_TASKS)}."
        ) from None

def get_task_to_metrics_names():
    return {task.name: list(task.metrics()) for task in get_tasks()}

def get_higher_is_better_map():
    """Return a dictionary mapping metric names to their optimization direction."""
    return {
        name: metric.higher_is_better
        for task in get_tasks()
        for name, metric in task.metrics().items()
    }

def get_default_val_metric(task_type: str) -> str:
    return get_task(task_type).default_validation_metric()

def resolve_val_metric(task_type: str, val_metric: str | None = None) -> str:
    if not val_metric:
        return get_default_val_metric(task_type)

    allowed = list(get_task(task_type).metrics())
    if val_metric not in allowed:
        raise ValueError(
            f"Validation metric '{val_metric}' is invalid for task type '{task_type}'. "
            f"Allowed metrics: {', '.join(allowed)}"
        )
    return val_metric
