from enum import StrEnum


class TaskTypes(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CONDITIONAL_GENERATION = "conditional_generation"
    TEMPORAL_GENERATION = "temporal_generation"


# Task types that predict sets of cells (h5ad) for conditions, scored against labels.h5ad.
GENERATION_TASK_TYPES = frozenset({TaskTypes.CONDITIONAL_GENERATION, TaskTypes.TEMPORAL_GENERATION})


def is_generation_task(task_type: str | None) -> bool:
    return task_type in GENERATION_TASK_TYPES
