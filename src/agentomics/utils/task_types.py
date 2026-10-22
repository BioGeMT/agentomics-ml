from enum import StrEnum


class TaskTypes(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CONDITIONAL_GENERATION = "conditional_generation"
