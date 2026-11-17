from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from agentomics.utils.data_module_config import DataModuleConfig
from agentomics.utils.ml_tasks.base import MLTask
from agentomics.utils.ml_tasks.classification import ClassificationTask
from agentomics.utils.ml_tasks.regression import RegressionTask


class DataCollectionSpec(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    task_type: MLTask
    description: str = Field(min_length=1)
    inputs: list[str] = Field(min_length=1)
    label: str = Field(min_length=1)
    classes: list[str] | None = None
    known_sources: list[str] = []
    max_download_gb: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    strategy: str = Field(default=DataModuleConfig.DEFAULT_STRATEGY, min_length=1)

    @model_validator(mode="after")
    def _check_classes(self) -> DataCollectionSpec:
        if isinstance(self.task_type, ClassificationTask) and len(set(self.classes or [])) < 2:
            raise ValueError("classification needs at least two distinct classes")
        if isinstance(self.task_type, RegressionTask) and self.classes:
            raise ValueError("regression takes no classes")
        return self
