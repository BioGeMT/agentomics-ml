from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from agentomics.utils.data_module_config import DataModuleConfig
from agentomics.utils.task_types import TaskTypes


class DataCollectionSpec(BaseModel, extra="forbid"):
    task_type: TaskTypes
    description: str = Field(min_length=1)
    inputs: list[str] = Field(min_length=1)
    label: str = Field(min_length=1)
    classes: list[str] | None = None
    known_sources: list[str] = []
    max_download_gb: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    strategy: str = Field(default=DataModuleConfig.DEFAULT_STRATEGY, min_length=1)

    @model_validator(mode="after")
    def _check_classes(self) -> DataCollectionSpec:
        if self.task_type == TaskTypes.CLASSIFICATION and len(set(self.classes or [])) < 2:
            raise ValueError("classification needs at least two distinct classes")
        if self.task_type == TaskTypes.REGRESSION and self.classes:
            raise ValueError("regression takes no classes")
        return self
