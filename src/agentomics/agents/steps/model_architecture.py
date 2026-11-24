from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from agentomics.agents.steps.base import AgentomicsAgenticStep, AgenticStepOutput


ArchitectureFamily = Literal[
    "linear", "decision_tree", "random_forest", "gradient_boosting", "svm",
    "nearest_neighbors", "naive_bayes", "gaussian_process",
    "mlp", "cnn", "rnn", "transformer", "state_space", "gnn", "other",
]


class ArchitectureCategories(BaseModel):
    families: list[ArchitectureFamily] = Field(
        min_length=1,
        description="Families of every model in the predictor, including all ensemble members. Use 'other' for unlisted families.",
    )
    is_ensemble: bool = Field(description="Whether predictions combine several separately trained models.")


class ModelArchitectureOutput(AgenticStepOutput):
    categories: ArchitectureCategories = Field(description="Categories of the planned predictor.")
    architecture: str = Field(
        description="""
        The machine learning model type and architecture for your task.
        """
    )
    hyperparameters: str = Field(
        description="""
        The hyperparameters you have chosen for your model.
        """
    )

class ModelArchitectureStep(AgentomicsAgenticStep):
    step_id = "model_architecture"
    display_name = "ARCHITECTURE"
    output_type = ModelArchitectureOutput

    def step_prompt(self) -> str:
        return """Your next task: choose the model architecture and hyperparameters.
        """
