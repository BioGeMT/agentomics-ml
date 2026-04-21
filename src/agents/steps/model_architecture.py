from __future__ import annotations

from pydantic import Field
from agents.steps.base import AgenticStep, AgenticStepOutput


class ModelArchitectureOutput(AgenticStepOutput):
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

class ModelArchitectureStep(AgenticStep):
    step_id = "model_architecture"
    display_name = "ARCHITECTURE"
    output_type = ModelArchitectureOutput

    def step_prompt(self) -> str:
        return """Your next task: choose the model architecture and hyperparameters.
        """
