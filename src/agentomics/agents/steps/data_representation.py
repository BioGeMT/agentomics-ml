from __future__ import annotations

from pydantic import Field
from agentomics.agents.steps.base import AgenticStep, AgenticStepOutput


class DataRepresentationOutput(AgenticStepOutput):
    representation: str = Field(
        description="""
        How will the data be represented, including any transformations, encodings, normalizations, features, and label transformations.
        """
    )

class DataRepresentationStep(AgenticStep):
    step_id = "data_representation"
    display_name = "REPRESENTATION"
    output_type = DataRepresentationOutput

    def step_prompt(self) -> str:
        return "Your next task: define the data representation."
