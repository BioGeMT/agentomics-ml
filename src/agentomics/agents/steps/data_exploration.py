from __future__ import annotations

from pydantic import Field
from agentomics.agents.steps.base import AgenticStep, AgenticStepOutput
from agentomics.runtime.read_write_utils import load_current_iteration_index


class DataExplorationOutput(AgenticStepOutput):
    data_description: str = Field(
        description="""
        The description of the data, including descriptional statistics and insights you gathered from exploring the data. Include domain-specific features that are relevant to your task.
        """
    )
    feature_analysis: str = Field(
        description="""
        Analysis of individual features: distributions, correlations with target.
        """
    )
    domain_insights: str = Field(
        description="""
        Domain-specific insights you gathered from exploring the data.

        Include:
        - Data type characteristics: properties unique to this type of data
        - Domain context: insights from the dataset description that inform modeling choices
        - Domain-specific challenges or opportunities present in the data
        """
    )
    id_to_sample_info: str = Field(
        description="""
        Explain how samples or conditions are identified and how those identifiers map to the input data and measured responses.
        Provide enough detail for downstream steps to load the corresponding data without repeating exploration.
        """
    )

class DataExplorationStep(AgenticStep):
    step_id = "data_exploration"
    display_name = "DATA EXPLORATION"
    output_type = DataExplorationOutput

    def step_prompt(self) -> str:
        iteration = load_current_iteration_index(self.config)
        if(iteration != 0):
            extra_info = "Note: If you gathered enough information from your previous exploration and don't need to explore the data further, return 'Exploration skipped' in all the json fields (data_description, feature_analysis, domain_insights, id_to_sample_info)."
        else:
            extra_info = ""
        return f"""
        Your next task: explore the dataset. Be thorough, understanding the data deeply will inform subsequent steps for model development.
        {extra_info}
        """
