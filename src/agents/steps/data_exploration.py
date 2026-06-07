from __future__ import annotations

from pydantic import Field
from agents.steps.base import AgenticStep, AgenticStepOutput
from runtime.read_write_utils import load_current_iteration_index


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
        How sample IDs in labels.csv relate to the data in input/.
        For example: IDs are row indices in a single CSV file, filename stems of per-sample files, or a column value in a tabular file.
        This must be concrete and specific enough for downstream steps to correctly load data for a given ID without re-exploring.
        """
    )
    supplementary_insights: str | None = Field(
        default=None,
        description="""
        If a supplementary/ folder exists in the dataset, summarize what it contains and how the materials could inform modeling decisions (e.g., a paper describing a specific encoding scheme, reference data, helper scripts).
        Summarize this extensively since it will be used to inform downstream steps and iterations. Do not output the raw contents of the supplementary materials themselves. If you do not explore some supplementary content fully, be explicit about it and suggest that it could be explored in a future iteration.
        Return None if no supplementary/ folder exists or is empty.
        """
    )

class DataExplorationStep(AgenticStep):
    step_id = "data_exploration"
    display_name = "DATA EXPLORATION"
    output_type = DataExplorationOutput

    def step_prompt(self) -> str:
        iteration = load_current_iteration_index(self.config)
        if(iteration != 0):
            extra_info = "Note: If you gathered enough information from your previous exploration and don't need to explore the data further, return 'Exploration skipped' in all the json fields (data_description, feature_analysis, domain_insights, id_to_sample_info, supplementary_insights)."
        else:
            extra_info = ""
        return f"""
        Your next task: explore the dataset. Be thorough, understanding the data deeply will inform subsequent steps for model development.
        {extra_info}
        """
