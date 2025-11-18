from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

class DataExploration(BaseModel):
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
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during data exploration step. Populated programmatically.
        """
    )


def get_data_exploration_prompt(iteration):
    if(iteration != 0):
        extra_info = "Note: If you gathered enough information from your previous exploration and don't need to explore the data further, return 'Exploration skipped' in all the json fields (data_description, feature_analysis, domain_insights)."
    else:
        extra_info = ""
    return f"""
    Your first task: explore the dataset. Be thorough, understanding the data deeply will inform subsequent steps for model development.
    {extra_info}
    """