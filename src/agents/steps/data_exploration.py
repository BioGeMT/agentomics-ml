from pydantic import BaseModel, Field

class DataExploration(BaseModel):
    data_description: str = Field(
        description="""
        The description of the data, including descriptional statistics and insights you gathered from exploring the data. Include domain-specific features that are relevant to your task.
        """
    )
    feature_analysis: str = Field(
        description="""
        Analysis of individual features: distributions, correlations with target, and potential predictive power.
        """
    )
    reasoning: str = Field(
        description="""
        The reasoning behind your data exploration choices.
        """
    )


def get_data_exploration_prompt(iteration):
    if(iteration != 0):
        extra_info = "Note: If you gathered enough information from your previous exploration and don't need to explore the data further, return 'Exploration skipped' in all fields."
    else:
        extra_info = ""    
    return f"""
    Your first task: explore the dataset.
    {extra_info}
    """