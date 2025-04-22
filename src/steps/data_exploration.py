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

    recommended_focus: str = Field(
        description="""
        Recommended areas to focus on for feature engineering and model selection based on the exploration findings.
        """
    )