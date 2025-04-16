from pydantic import BaseModel, Field

class DataExploration(BaseModel):
        data_description: str = Field(
            description="""
            The description of the data, including descriptional statistics and insights you gathered from exploring the data. Include domain-specific features that are relevant to your task.
            """
        )