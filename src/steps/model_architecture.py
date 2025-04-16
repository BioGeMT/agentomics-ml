from pydantic import BaseModel, Field

class ModelArchitecture(BaseModel):
        architecture: str = Field(
            description="""
            The machine learning model type and architecture you have chosen for your task.
            """
        )
        reasoning: str = Field(
            description="""
            The reasoning behind your choice of architecture
            """
        )