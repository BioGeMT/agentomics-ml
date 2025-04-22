from pydantic import BaseModel, Field

class ModelArchitecture(BaseModel):
    architecture: str = Field(
        description="""
        The machine learning model type and architecture you have chosen for your task.
        """
    )
    hyperparameters: str = Field(
        description="""
        The hyperparameters you have chosen for your model.
        """
    )
    reasoning: str = Field(
        description="""
        The reasoning behind your choice of architecture and hyperparameters.
        """
    )