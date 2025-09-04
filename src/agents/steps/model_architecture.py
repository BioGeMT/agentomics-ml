from pydantic import BaseModel, Field

class ModelArchitecture(BaseModel):
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
    reasoning: str = Field(
        description="""
        The reasoning behind your model architecture and hyperparameter choices.
        """
    )

def get_model_architecture_prompt():
    return "Next task: choose the model architecture and hyperparameters."