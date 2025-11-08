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
    files_created: list[str]|None = Field(
        description="""
        This field should be passed as an empty list, as this will be overwritten and populated programatically.
        """
    )

def get_model_architecture_prompt():
    return "Your next task: choose the model architecture and hyperparameters."