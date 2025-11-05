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
    files_created: list[str] = Field(
        description="""
        A list of files that were created during this step. Leave out the '/workspace/runs/<your_id>/' part of each path.
        """
    )

def get_model_architecture_prompt():
    return "Next task: choose the model architecture and hyperparameters."