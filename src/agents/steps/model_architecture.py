from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

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
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during model architecture step. Populated programmatically.
        """
    )

def get_model_architecture_prompt():
    return """Your next task: choose the model architecture and hyperparameters.
    Goal: Select an approach that balances model capacity and generalization, given your dataset characteristics and available resources.
    """