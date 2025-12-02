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


def get_model_architecture_prompt(user_prompt: str = None, is_fork: bool = False):
    """
    Prompt for choosing model architecture and hyperparameters.

    If a custom user prompt is provided (either from the CLI or inherited
    when forking), append it as explicit user instructions.
    """
    base_prompt = "Your next task: choose the model architecture and hyperparameters."
    
    if user_prompt:
        return f"{base_prompt}\n\nUser instructions: {user_prompt}"
    
    return base_prompt
