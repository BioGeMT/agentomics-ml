from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

class ModelTraining(BaseModel):
    path_to_train_file: str = Field(
        description="Absolute path to the generated train.py"
    )
    path_to_model_file: str = Field(
        description="Absolute path to the trained model file"
    )
    training_summary: str = Field(
        description="Short summary of the training implementation. Don't include any metrics in this summary."
    )
    unresolved_issues: str|None = Field(
        description="Issues that remain unresolved and could impact performance and/or metrics. (e.g. expected GPU to be available but is inaccessible during training, foundation model could not be loaded, etc...). Can be empty."
    )
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during model training step. Populated programmatically.
        """
    )

def get_model_training_prompt():
    return """
    Your next task: implement any necessary code for training a model. Then train a single model.
    The train script should save any files necessary to use the trained model for predictions (e.g. model file, tokenizers, ...).
    If your model can be accelerated by GPU, implement the code to use GPU.
    """