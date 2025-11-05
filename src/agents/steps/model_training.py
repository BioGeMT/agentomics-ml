from pydantic import BaseModel, Field

class ModelTraining(BaseModel):
    path_to_train_file: str = Field(
        description="Absolute path to the generated train.py"
    )
    path_to_model_file: str = Field(
        description="Absolute path to the trained model file"
    )
    training_summary: str = Field(
        description="Short summary of the training implementation"
    )
    files_created: list[str] = Field(
        description="""
        A list of files that were created during this step.
        """
    )

def get_model_training_prompt():
    return """
    Next task: implement any necessary code for training a model. Then train a single model.
    The train script should save any files necessary to use the trained model for predictions (e.g. model file, tokenizers, ...).
    If your model can be accelerated by GPU, implement the code to use GPU.
    """