from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

class ModelTraining(BaseModel):
    path_to_train_file: str = Field(
        description="Absolute path to the generated training script Python file"
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

def get_model_training_prompt(config):
    return f"""
    Your next task: implement training code and train your model to optimal performance.
    Training guidelines:
    - Train until convergence or early stopping and output the best model.
    - Save all artifacts needed for inference (model file, tokenizers, etc...).
    - If you failed to implement your intended model, when you call the final_output tool, put into unresolved issues what went wrong.
    {"Use GPU if available for models that benefit from acceleration" if config.check_gpu_availability() else "Implement efficient CPU only training, as you don't have access to GPUs."}
    """