from pydantic import BaseModel, Field

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
    files_created: list[str] = Field(
        description="""
        A list of files that were created during this step. Leave out the '/workspace/runs/<your_id>/' part of each path.
        """
    )

def get_model_training_prompt():
    return """
    Your next task: implement any necessary code for training a model. Then train a single model.
    
    Requirements:
    1. The train script should save any files necessary to use the trained model for predictions (e.g. model file, tokenizers, ...).
    2. If your model can be accelerated by GPU, implement the code to use GPU.
    """

def get_model_training_prompt_with_progress():
    base_prompt = get_model_training_prompt()
    progress_requirement = """
    3. You MUST use the TrainingProgress class from client_utils.training_progress to report training progress:
       - Import it using: from client_utils.training_progress import TrainingProgress
       - Create an instance: progress = TrainingProgress()
       - Report batch progress in your training loop using:
         progress.report_batch(epoch=current_epoch, batch=batch_idx, metrics={"loss": loss_value})
       - Call progress.end_epoch(epoch) at the end of each epoch
    
    This will help monitor the training progress and ensure we can track metrics during the process.
    """
    return base_prompt.rstrip() + "\n    " + progress_requirement