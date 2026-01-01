from pydantic import BaseModel, Field

class ModelTraining(BaseModel):
    path_to_train_file: str = Field(
        description="Absolute path to the generated train.py"
    )

    path_to_model_file: str = Field(
        description="Absolute path to the trained model file"
    )

def get_model_training_prompt(config):
    if 'model_training' in config.steps_to_skip:
        return """
        When training the model, the python file should be named train.py.
        """
    return """
    Next task: implement any necessary code for training a model. Then train a single model.
    The train script should save any files necessary to use the trained model for predictions (e.g. model file, tokenizers, ...).
    If your model can be accelerated by GPU, implement the code to use GPU.
    """