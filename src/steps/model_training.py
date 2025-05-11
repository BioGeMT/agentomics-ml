from pydantic import BaseModel, Field

class ModelTraining(BaseModel):
    path_to_train_file: str = Field(
        description="Absolute path to the generated train.py"
    )

    path_to_model_file: str = Field(
        description="Absolute path to the trained model file"
    )

def get_model_training_prompt():
    return """
    Next task: implement any necessary code for training a model. Then train the model.
    The train script should save any files necessary for inference.
    If your model can be accelerated by GPU, implement the code to use GPU.
    """