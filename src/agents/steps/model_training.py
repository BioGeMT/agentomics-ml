from pydantic import BaseModel, Field

class ModelTraining(BaseModel):
    path_to_train_file: str = Field(
        description="Absolute path to the generated train.py"
    )

    path_to_model_file: str = Field(
        description="Absolute path to the trained model file"
    )

def get_model_training_prompt(config):
    prompt = f"""implement any necessary code for training a model. Then train a single model.                            
      The train script should save any files necessary to use the trained model for predictions (e.g. model file, tokenizers, ...).
      If your model can be accelerated by GPU, implement the code to use GPU.
      The python file for training should be named train.py and saved in {config.runs_dir / config.agent_id}.
      """
    if 'model_training' in config.steps_to_skip:
        return "For the training strategy: \n" + prompt
    return "Next task: " + prompt