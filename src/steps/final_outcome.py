from pydantic import BaseModel, Field

class FinalOutcome(BaseModel):
    path_to_inference_file: str = Field(
        description="Absolute path to the generated inference.py"
    )
    path_to_train_file: str = Field(
        description="Absolute path to the generated train.py"
    )

def get_final_outcome_prompt():
    return """
    Next task: implement any necessary code, train a model, and create inference.py file.
    The train script should save any files necessary for inference.
    If your model can be accelerated by GPU, implement the code to use GPU.
    The inference script will be taking the following named arguments:
    --input (an input file path, file is of the same format as your training data (except the target column))
    --output (the output file path, this file should be a one column csv file with the predictions containing a score from 0 to 1, the column name should be 'prediction')
    """

