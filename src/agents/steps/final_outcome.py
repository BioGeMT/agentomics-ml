from pydantic import BaseModel, Field

class FinalOutcome(BaseModel):
    path_to_inference_file: str = Field(
        description="Absolute path to the generated inference.py"
    )

def get_final_outcome_prompt():
    return """
    Next task: create inference.py file.
    The inference script will be taking the following named arguments:
    If your model can be accelerated by GPU, implement the code to use GPU.
    --input (an input file path, file is of the same format as your training data (except the target column))
    --output (the output file path, this file should be a one column csv file with the predictions, the column name should be 'prediction')
    """

