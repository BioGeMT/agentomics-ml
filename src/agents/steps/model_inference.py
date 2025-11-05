import os
import stat
from pathlib import Path
from pydantic import BaseModel, Field

from utils.dataset_utils import get_classes_integers

class ModelInference(BaseModel):
    path_to_inference_file: str = Field(
        description="Absolute path to the generated inference.py"
    )
    inference_summary: str = Field(
        description="Short summary of the inference implementation"
    )
    files_created: list[str] = Field(
        description="""
        A list of files that were created during this step. Leave out the '/workspace/runs/<your_id>/' part of each path.
        """
    )

def get_model_inference_prompt(config):
    """
    Generate the final outcome prompt with specific target classes.
    
    Args:
        target_classes: List of target class values for classification tasks (e.g., ['0', '1', '2']).
                       Should be None or empty for regression tasks.
    """

    if config.task_type == 'classification':
        columns_desc = "\n\t\t".join([
            "- 'prediction': the predicted class (int)",
            *[f"- 'probability_{str(cls)}': probability for class {str(cls)} (float)" for cls in get_classes_integers(config)]
        ])
        output_file_description = f"This file should be a csv file with the following columns:\n{columns_desc}"
    elif config.task_type == 'regression':
        output_file_description = "This file should be a csv file with a single column named 'prediction' containing the predicted continuous values."
    else:
        raise ValueError(f"Unknown task type: {config.task_type}. Supported types are 'classification' and 'regression'.")
    
    #TODO "Except the target column" - use target/class/numeric_label?
    return f"""
    Your next task: create inference.py file.
    If your model can be accelerated by GPU, implement the code to use GPU.
    The inference script must produce a prediction for every single input. Don't skip any samples.
    The inference script will be taking the following named arguments:
    --input (an input file path). This file is of the same format as your training data (except the target column)
    --output (the output file path). {output_file_description}
    """

def lock_inference_file(path_to_inference_file):
    inference_file = Path(path_to_inference_file)
    if inference_file.exists():
        read_only_mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        os.chmod(inference_file, read_only_mode)