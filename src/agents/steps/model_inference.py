from pydantic import BaseModel, Field

from utils.dataset_utils import get_classes_integers

class ModelInference(BaseModel):
    path_to_inference_file: str = Field(
        description="Absolute path to the generated inference.py"
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
    Next task: create inference.py file.
    If your model can be accelerated by GPU, implement the code to use GPU.
    The inference script will be taking the following named arguments:
    --input (an input file path). This file is of the same format as your training data (except the target column)
    --output (the output file path). {output_file_description}
    """
