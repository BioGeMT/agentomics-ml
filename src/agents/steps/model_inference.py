import os
import stat
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from utils.dataset_utils import get_classes_integers

class ModelInference(BaseModel):
    path_to_inference_file: str = Field(
        description="Absolute path to the generated inference.py"
    )
    inference_summary: str = Field(
        description="Short summary of the inference implementation"
    )
    unresolved_issues: str|None = Field(
        description="Issues that remain unresolved and could impact performance and/or metrics. (e.g. expected GPU to be available but is inaccessible during inference, foundation model could not be loaded, etc...). Can be empty."
    )
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during model inference step. Populated programmatically.
        """
    )

def get_model_inference_prompt(config, training_artifacts_dir):
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
    #TODO validate the script uses the artifacts-dir stuff and has not hard-coded paths
    return f"""
    Your next task: create inference.py file.
    If your model can be accelerated by GPU, implement the code to use GPU.
    The inference script must produce a prediction for every single input. Don't skip any samples.
    The inference script must use the same architecture as your current trained model from 'train.py' and use the artifacts produced by that script (located at '{training_artifacts_dir}').
    The inference script will be taking the following named arguments:
    --input (an input file path). This file is of the same format as your training data (except the target column)
    --output (the output file path). {output_file_description}
    --artifacts-dir (the folder that will include training artifacts from the training step that are needed to run inference (for example model weights, tokenizers, etc..). It should be optional, with the following dir as a default: '{training_artifacts_dir}'. If a different path is provided, your script must adapt to the new source. You can assume the artifact files will always have the same name. 
    """

def lock_inference_file(path_to_inference_file):
    inference_file = Path(path_to_inference_file)
    if inference_file.exists():
        read_only_mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        os.chmod(inference_file, read_only_mode)