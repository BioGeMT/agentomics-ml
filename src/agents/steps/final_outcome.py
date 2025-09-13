from pydantic import BaseModel, Field
from typing import List, Optional

class FinalOutcome(BaseModel):
    path_to_inference_file: str = Field(
        description="Absolute path to the generated inference.py"
    )

def get_final_outcome_prompt(target_classes: Optional[List[str]] = None):
    """
    Generate the final outcome prompt with specific target classes.
    
    Args:
        target_classes: List of target class values for classification tasks (e.g., ['0', '1', '2']).
                       Should be None or empty for regression tasks.
    """
    if target_classes:
        # Classification task - generate probability columns
        probability_columns = ", ".join([f"probability_{cls}" for cls in target_classes])
        probability_description = "\n        ".join([
            "- 'prediction': the predicted class",
            *[f"- 'probability_{cls}': probability for class {cls}" for cls in target_classes]
        ])
        
        return f"""
        Next task: create inference.py file.
        The inference script will be taking the following named arguments:
        If your model can be accelerated by GPU, implement the code to use GPU.
        --input (an input file path, file is of the same format as your training data (except the target column))
        --output (the output file path, this file should be a csv file with the following columns:
            {probability_description}
        The format should be: prediction, {probability_columns}
        
        IMPORTANT: Make sure to create ALL the probability columns listed above, even if some classes
        might not appear in the test data. The evaluation system expects all probability columns to be present.
        """
    else:
        # Regression task - no probability columns needed
        return """
        Next task: create inference.py file.
        The inference script will be taking the following named arguments:
        If your model can be accelerated by GPU, implement the code to use GPU.
        --input (an input file path, file is of the same format as your training data (except the target column))
        --output (the output file path, this file should be a csv file with a single column named 'prediction'
        containing the predicted continuous values.
        """