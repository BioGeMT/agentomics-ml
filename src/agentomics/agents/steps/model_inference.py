import os
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, RunContext, ModelRetry

from agentomics.utils.dataset_utils import get_classes_integers
from agentomics.agents.agent_utils import does_file_contain_iteration_pattern, does_file_contain_string, get_new_rundir_files
from agentomics.run_logging.evaluate_log_run import run_inference_and_log

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
    The inference script must produce a prediction for every single input. Don't skip any samples. The 'id' column from the input file must be preserved in the output file.
    The inference script must use the same architecture as your current trained model from 'train.py' and use the artifacts produced by that script (located at '{training_artifacts_dir}').
    The inference script will be taking the following named arguments:
    --input (an input file path). This file is of the same format as your training data (except the target column)
    --output (the output file path). {output_file_description}
    --artifacts-dir (the folder that contains training artifacts from the training step that are needed to run inference (for example model weights, tokenizers, etc..). The following dir should be used as a default: '{training_artifacts_dir}'. If a different path is provided, your script must adapt to the new source. You can assume the artifact files will always have the same name. 
    The script must not accept any other parameters.
    """

def create_model_inference_agent(config, model, tools):
    inference_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config.temperature},
        output_type= ModelInference,
        retries=config.max_validation_retries,
        deps_type=dict,
    )

    @inference_agent.output_validator
    async def validate_inference(ctx: RunContext[dict], result: ModelInference) -> ModelInference:
        if not os.path.exists(result.path_to_inference_file):
            raise ModelRetry(f"Inference file does not exist at {result.path_to_inference_file}")
        if os.path.islink(result.path_to_inference_file):
            raise ModelRetry(f"Inference file ({result.path_to_inference_file}) cannot be a symbolic link, create a non-symlinked copy of it.")
        if does_file_contain_iteration_pattern(result.path_to_inference_file):
            raise ModelRetry("Inference file contains path containing a forbidden string 'iteration_' or references an iteration folder, which will not accessible during final testing. If you want to re-use a file from a past iteration, copy it into the current working directory and use its path.")
        if does_file_contain_string(result.path_to_inference_file, "train.csv") or does_file_contain_string(result.path_to_inference_file, "validation.csv"):
            raise ModelRetry("Inference file contains references to dataset split files ('train.csv' or 'validation.csv' detected), which will not be accessible during final testing.")
        #TODO improve validation with info about the artifacts-dir
        run_inference_and_log(config, iteration=-1, evaluation_stage='dry_run')
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result

    return inference_agent    