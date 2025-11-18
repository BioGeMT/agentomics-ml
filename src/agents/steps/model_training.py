import subprocess
import traceback
import shutil

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import ModelRetry
import pandas as pd

class ModelTraining(BaseModel):
    path_to_train_file: str = Field(
        description="Absolute path to the generated training script Python file"
    )
    path_to_model_file: str = Field(
        description="Absolute path to the trained model file"
    )
    path_to_artifacts_dir: str = Field(
        description="Absolute path to the artifacts folder produced by training (this folder should be the parent of path_to_model_file)"
    )
    training_summary: str = Field(
        description="Short summary of the training implementation. Don't include any metrics in this summary."
    )
    unresolved_issues: str|None = Field(
        description="Issues that remain unresolved and could impact performance and/or metrics. (e.g. expected GPU to be available but is inaccessible during training, foundation model could not be loaded, etc...). Can be empty."
    )
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during model training step. Populated programmatically.
        """
    )

def get_model_training_prompt(config):
    return f"""
    Your next task: implement training code and train your model.
    Training guidelines:
    - Train until validation performance stops improving, and output the best checkpoint.
    - Save all artifacts needed for inference (model file, tokenizers, etc...).
    - If you failed to implement your intended model, when you call the final_output tool, put into unresolved issues what went wrong.
    {"If your model can be accelerated by GPU, implement the code to use GPU." if config.check_gpu_availability() else ""}

    The train script should take the following parameters
    --train-data (a path to the training data csv)
    --validation-data (a path to the validation data csv. For example for the purposes of early-stopping. If the training script doesn't need validation data, still include the argument for compatibility and don't use it.)
    --artifacts-dir (path to a directory that will be populated by the training script with artifacts needed to use the trained model for predictions (e.g. produced model weights, produced tokenizers, ...). This directory should not contain any other external sources like imported scripts, conda packages, foundation models, etc..)
    """

def retrain_and_check(config, train_data_path, valid_data_path, train_script_path, model_file_name):
    run_dir = config.runs_dir / config.agent_id
    conda_path = run_dir / ".conda" / "envs" / f"{config.agent_id}_env"
    command_prefix = f"cd {run_dir} && conda run -p {conda_path}"

    # Create temporary artifacts folder
    temp_artifacts_dir = run_dir / "temp_retrain_artifacts"
    temp_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary subset data files
    temp_train_path = run_dir / "temp_train_subset.csv"
    temp_valid_path = run_dir / "temp_valid_subset.csv"

    try:
        # Create balanced subsets of training and validation data
        target_col = config.get_numeric_label_col_name()
        samples_per_label = 100

        # Sample training data
        train_df = pd.read_csv(train_data_path)
        train_subset = train_df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), samples_per_label), random_state=42)
        ).reset_index(drop=True)
        train_subset.to_csv(temp_train_path, index=False)

        # Sample validation data
        valid_df = pd.read_csv(valid_data_path)
        valid_subset = valid_df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), samples_per_label), random_state=42)
        ).reset_index(drop=True)
        valid_subset = valid_df.head(samples_per_label * 2)
        valid_subset.to_csv(temp_valid_path, index=False)

        # Run training script on subset
        command = f"{command_prefix} python {train_script_path} --train-data {temp_train_path} --validation-data {temp_valid_path} --artifacts-dir {temp_artifacts_dir}"
        training_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True)
        if(training_out.returncode != 0):
            raise ModelRetry(f"Training script validaiton failed: Return code: {training_out.returncode}\nStderr: {training_out.stderr}, Stdout: {training_out.stdout}")
        
        # Check if model file was created
        expected_model_path = temp_artifacts_dir / model_file_name
        if not expected_model_path.exists():
            error_msg = f"Training script validation failed: After running the training script, model file '{model_file_name}' was not created in the specified artifacts folder. "
            error_msg += f"Return code: {training_out.returncode}. "
            error_msg += f"Stderr: {training_out.stderr}"
            error_msg += f"Stdout: {training_out.stdout}"
            raise ModelRetry(error_msg)
        print('TRAINING REPRODUCIBILITY OK')

    except Exception as e:
        if isinstance(e, ModelRetry):
            raise
        traceback_msg = traceback.format_exc()
        raise ModelRetry(f"Training script validation failed: {traceback_msg}")
    finally:
        # Clean up temporary files and folder
        if temp_train_path.exists():
            temp_train_path.unlink()
        if temp_valid_path.exists():
            temp_valid_path.unlink()
        if temp_artifacts_dir.exists():
            shutil.rmtree(temp_artifacts_dir)