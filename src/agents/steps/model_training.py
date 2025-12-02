import subprocess
import traceback
import shutil
import warnings

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import ModelRetry
import pandas as pd

class ModelTraining(BaseModel):
    path_to_train_file: str = Field(
        description="Absolute path to the generated 'train.py'"
    )
    path_to_model_file: str = Field(
        description="Absolute path to the trained model file"
    )
    path_to_artifacts_dir: str = Field(
        description="Absolute path to the folder with artifacts produced by training. Must be called 'training_artifacts'. (This folder should be the parent of path_to_model_file and a sibling to train.py)"
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

        # Sample training data
        train_subset = get_dataset_subset(train_data_path, target_col, config.task_type)
        train_subset.to_csv(temp_train_path, index=False)

        # Sample validation data
        valid_subset = get_dataset_subset(valid_data_path, target_col, config.task_type)
        valid_subset.to_csv(temp_valid_path, index=False)

        # Run training script on subset
        command = f"{command_prefix} python \"{train_script_path}\" --train-data \"{temp_train_path}\" --validation-data \"{temp_valid_path}\" --artifacts-dir \"{temp_artifacts_dir}\""
        training_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True)
        if(training_out.returncode != 0):
            raise ModelRetry(f"Training script validaiton failed: Return code: {training_out.returncode}\nStderr: {training_out.stderr}, Stdout: {training_out.stdout}")
        
        # Check if model file was created
        expected_model_path = temp_artifacts_dir / model_file_name
        if not expected_model_path.exists():
            # List all files that were actually created
            created_files = list(temp_artifacts_dir.iterdir())
            created_files_names = [f.name for f in created_files]
            error_msg = f"Training script validation failed: After running the training script, model file '{model_file_name}' was not created in the specified artifacts folder ({temp_artifacts_dir}). "
            error_msg += f"Return code: {training_out.returncode}. "
            if created_files_names:
                error_msg += f"Files found in artifacts folder: {created_files_names}. "
            else:
                error_msg += f"No files were created in the artifacts folder. "
            error_msg += f"Stderr: {training_out.stderr.decode() if isinstance(training_out.stderr, bytes) else training_out.stderr} "
            error_msg += f"Stdout: {training_out.stdout.decode() if isinstance(training_out.stdout, bytes) else training_out.stdout}"
            raise ModelRetry(error_msg)
        print('TRAINING REPRODUCIBILITY OK')

        # Log all files created in artifacts dir before cleanup
        created_files = list(temp_artifacts_dir.iterdir())
        created_files_names = [f.name for f in created_files]
        return created_files_names

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

def get_dataset_subset(data_path, target_col, task_type):
    df = pd.read_csv(data_path)
    clf_per_label_samples = 100
    reg_samples = 1000

    if task_type == 'classification':
        # For classification: balance samples per label
        # Suppress FutureWarning about groupby.apply operating on grouping columns
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            try:
                # Try with include_groups parameter (pandas >= 2.2.0)
                subset = df.groupby(target_col, group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), clf_per_label_samples), random_state=42),
                    include_groups=False
                ).reset_index(drop=True)
            except TypeError:
                # Fall back to old behavior if include_groups is not available
                subset = df.groupby(target_col, group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), clf_per_label_samples), random_state=42)
                ).reset_index(drop=True)
    elif task_type == 'regression':
        # For regression: random sample from entire dataset
        total_samples = min(len(df), reg_samples)
        subset = df.sample(n=total_samples, random_state=42).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown task type: {task_type}. Supported types are 'classification' and 'regression'.")

    return subset


def get_model_training_prompt(user_prompt: str = None, is_fork: bool = False):
    """
    Prompt for implementing and running model training.

    Supports an optional user_prompt (including when forking) that is
    appended as explicit user instructions.
    """
    base_prompt = """
    Your next task: implement any necessary code for training a model. Then train a single model.
    The train script should save any files necessary to use the trained model for predictions (e.g. model file, tokenizers, ...).
    If your model can be accelerated by GPU, implement the code to use GPU.
    """
    
    if user_prompt:
        return f"""
    {base_prompt.strip()}
    
    User instructions: {user_prompt}
    """
    
    return base_prompt
