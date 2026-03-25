import subprocess
import traceback
import shutil
import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import ModelRetry, Agent, RunContext
import pandas as pd

from agentomics.utils.text_processing_utils import concise_output, collapse_repeated_lines
from agentomics.agents.agent_utils import get_new_rundir_files, does_file_contain_iteration_pattern

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
    The script must not accept any other parameters.
    """

def create_model_training_agent(config, model, tools):
    training_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=ModelTraining,
        retries=config.max_validation_retries,
        deps_type=dict,
    )
        
    @training_agent.output_validator
    async def validate_training(ctx: RunContext[dict], result: ModelTraining) -> ModelTraining:
        if not os.path.exists(result.path_to_train_file):
            raise ModelRetry(f"Train file does not exist. {result.path_to_train_file}")
        if not Path(result.path_to_train_file).name.strip() == 'train.py':
            raise ModelRetry(f"Train file must be called 'train.py' , currently is named {Path(result.path_to_train_file).name.strip()}")
        if os.path.islink(result.path_to_train_file):
            raise ModelRetry(f"Train file ({result.path_to_train_file}) cannot be a symbolic link, create a non-symlinked copy of it.")
        if not os.path.exists(result.path_to_model_file):
            raise ModelRetry(f"Model file does not exist at {result.path_to_model_file}")
        if ctx.deps['run_dir'] not in Path(result.path_to_artifacts_dir).parents:
            raise ModelRetry(f"path_to_artifacts_dir ({result.path_to_artifacts_dir}) must be a child of your run dir ({ctx.deps['run_dir']})")
        if Path(result.path_to_artifacts_dir).name.strip() != 'training_artifacts':
            raise ModelRetry(f"The artifacts folder produced by training must be called 'training_artifacts', currently is named {Path(result.path_to_artifacts_dir).name.strip()}")
        if (Path(result.path_to_train_file).resolve().parent / 'training_artifacts').resolve() != Path(result.path_to_artifacts_dir).resolve():
            raise ModelRetry(f"The artifacts folder produced by training must be a sibling to train.py.")
        if Path(result.path_to_artifacts_dir).resolve() not in Path(result.path_to_model_file).parents:
            raise ModelRetry(f"Model file ({result.path_to_model_file}) must be inside the artifacts folder ({result.path_to_artifacts_dir})")
        if does_file_contain_iteration_pattern(result.path_to_train_file):
            raise ModelRetry(f"Train file ({result.path_to_train_file}) contains path containing a forbidden string 'iteration_' or references an iteration folder, which will not accessible during final testing. If you want to re-use a file from a past iteration, copy it into the current working directory and use its path.")
        created_files_names = retrain_and_check(
            config=config,
            train_data_path=ctx.deps['train_csv_path'],
            valid_data_path=ctx.deps['validation_csv_path'],
            train_script_path = result.path_to_train_file,
            model_file_name = Path(result.path_to_model_file).name,
        )
        existing_files = list(Path(result.path_to_artifacts_dir).iterdir())
        existing_files_names = [f.name for f in existing_files]

        # Check if created files match existing files in artifacts directory
        if set(created_files_names) != set(existing_files_names):
            extras_in_submitted_folder = set(existing_files_names) - set(created_files_names)
            extras_in_retrain_folder = set(created_files_names) - set(existing_files_names)
            if(len(extras_in_submitted_folder) > 0):
                error_msg = f"Artifacts directory contains extra files, probably from a previous failed training attempt.\n"
                error_msg += f"Files created using the current training script: {created_files_names}\n"
                error_msg += f"Files existing in artifacts directory: {existing_files_names}\n"
                error_msg += f"Extra files that should be cleaned up: {list(extras_in_submitted_folder)}\n"
                error_msg += f"Please clean up the artifacts directory at {result.path_to_artifacts_dir} and try again."
                raise ModelRetry(error_msg)
            else:
                print(f"Warning: Training script creates some extra files compared to the submitted training artifacts: {extras_in_retrain_folder}")

        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result

    return training_agent

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
            message = f"Training script validaiton failed: Return code: {training_out.returncode}\nStderr: {training_out.stderr}, Stdout: {training_out.stdout}"
            message = collapse_repeated_lines(message)
            message = concise_output(message)
            raise ModelRetry(message)
        
        # Check if model file was created
        expected_model_path = temp_artifacts_dir / model_file_name
        if not expected_model_path.exists():
            error_msg = f"Training script validation failed: After running the training script, model file '{model_file_name}' was not created in the specified artifacts folder. "
            error_msg += f"Return code: {training_out.returncode}. "
            error_msg += f"Stderr: {training_out.stderr}"
            error_msg += f"Stdout: {training_out.stdout}"
            error_msg = collapse_repeated_lines(error_msg)
            error_msg = concise_output(error_msg)
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
        traceback_msg = collapse_repeated_lines(traceback_msg)
        traceback_msg = concise_output(traceback_msg)
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