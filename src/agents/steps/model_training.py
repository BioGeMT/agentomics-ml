from __future__ import annotations

import os
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext

import agents.injectables.training_reporter as _training_reporter_module
from agents.steps.base import AgenticStep, AgenticStepOutput
from agents.steps.data_split import DataSplitStep
from runtime.conda_utils import get_shared_environment_path
from runtime.read_write_utils import does_file_contain_iteration_pattern
from runtime.step_outputs import require_step_output
from runtime.system_resources import check_gpu_availability
from datasets.dataset_utils import get_numeric_label_col_from_prepared_dataset
from utils.text_processing_utils import collapse_repeated_lines, concise_output


class ModelTrainingOutput(AgenticStepOutput):
    path_to_train_file: str = Field(description="Absolute path to the generated 'train.py'")
    path_to_model_file: str = Field(description="Absolute path to the trained model file")
    path_to_artifacts_dir: str = Field(
        description=(
            "Absolute path to the folder with artifacts produced by training. "
            "Must be called 'training_artifacts'. "
            "(This folder should be the parent of path_to_model_file and a sibling to train.py)"
        )
    )
    training_summary: str = Field(
        description="Short summary of the training implementation. Don't include any metrics in this summary."
    )
    unresolved_issues: str | None = Field(
        description=(
            "Issues that remain unresolved and could impact performance and/or metrics. "
            "(e.g. expected GPU to be available but is inaccessible during training, "
            "foundation model could not be loaded, etc...). Can be empty."
        )
    )

class ModelTrainingStep(AgenticStep):
    step_id = "model_training"
    display_name = "TRAINING"
    output_type = ModelTrainingOutput

    def injected_scripts(self) -> list[Path]:
        if self.config.disable_training_reporting:
            return []
        return [Path(_training_reporter_module.__file__)]

    def attach_output_validator(self, agent: Agent[dict, AgenticStepOutput]) -> None:
        @agent.output_validator
        async def validate_training(ctx: RunContext[dict], result: ModelTrainingOutput) -> ModelTrainingOutput:
            if not os.path.exists(result.path_to_train_file):
                raise ModelRetry(f"Train file does not exist. {result.path_to_train_file}")
            if Path(result.path_to_train_file).name.strip() != "train.py":
                raise ModelRetry(f"Train file must be called 'train.py' , currently is named {Path(result.path_to_train_file).name.strip()}")
            if os.path.islink(result.path_to_train_file):
                raise ModelRetry(f"Train file ({result.path_to_train_file}) cannot be a symbolic link, create a non-symlinked copy of it.")
            if not os.path.exists(result.path_to_model_file):
                raise ModelRetry(f"Model file does not exist at {result.path_to_model_file}")
            workspace_dir = self.config.current_step_dir
            workspace_root = workspace_dir.resolve(strict=False)
            if Path(result.path_to_train_file).resolve().parent != workspace_root:
                raise ModelRetry(f"Train file must be created directly in {workspace_dir} as train.py.")
            for candidate_path in [result.path_to_train_file, result.path_to_artifacts_dir, result.path_to_model_file]:
                if not Path(candidate_path).expanduser().resolve(strict=False).is_relative_to(workspace_root):
                    raise ModelRetry(f"{candidate_path} must stay inside {workspace_dir}.")
            if Path(result.path_to_artifacts_dir).name.strip() != "training_artifacts":
                raise ModelRetry(f"The artifacts folder produced by training must be called 'training_artifacts', currently is named {Path(result.path_to_artifacts_dir).name.strip()}")
            if (Path(result.path_to_train_file).resolve().parent / "training_artifacts").resolve() != Path(result.path_to_artifacts_dir).resolve():
                raise ModelRetry("The artifacts folder produced by training must be a sibling to train.py.")
            if Path(result.path_to_artifacts_dir).resolve() not in Path(result.path_to_model_file).resolve().parents:
                raise ModelRetry(f"Model file ({result.path_to_model_file}) must be inside the artifacts folder ({result.path_to_artifacts_dir})")
            if does_file_contain_iteration_pattern(result.path_to_train_file):
                raise ModelRetry(f"Train file ({result.path_to_train_file}) contains path containing a forbidden string 'iteration_' or references an iteration folder, which will not accessible during final testing. If you want to re-use a file from a past iteration, copy it into the current working directory and use its path.")
            created_files_names = self._validate_training_run(
                train_data_path=ctx.deps["train_csv_path"],
                valid_data_path=ctx.deps["validation_csv_path"],
                train_script_path=result.path_to_train_file,
                model_file_name=Path(result.path_to_model_file).name,
            )
            existing_files_names = [path.name for path in Path(result.path_to_artifacts_dir).iterdir()]

            if set(created_files_names) != set(existing_files_names):
                extras_in_submitted_folder = set(existing_files_names) - set(created_files_names)
                extras_in_retrain_folder = set(created_files_names) - set(existing_files_names)
                if extras_in_submitted_folder:
                    error_msg = "Artifacts directory contains extra files, probably from a previous failed training attempt.\n"
                    error_msg += f"Files created using the current training script: {created_files_names}\n"
                    error_msg += f"Files existing in artifacts directory: {existing_files_names}\n"
                    error_msg += f"Extra files that should be cleaned up: {list(extras_in_submitted_folder)}\n"
                    error_msg += f"Please clean up the artifacts directory at {result.path_to_artifacts_dir} and try again."
                    raise ModelRetry(error_msg)
                print(
                    "Warning: Training script creates some extra files compared to the submitted training artifacts: "
                    f"{extras_in_retrain_folder}"
                )

            return result

    def step_prompt(self) -> str:
        reporting_requirement = "" if self.config.disable_training_reporting else f"""
        - Always report training status using the helpers package:
          from helpers.training_reporter import TrainingReporter
          reporter = TrainingReporter()
        - Use reporter.report_epoch(...) whenever your chosen training API naturally exposes epoch summaries or epoch callbacks.
        - You may also use reporter.report_batch(...) for long epochs when your chosen training API already exposes a true batch loop or batch callback and the extra reporting is cheap. Do not invent pseudo-batches or rewrite the training approach just to force batch-level reporting.
        - When you report a validation metric, report the run's main validation metric: {self.config.val_metric}.
        - If your chosen library exposes useful callbacks for progress reporting, prefer using those callbacks instead of ad-hoc print statements.
        - If the chosen training API exposes no real epoch or batch progress hooks, call reporter.report_unavailable("...") once with a concrete reason instead of inventing fake progress.
        - If you implement early stopping and know how many epochs remain before stopping, include early_stopping_patience_remaining in reporter.report_epoch(...).
"""
        return f"""
        Your next task: implement training code and train your model.
        Training guidelines:
        - Train until validation performance stops improving, and output the best checkpoint.
        - Save all artifacts needed for inference (model file, tokenizers, etc...).
        - If you failed to implement your intended model, when you call the final_result tool, put into unresolved issues what went wrong.
        {"- If your model can be accelerated by GPU, implement the code to use GPU." if check_gpu_availability() else ""}
        {reporting_requirement}
        - Save the training script directly as train.py in the current step directory, not inside a nested folder.
        The train script should take the following parameters
        --train-data (a path to the training data csv)
        --validation-data (a path to the validation data csv. For example for the purposes of early-stopping. If the training script doesn't need validation data, still include the argument for compatibility and don't use it.)
        --artifacts-dir (path to a directory that will be populated by the training script with artifacts needed to use the trained model for predictions (e.g. produced model weights, produced tokenizers, ...). This directory should not contain any other external sources like imported scripts, conda packages, foundation models, etc..)
        The script must not accept any other parameters.
        """

    def build_deps(self, step_started_at: datetime) -> dict[str, object]:
        data_split = require_step_output(self.config, DataSplitStep.step_id, self.config.current_iteration_dir)
        return {
            "start_time": step_started_at,
            "train_csv_path": data_split.train_path,
            "validation_csv_path": data_split.val_path,
        }

    def _validate_training_run(self, train_data_path: str, valid_data_path: str, train_script_path: str, model_file_name: str) -> list[str]:
        run_dir = self.config.current_iteration_dir
        conda_path = get_shared_environment_path(self.config)
        command_prefix = f"cd {run_dir} && conda run -p {conda_path}"

        temp_artifacts_dir = run_dir / "temp_retrain_artifacts"
        temp_train_path = run_dir / "temp_train_subset.csv"
        temp_valid_path = run_dir / "temp_valid_subset.csv"

        try:
            target_col = get_numeric_label_col_from_prepared_dataset(self.config.prepared_dataset_dir)
            train_subset = self._get_dataset_subset(train_data_path, target_col)
            train_subset.to_csv(temp_train_path, index=False)
            valid_subset = self._get_dataset_subset(valid_data_path, target_col)
            valid_subset.to_csv(temp_valid_path, index=False)

            temp_artifacts_dir.mkdir(parents=True, exist_ok=True)
            command = (
                f"{command_prefix} python \"{train_script_path}\" "
                f"--train-data \"{temp_train_path}\" "
                f"--validation-data \"{temp_valid_path}\" "
                f"--artifacts-dir \"{temp_artifacts_dir}\""
            )
            training_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True)
            if training_out.returncode != 0:
                raise ModelRetry(
                    self._build_training_retry_message(
                        "Training script validation failed",
                        returncode=training_out.returncode,
                        stdout=training_out.stdout,
                        stderr=training_out.stderr,
                    )
                )

            expected_model_path = temp_artifacts_dir / model_file_name
            if not expected_model_path.exists():
                raise ModelRetry(
                    self._build_training_retry_message(
                        f"Training script validation failed: model file '{model_file_name}' was not created in the specified artifacts folder.",
                        returncode=training_out.returncode,
                        stdout=training_out.stdout,
                        stderr=training_out.stderr,
                    )
                )
            training_outputs = (
                training_out.stdout.decode("utf-8", errors="replace"),
                training_out.stderr.decode("utf-8", errors="replace"),
            )
            emitted_training_report = any(
                line.startswith("TRAINING_REPORT:")
                for output in training_outputs
                for line in output.splitlines()
            )
            if not self.config.disable_training_reporting and not emitted_training_report:
                raise ModelRetry(
                    self._build_training_retry_message(
                        "Training script validation failed: training script did not emit any TRAINING_REPORT lines during validation. "
                        "Use TrainingReporter to emit at least one report event, or call reporter.report_unavailable(...) when no intermediate progress is available.",
                        returncode=training_out.returncode,
                        stdout=training_out.stdout,
                        stderr=training_out.stderr,
                    )
                )
            return [path.name for path in temp_artifacts_dir.iterdir()]
        except Exception as error:
            if isinstance(error, ModelRetry):
                raise
            traceback_msg = concise_output(collapse_repeated_lines(traceback.format_exc()))
            raise ModelRetry(f"Training script validation failed: {traceback_msg}") from error
        finally:
            for temporary_path in [temp_train_path, temp_valid_path]:
                if temporary_path.exists():
                    temporary_path.unlink()
            if temp_artifacts_dir.exists():
                shutil.rmtree(temp_artifacts_dir)

    def _get_dataset_subset(self, data_path: str, target_col: str) -> pd.DataFrame:
        dataframe = pd.read_csv(data_path)
        if self.config.task_type == "classification":
            samples_per_label = 100
            return dataframe.groupby(target_col, group_keys=False).apply(
                lambda frame: frame.sample(n=min(len(frame), samples_per_label), random_state=42)
            ).reset_index(drop=True)
        if self.config.task_type == "regression":
            total_samples = min(len(dataframe), 1000)
            return dataframe.sample(n=total_samples, random_state=42).reset_index(drop=True)
        raise ValueError(
            f"Unknown task type: {self.config.task_type}. Supported types are 'classification' and 'regression'."
        )

    def _build_training_retry_message(self, prefix: str, returncode: int, stdout: bytes | str, stderr: bytes | str) -> str:
        message = f"{prefix}: Return code: {returncode}\nStderr: {stderr}, Stdout: {stdout}"
        return concise_output(collapse_repeated_lines(message))
