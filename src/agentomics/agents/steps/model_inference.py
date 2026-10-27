from __future__ import annotations

import os
import traceback
from pathlib import Path

import pandas as pd
from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from agentomics.datasets.conditional_generation_contract import (
    CONTEXT_COLUMN_NAME,
    CONTROL_LABEL,
    CONTROLS_FILE_NAME,
    MAX_COUNTS_PER_CELL,
    N_CELLS_COLUMN_NAME,
    PERTURBATIONS_FILE_NAME,
    TARGET_GENE_COLUMN_NAME,
    validate_generated_cells,
)
from agentomics.datasets.data_contract import (
    GENERATION_LABELS_FILE_NAME,
    ID_COLUMN_NAME,
    INPUT_DIR_NAME,
    LABELS_FILE_NAME,
    MINI_TRAIN_SPLIT,
    NUMERIC_LABEL_COLUMN_NAME,
    PREDICTION_COLUMN_NAME,
    SUPPLEMENTARY_DIR_NAME,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    labels_file_name,
    predictions_file_suffix,
)

from agentomics.agents.steps.base import AgenticStep, AgenticStepOutput
from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.model_training import ModelTrainingStep
from agentomics.runtime.filesystem import remove_path
from agentomics.runtime.inference_runner import compute_metrics, run_inference_on_split
from agentomics.runtime.read_write_utils import (
    does_file_contain_iteration_pattern,
    does_file_contain_string,
    load_dataset_metadata,
)
from agentomics.runtime.step_outputs import require_step_output
from agentomics.utils.task_types import TaskTypes
from agentomics.utils.text_processing_utils import collapse_repeated_lines, concise_output


class ModelInferenceOutput(AgenticStepOutput):
    path_to_inference_file: str = Field(description="Absolute path to the generated inference.py")
    inference_summary: str = Field(description="Short summary of the inference implementation")
    unresolved_issues: str | None = Field(
        description=(
            "Issues that remain unresolved and could impact performance and/or metrics. "
            "(e.g. expected GPU to be available but is inaccessible during inference, "
            "foundation model could not be loaded, etc...). Can be empty."
        )
    )

class ModelInferenceStep(AgenticStep):
    step_id = "model_inference"
    display_name = "INFERENCE"
    output_type = ModelInferenceOutput

    def attach_output_validator(self, agent: Agent[dict, AgenticStepOutput]) -> None:
        @agent.output_validator
        async def validate_inference(ctx: RunContext[dict], result: ModelInferenceOutput) -> ModelInferenceOutput:
            self._validate_inference_file(result.path_to_inference_file)
            self._validate_dry_run()
            return result

    def step_prompt(self) -> str:
        model_training = require_step_output(self.config, ModelTrainingStep.step_id, self.config.current_iteration_dir)
        if self.config.task_type == TaskTypes.CLASSIFICATION:
            probability_columns = "\n".join(
                f"- 'probability_{class_id}': probability for class {class_id} (float)"
                for class_id in sorted(load_dataset_metadata(self.config)["label_to_scalar"].values())
            )
            output_file_description = f"A csv with columns:\n- 'id': the input sample id\n- 'prediction': the predicted class (int)\n{probability_columns}"
        elif self.config.task_type == TaskTypes.REGRESSION:
            output_file_description = "A csv with columns 'id' and 'prediction', where 'prediction' contains the predicted continuous value."
        elif self.config.task_type == TaskTypes.CONDITIONAL_GENERATION:
            output_file_description = self._generated_cells_description()
        else:
            raise ValueError(f"Unknown task type: {self.config.task_type}. Supported types are {TaskTypes}.")

        return f"""
        Your next task: create inference.py file.
        If your model can be accelerated by GPU, implement the code to use GPU.
        {self._coverage_requirement()}
        The inference script must use the same architecture as your current trained model from 'train.py' and use the artifacts produced by that script (located at '{model_training.path_to_artifacts_dir}').
        The inference script will be taking the following named arguments:
        --input (a split input folder path). This folder has the same structure as train/input and contains no labels.
        --output (the output file path). {output_file_description}
        --artifacts-dir (the folder that contains training artifacts from the training step that are needed to run inference (for example model weights, tokenizers, etc..). The following dir should be used as a default: '{model_training.path_to_artifacts_dir}'. If a different path is provided, your script must adapt to the new source. You can assume the artifact files will always have the same name.
        The script must not accept any other parameters.
        """

    def _coverage_requirement(self) -> str:
        if self.config.task_type == TaskTypes.CONDITIONAL_GENERATION:
            return (
                f"The inference script must generate cells for every condition listed in {INPUT_DIR_NAME}/{PERTURBATIONS_FILE_NAME}. "
                "Don't skip any conditions."
            )
        return (
            "The inference script must produce a prediction for every single input sample. Don't skip any samples. "
            "The 'id' values from the input folder data must be preserved in the output file."
        )

    @staticmethod
    def _generated_cells_description() -> str:
        return f"""An h5ad file (AnnData) of generated cells:
        - For every row of {INPUT_DIR_NAME}/{PERTURBATIONS_FILE_NAME}, exactly '{N_CELLS_COLUMN_NAME}' generated cells (one row each), with obs columns '{CONTEXT_COLUMN_NAME}' and '{TARGET_GENE_COLUMN_NAME}' set to that row's values.
        - No control ('{CONTROL_LABEL}') cells.
        - Genes (var_names): those of {INPUT_DIR_NAME}/{CONTROLS_FILE_NAME}, in the same order.
        - X: finite, non-negative whole counts, at most {MAX_COUNTS_PER_CELL} in total per cell. Store X as a scipy sparse matrix. Predictions must not contain NaN or infinity.
        - Input controls may contain NaN for unavailable measurements. Handle them consistently with training and generate predictions for the full input gene panel; do not copy missing values into the output or omit those genes.
        - Sample each cell from the predicted distribution of its condition, so the cells of one condition vary like real cells do. Repeating one profile for all cells of a condition removes that variation and breaks the differential-expression based metrics."""

    def _validate_inference_file(self, inference_file_path: str) -> None:
        self._validate_inference_file_exists(inference_file_path)
        self._validate_inference_file_location(inference_file_path)
        self._validate_inference_file_references(inference_file_path)

    def _validate_inference_file_exists(self, inference_file_path: str) -> None:
        if not os.path.exists(inference_file_path):
            raise ModelRetry(f"Inference file does not exist at {inference_file_path}")
        if os.path.islink(inference_file_path):
            raise ModelRetry(
                f"Inference file ({inference_file_path}) cannot be a symbolic link, create a non-symlinked copy of it."
            )

    def _validate_inference_file_location(self, inference_file_path: str) -> None:
        workspace_dir = self.config.current_step_dir
        workspace_root = workspace_dir.resolve(strict=False)
        candidate_path = Path(inference_file_path).expanduser().resolve(strict=False)
        if not candidate_path.is_relative_to(workspace_root):
            raise ModelRetry(f"{inference_file_path} must stay inside {workspace_dir}.")

    def _validate_inference_file_references(self, inference_file_path: str) -> None:
        if does_file_contain_iteration_pattern(inference_file_path):
            raise ModelRetry(
                "Inference file contains path containing a forbidden string 'iteration_' or references an "
                "iteration folder, which will not accessible during final testing. If you want to re-use "
                "a file from a past iteration, copy it into the current working directory and use its path."
            )
        for labels_file_name in (LABELS_FILE_NAME, GENERATION_LABELS_FILE_NAME):
            if does_file_contain_string(inference_file_path, labels_file_name):
                raise ModelRetry(
                    f"Inference file contains references to {labels_file_name}, which will not be accessible during final testing."
                )
        dataset_supplementary_path = self.config.dataset_dir / SUPPLEMENTARY_DIR_NAME
        if dataset_supplementary_path.is_dir() and does_file_contain_string(
            inference_file_path, SUPPLEMENTARY_DIR_NAME + "/"
        ):
            raise ModelRetry(
                "Inference file references supplementary/. "
                "The inference script receives only the input/ folder and must not depend on supplementary/."
            )
        data_split = require_step_output(self.config, DataSplitStep.step_id, self.config.current_iteration_dir)
        for split_name, split_path in [
            (TRAIN_SPLIT, data_split.train_path),
            (VALIDATION_SPLIT, data_split.val_path),
            (MINI_TRAIN_SPLIT, data_split.mini_train_path),
        ]:
            if does_file_contain_string(inference_file_path, split_path):
                raise ModelRetry(
                    f"Inference file references the {split_name} split path. "
                    "The inference script must not depend on any split data — it receives only --input and --artifacts-dir at runtime."
                )

    def _validate_dry_run(self) -> None:
        dataset_metadata = load_dataset_metadata(self.config)
        model_training = require_step_output(self.config, ModelTrainingStep.step_id, self.config.current_iteration_dir)
        data_split = require_step_output(self.config, DataSplitStep.step_id, self.config.current_iteration_dir)
        evaluation_stage = "dry_run"
        split_path = Path(data_split.mini_train_path)
        labels_path = split_path / labels_file_name(self.config.task_type)
        output_path = self.config.current_step_dir / f"eval_predictions_dry_run{predictions_file_suffix(self.config.task_type)}"
        inference_result = run_inference_on_split(
            split_path=split_path,
            output_path=output_path,
            conda_env_path=self.config.shared_environment_path,
            inference_script_path=self.config.current_step_dir / "inference.py",
            training_artifacts_dir=Path(model_training.path_to_artifacts_dir),
        )
        if inference_result.returncode != 0:
            print("DRY RUN EVAL FAIL during inference:", inference_result.stderr)
            message = collapse_repeated_lines(f"Inference script validation failed: {str(inference_result)}")
            raise ModelRetry(concise_output(message))
        if self.config.task_type == TaskTypes.CONDITIONAL_GENERATION:
            self._validate_dry_run_generated_cells(
                input_path=split_path / INPUT_DIR_NAME,
                predictions_path=output_path,
                inference_result=inference_result,
            )
        else:
            self._validate_dry_run_predictions(
                labels_path=labels_path,
                predictions_path=output_path,
                inference_result=inference_result,
            )
        self._compute_dry_run_metrics(
            output_path=output_path,
            labels_path=labels_path,
            numeric_label_col=dataset_metadata.get("numeric_label_col"),
            task_type=dataset_metadata["task_type"],
            evaluation_stage=evaluation_stage,
        )
        print("DRY RUN EVAL SUCCESS")

    def _validate_dry_run_predictions(self, labels_path: Path, predictions_path: Path, inference_result) -> None:
        if not predictions_path.exists():
            inference_summary = concise_output(collapse_repeated_lines(str(inference_result)))
            raise ModelRetry(f"Inference doesn't produce predictions. Stderr/stdout: {inference_summary}")

        try:
            predictions_df = pd.read_csv(predictions_path, dtype=str, keep_default_na=False) #TODO better naming
        except Exception as error:
            traceback_summary = concise_output(collapse_repeated_lines(traceback.format_exc()))
            raise ModelRetry(f"Inference produced faulty predictions csv file. {error}\n{traceback_summary}") from error

        required_columns = {ID_COLUMN_NAME, PREDICTION_COLUMN_NAME}
        missing_columns = required_columns - set(predictions_df.columns)
        if missing_columns:
            raise ModelRetry(f"{predictions_path} is missing required columns: {sorted(missing_columns)}")

        ids = predictions_df[ID_COLUMN_NAME].str.strip()
        if (ids == "").any():
            raise ModelRetry(f"{predictions_path} has an empty id")

        duplicates = ids[ids.duplicated()].drop_duplicates().head(20).tolist()
        if duplicates:
            raise ModelRetry(
                f"{predictions_path} contains duplicate ids. "
                f"First (up to 20) duplicates: {duplicates}"
            )

        expected_ids = pd.read_csv(labels_path, dtype={ID_COLUMN_NAME: str}, keep_default_na=False)[ID_COLUMN_NAME].tolist()
        prediction_ids = set(ids.tolist())
        expected_id_set = set(expected_ids)
        
        missing = sorted(expected_id_set - prediction_ids)[:20]
        if missing:
            raise ModelRetry(
                "Inference script must produce a prediction for every labeled sample. "
                f"First (up to 20) label ids missing from predictions: {missing}"
            )

    def _validate_dry_run_generated_cells(self, input_path: Path, predictions_path: Path, inference_result) -> None:
        if not predictions_path.exists():
            inference_summary = concise_output(collapse_repeated_lines(str(inference_result)))
            raise ModelRetry(f"Inference doesn't produce generated cells. Stderr/stdout: {inference_summary}")
        try:
            validate_generated_cells(predictions_path, input_path)
        except ValueError as error:
            raise ModelRetry(f"Inference produced invalid generated cells: {error}") from error

    def _compute_dry_run_metrics(
        self,
        output_path: Path,
        labels_path: Path,
        numeric_label_col: str,
        task_type: str,
        evaluation_stage: str,
    ) -> None:
        try:
            compute_metrics(
                results_file=output_path,
                labels_path=labels_path,
                numeric_label_col=numeric_label_col,
                task_type=task_type,
                evaluation_stage=evaluation_stage,
            )
            # TODO should this be in finally?
            remove_path(output_path)
        except Exception:
            message = concise_output(collapse_repeated_lines(f"FAIL DURING DRY RUN METRICS COMPUTATION. {traceback.format_exc()}"))
            print(message)
            raise ModelRetry(message)
