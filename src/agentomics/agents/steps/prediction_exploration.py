from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_ai import Agent

from agentomics.agents.steps.base import AgenticStep, AgenticStepOutput
from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.validation_evaluation import ValidationEvaluationStep
from agentomics.datasets.conditional_generation_contract import CONTROLS_FILE_NAME
from agentomics.datasets.data_contract import (
    GENERATION_LABELS_FILE_NAME,
    INPUT_DIR_NAME,
    LABELS_FILE_NAME,
    predictions_file_suffix,
)
from agentomics.utils.task_types import TaskTypes
from agentomics.runtime.step_outputs import require_step_output


class PredictionExplorationOutput(AgenticStepOutput):
    statistics: str = Field(
        description="""
        Statistics that provide insight into the successes, fails, and biases of the model predictions of the validation set.
        """
    )
    insights: str = Field(
        description="""
        Insights about validation set predictions that are useful for future modeling attempts.
        Don't provide concrete implementation recommendations for improvement.
        """
    )

class PredictionExplorationStep(AgenticStep):
    step_id = "prediction_exploration"
    display_name = "PREDICTION EXPLORATION"
    output_type = PredictionExplorationOutput

    def step_prompt(self) -> str:
        #TODO all requires should be deps?
        current_iteration_dir = self.config.current_iteration_dir
        data_split = require_step_output(self.config, DataSplitStep.step_id, current_iteration_dir)
        validation_path = data_split.val_path
        is_generation = self.config.task_type == TaskTypes.CONDITIONAL_GENERATION
        labels_path = Path(validation_path) / (GENERATION_LABELS_FILE_NAME if is_generation else LABELS_FILE_NAME)
        predictions_path = (
            self.config.archived_step_dir(ValidationEvaluationStep.step_id)
            / f"eval_predictions_validation{predictions_file_suffix(self.config.task_type)}"
        )
        controls_note = (
            f" The control cells the predicted effects are measured against are in {Path(validation_path) / INPUT_DIR_NAME / CONTROLS_FILE_NAME}."
            if is_generation
            else ""
        )
        return f"""
            Your next task: Identify where the model's predictions on the validation set ({validation_path}) succeed, fail, and prediction biases.
            The predictions of the inference script on the validation set are already in {predictions_path}, and the validation labels are in {labels_path}.{controls_note}
            Analyze these predictions; do not run inference again. To analyze a subset of the validation set, filter these predictions.
            If you need to write code for the analysis, create a separate script.
            """
    
    def build_deps(self, step_started_at) -> dict[str, object]:
        return {"start_time": step_started_at}
