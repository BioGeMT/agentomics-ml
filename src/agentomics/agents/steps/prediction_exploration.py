from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_ai import Agent

from agentomics.agents.steps.base import AgenticStep, AgenticStepOutput
from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.validation_evaluation import ValidationEvaluationStep
from agentomics.datasets.data_contract import LABELS_FILE_NAME
from agentomics.runtime.step_outputs import require_step_output
from agentomics.utils.exceptions import IterationRunFailed


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

    async def run(self) -> None:
        try:
            await super().run()
        except IterationRunFailed:
            # Validation Evaluation already scored the model, so keep the iteration without the analysis.
            print("Prediction exploration failed, keeping the iteration")
            self.on_step_success(PredictionExplorationOutput(
                statistics="Not available: prediction exploration failed in this iteration.",
                insights="Not available: prediction exploration failed in this iteration.",
            ))

    def step_prompt(self) -> str:
        #TODO all requires should be deps?
        current_iteration_dir = self.config.current_iteration_dir
        data_split = require_step_output(self.config, DataSplitStep.step_id, current_iteration_dir)
        validation_path = data_split.val_path
        labels_path = Path(validation_path) / LABELS_FILE_NAME
        predictions_path = (
            self.config.archived_step_dir(ValidationEvaluationStep.step_id) / "eval_predictions_validation.csv"
        )
        return f"""
            Your next task: Identify where the model's predictions on the validation set ({validation_path}) succeed, fail, and prediction biases.
            The predictions of the inference script on the validation set are already in {predictions_path}, and the validation labels are in {labels_path}.
            Analyze these predictions; do not run inference again. To analyze a subset of the validation set, filter these predictions.
            If you need to write code for the analysis, create a separate script.
            """
    
    def build_deps(self, step_started_at) -> dict[str, object]:
        return {"start_time": step_started_at}
