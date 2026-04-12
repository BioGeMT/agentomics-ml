from __future__ import annotations

from pydantic import Field
from pydantic_ai import Agent

from agents.steps.base import AgenticStep, AgenticStepOutput
from agents.steps.data_split import DataSplitStep
from agents.steps.model_inference import ModelInferenceStep
from runtime.step_outputs import require_step_output


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
        model_inference = require_step_output(self.config, ModelInferenceStep.step_id, current_iteration_dir)
        validation_path = data_split.val_path
        inference_path = model_inference.path_to_inference_file
        return f"""
            Your next task: Generate predictions on the validation set ({validation_path}) and identify where those predictions succeed, fail, and prediction biases.
            You can use but not modify the inference script ({inference_path}). If you need to write code for prediction generation and/or analysis, create a separate script.
            """
    
    def build_deps(self, step_started_at) -> dict[str, object]:
        return {"start_time": step_started_at}
