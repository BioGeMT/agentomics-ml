from __future__ import annotations

from agents.steps.base import RuntimeStep
from agents.steps.data_exploration import DataExplorationStep
from agents.steps.data_representation import DataRepresentationStep
from agents.steps.data_split import DataSplitStep
from agents.steps.iteration_plan import IterationPlanStep
from agents.steps.model_architecture import ModelArchitectureStep
from agents.steps.model_inference import ModelInferenceStep
from agents.steps.model_training import ModelTrainingStep
from agents.steps.prediction_exploration import PredictionExplorationStep
from agents.steps.validation_evaluation import ValidationEvaluationStep
from utils.config import Config


STEP_CLASS_REGISTRY: dict[str, type[RuntimeStep]] = {
    step_cls.step_id: step_cls
    for step_cls in {
        IterationPlanStep,
        DataExplorationStep,
        DataSplitStep,
        DataRepresentationStep,
        ModelArchitectureStep,
        ModelTrainingStep,
        ModelInferenceStep,
        PredictionExplorationStep,
        ValidationEvaluationStep,
    }
}

def get_step_sequence(config: Config) -> list[type[RuntimeStep]]:
    return [get_step_class(str(step_id)) for step_id in config.step_sequence]

def get_step_class(step_id: str) -> type[RuntimeStep]:
    return STEP_CLASS_REGISTRY[step_id]
