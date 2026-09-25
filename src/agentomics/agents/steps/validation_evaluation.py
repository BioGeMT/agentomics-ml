from __future__ import annotations

import traceback
from pathlib import Path

from pydantic import BaseModel

from agentomics.agents.steps.base import RuntimeStep
from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.model_inference import ModelInferenceStep
from agentomics.agents.steps.model_training import ModelTrainingStep
from agentomics.run_logging.logging_helpers import log_iteration_metrics, log_new_best
from agentomics.runtime.inference_runner import compute_metrics, run_inference_on_split
from agentomics.runtime.read_write_utils import (
    load_best_iteration_snapshot_iteration,
    load_dataset_metadata,
)
from agentomics.runtime.step_outputs import load_step_output, require_step_output
from agentomics.utils.config import Config
from agentomics.utils.exceptions import AgentScriptFailed, IterationRunFailed
from agentomics.utils.metrics import get_higher_is_better_map
from agentomics.datasets.data_contract import LABELS_FILE_NAME


class ValidationEvaluationOutput(BaseModel):
    metrics: dict[str, float]
    is_new_best: bool
    status: str


class ValidationEvaluationStep(RuntimeStep):
    step_id = "validation_evaluation"
    display_name = "VALIDATION EVALUATION"
    output_type = ValidationEvaluationOutput

    @staticmethod
    def is_new_best_iteration(config: Config, new_metrics: dict[str, float]) -> bool:
        metric_name = config.val_metric
        validation_metric_key = f"validation/{metric_name}"
        if validation_metric_key not in new_metrics:
            return False

        best_iteration = load_best_iteration_snapshot_iteration(config)
        if best_iteration is None:
            return True

        current_split = require_step_output(
            config, DataSplitStep.step_id, config.current_iteration_dir
        )
        best_iteration_dir = config.iteration_dir(best_iteration)
        best_split = require_step_output(
            config, DataSplitStep.step_id, best_iteration_dir
        )
        if current_split.split_version != best_split.split_version:
            return True

        validation_output = load_step_output(
            config,
            ValidationEvaluationStep.step_id,
            iteration_dir=best_iteration_dir,
        )
        best_metrics = {} if validation_output is None else dict(validation_output.metrics)
        higher_is_better = get_higher_is_better_map()[metric_name]
        new_value = new_metrics[validation_metric_key]
        best_value = best_metrics[validation_metric_key]
        return new_value > best_value if higher_is_better else new_value < best_value

    def on_iteration_end(self, iteration: int) -> None:
        output = load_step_output(self.config, self.step_id, self.config.current_iteration_dir)
        if output is not None:
            log_iteration_metrics(
                metrics=output.metrics,
                iteration=iteration,
                task_type=self.config.task_type,
            )
            log_new_best(iteration=iteration, is_new_best=output.is_new_best)

    async def _execute(self) -> ValidationEvaluationOutput:
        new_metrics = self._run_inference_on_all_splits()
        current_iteration_is_new_best = self.is_new_best_iteration(self.config, new_metrics)
        return ValidationEvaluationOutput(
            metrics=new_metrics,
            is_new_best=current_iteration_is_new_best,
            status="success",
        )

    def _run_inference_on_all_splits(self) -> dict[str, float]:
        #TODO printouts remove?
        print("Starting evaluation phase")
        metrics: dict[str, float] = {}
        dataset_metadata = load_dataset_metadata(self.config)
        conda_env_path = self.config.shared_environment_path
        model_inference = require_step_output(self.config, ModelInferenceStep.step_id, self.config.current_iteration_dir)
        model_training = require_step_output(self.config, ModelTrainingStep.step_id, self.config.current_iteration_dir)
        inference_script_path = Path(model_inference.path_to_inference_file)
        training_artifacts_dir = Path(model_training.path_to_artifacts_dir)
        data_split = require_step_output(self.config, DataSplitStep.step_id, self.config.current_iteration_dir)
        for evaluation_stage in ["validation", "train"]:
            print(f"  Running {evaluation_stage} inference...")
            split_path = Path(data_split.val_path if evaluation_stage == "validation" else data_split.train_path)
            labels_path = split_path / LABELS_FILE_NAME
            output_path = self.config.current_step_dir / f"eval_predictions_{evaluation_stage}.csv"
            try:
                result = run_inference_on_split(
                    split_path=split_path,
                    output_path=output_path,
                    conda_env_path=conda_env_path,
                    inference_script_path=inference_script_path,
                    training_artifacts_dir=training_artifacts_dir,
                )
                if result.returncode != 0:
                    raise AgentScriptFailed(f"Inference on {evaluation_stage} failed: {str(result)}")
                evaluation_metrics = compute_metrics(
                    results_file=output_path,
                    labels_path=labels_path,
                    numeric_label_col=dataset_metadata["numeric_label_col"],
                    task_type=dataset_metadata["task_type"],
                    evaluation_stage=evaluation_stage,
                )
                metrics.update(
                    {
                        f"{evaluation_stage}/{metric_name}": metric_value
                        for metric_name, metric_value in evaluation_metrics.items()
                    }
                )
            except AgentScriptFailed:
                exception_trace = traceback.format_exc()
                print(f"{evaluation_stage.title()} inference failed:\n", exception_trace)
                #TODO consider passing this to the iteration-planning step instead of failing whole iteration
                raise IterationRunFailed(
                    message=f"Inference on {evaluation_stage} data failed.",
                    context_messages=[],
                    exception_trace=exception_trace,
                ) from None
        return metrics
