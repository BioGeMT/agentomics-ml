from __future__ import annotations

import time
from typing import Any, TypedDict

from pydantic import Field
from pydantic_ai import Agent

from agents.prompts.prompts_utils import get_dataset_knowledge
from agents.steps.base import AgenticStep, AgenticStepOutput
from agents.steps.data_split import DataSplitStep
from agents.steps.validation_evaluation import ValidationEvaluationStep
from runtime.read_write_utils import (
    get_archived_iterations,
    get_last_successful_iteration,
    load_best_run_iteration,
    load_current_iteration_index,
    load_iteration_duration,
    load_iteration_metadata,
)
from runtime.step_outputs import load_step_output, load_step_outputs
from runtime.system_resources import get_resources_summary
from tools.tool_registry import get_tool_names
from utils.config import Config
from utils.foundation_models_utils import build_foundation_model_catalog, format_foundation_model_catalog
from utils.printing_utils import truncate_float


class IterationPlanOutput(AgenticStepOutput):
    data_exploration_instructions: str = Field(
        description="""
        Instructions for the exploration step (you can instruct to skip this step if it's not necessary)
        """
    )
    data_split_instructions: str = Field(
        description="""
        Instructions for the data split step (you can instruct to skip this step if it's not necessary).
        """
    )
    data_representation_instructions: str = Field(
        description="""
        Instructions for the data representation step.
        """
    )
    model_architecture_instructions: str = Field(
        description="""
        Instructions for the machine learning architecture design step.
        """
    )
    model_training_instructions: str = Field(
        description="""
        Instructions for the model training process, including hyperparameters and optimizers.
        """
    )
    model_inference_instructions: str = Field(
        description="""
        Instructions for creating the inference script for the trained model.
        """
    )
    prediction_exploration_instructions: str = Field(
        description="""
        Instructions for the prediction exploration step.
        """
    )
    other_instructions: str | None = Field(
        description="""
        Any other instructions that don't fit only one step (optional).
        """
    )

class IterationPlanStep(AgenticStep):
    step_id = "iteration_plan"
    display_name = "ITERATION PLAN"
    output_type = IterationPlanOutput
    history_excluded_step_ids = {step_id, ValidationEvaluationStep.step_id}

    class IterationHistoryRecord(TypedDict):
        outputs: list[Any]
        metrics: dict[str, float]
        duration: str
        split_version: int
        split_changed: bool
        is_new_best: bool

    def create_agent(self) -> Agent[dict, IterationPlanOutput]:
        return Agent(
            model=self.iteration_plan_model,
            retries=self.config.max_validation_retries,
            model_settings=self.provider.get_high_reasoning_model_settings(
                kwargs={"temperature": self.config.temperature}
            ),
            output_type=IterationPlanOutput,
            deps_type=dict,
        )

    def build_user_prompt(self) -> str:
        return self.step_prompt()

    def step_prompt(self) -> str:
        iteration_history_info = self._build_iteration_history_info()
        splitting_info = self._build_splitting_info()
        time_info = self._build_time_info()
        foundation_models_info = self._build_foundation_models_info()
        exploration_info = self._build_exploration_info()
        tools_info = ", ".join(get_tool_names(self.config, self.config.tool_ids))
        best_iteration_model_info = self._build_best_iteration_model_info()

        return f"""
        <prompt_of_the_iteration_agents>
        {self.config.user_prompt}
        </prompt_of_the_iteration_agents>

        <dataset_knowledge_from_dataset_description_md>
        {get_dataset_knowledge(self.config)}
        </dataset_knowledge_from_dataset_description_md>

        {iteration_history_info}
        <your_instructions>
        {f"<exploration_guidance>{exploration_info}</exploration_guidance>" if exploration_info else ""}
        The main goal of the run is to maximize the hidden test set generalization performance (main metric:{self.config.val_metric}) that will use the 'best iteration model'. {best_iteration_model_info}
        There are {time_info} left before the run ends. Then, the 'best iteration model' will be extracted and automatically evaluated on the hidden test set.
        Once an iteration produces a model with a better {self.config.val_metric} metric, that model will overwrite the 'best iteration model'.

        Your task is to provide concrete instructions for the current iteration agent on what to do or implement in each step of this iteration.
        Use the information from archived iterations and their metrics to inform your instructions.
        Your instructions can suggest anything from reusing a step from any iteration, making small changes to it, up to completelly changing the strategy of a step.
        The iteration agent will always perform the steps in the same sequence.
        Make your instructions concrete, don't offer various choice branches. The exception to this is that you might offer branching instructions for a specific step conditioned on the results of the previous steps of that iteration.
        If you instruct the agent to re-use or partially re-use a step or code from a certain iteration, refer to that iteration by its number.
        The iteration agent will have access to the code and steps' outputs from all archived iterations.
        For example if you want to instruct to re-use the same data representation from iteration 5, instruct "Re-use the data representation from iteration 5".
        The data exploration and splitting steps can be instructed to be skipped completely if they're not needed.
        {splitting_info}

        You're providing instructions to an LLM agent, never offer that you will take any actions to fix or implement fixes yourself.
        Provide only actionable instructions, don't include "Why this helps", "Expected outcomes", or any other non-actionable information.
        If you refer to concrete files, use their name, extension, and the iteration they're from. Don't refer to them by their full path.
        For example "Modify the train.py script from iteration 4 by changing the learning rate to 0.01" is a valid instruction.
        Don't provide instructions that go against the requirements in the common user prompt.
        Don't instruct the agent to update the 'best iteration model', as this is done automatically.
        Never refer to existing scripts or previous iteration agent's actions only as 'previous', 'existing', 'current, 'last', etc... Always mention the iteration number of what you're refering to.
        If you're requesting the agent to create specific files or folders, never request anything with the name 'iteration', 'iter', or similar. For example, prefer 'exploration_script.py' over 'exploration_scipt_iter3.py'. Simply refer to the agent's workspace path as 'your workspace'.

        The agent will have access to the train.csv and validation.csv files, all previous iteration files and step outputs, and the dataset_description.md file.
        The agent will have access to the following tools: {tools_info}.
        <foundation_models_info>
        The agent will have access to the following foundation models: {foundation_models_info}
        </foundation_models_info>
        The agent will have access to the following resources: {get_resources_summary()}

        The iteration history consists of experiments that used some train/validation split strategy (S), data representation (R), and model architecture (A) combinations (S x R x A).
        Based on the iteration history and remaining time choose to either:
        A) explore combinations (S x R x A) not present in the iteration history
        B) further develop an existing promising (S x R x A) combination. If the iteration history already contains a further developed version of a (S x R x A) combination that did not result in a validation {self.config.val_metric} improvement, you must not develop it further again.
        </your_instructions>
        """

    def get_message_history(self):
        return None

    def _build_iteration_history_info(self) -> str:
        sorted_iteration_to_records = {
            iteration: self._load_iteration_history_record(iteration)
            for iteration in sorted(get_archived_iterations(self.config, only_successful=True))  # TODO support only successful or all?
        }

        if len(sorted_iteration_to_records) == 0:
            return f"""
            <iteration_history>
            {len(sorted_iteration_to_records)} archived iterations completed in the current run so far
            </iteration_history>
            """

        iteration_summaries = ""
        for iteration, record in sorted_iteration_to_records.items():
            truncated_iteration_metrics = {
                metric_name: truncate_float(metric_value)
                for metric_name, metric_value in record["metrics"].items()
            }
            split_info = self._build_iteration_split_info(
                iteration=iteration,
                iteration_to_records=sorted_iteration_to_records,
            )
            extra_info = self._build_iteration_extra_info(record)
            standing_info = self._build_iteration_standing_info(
                iteration=iteration,
                was_new_best=record["is_new_best"],
            )
            iteration_summaries += f"""
                <iteration_{iteration}_summary>
                {f"<duration>{record['duration']}</duration>"}
                <steps_outputs>
                {record["outputs"] or "No outputs available."}
                </steps_outputs>
                {f"<extra_info>\n{extra_info}\n</extra_info>" if extra_info else ""}
                <metrics>
                {truncated_iteration_metrics if truncated_iteration_metrics else "Inference script failed, no metrics available."}
                </metrics>
                <standing_info>
                {standing_info}
                </standing_info>
                <split_info>
                {split_info}
                </split_info>
                </iteration_{iteration}_summary>
                """

        return f"""
        <iteration_history>
        {len(sorted_iteration_to_records)} archived iterations completed in the current run so far
        <iterations_summaries>
        {iteration_summaries}
        </iterations_summaries>
        </iteration_history>
        """

    def _load_iteration_history_record(self, iteration: int) -> IterationPlanStep.IterationHistoryRecord:
        iteration_dir = self.config.iteration_dir(iteration)
        steps_of_interest = [
            step_id
            for step_id in self.config.step_sequence
            if step_id not in self.history_excluded_step_ids
        ]
        val_output = load_step_output(self.config, ValidationEvaluationStep.step_id, iteration_dir)
        return self.IterationHistoryRecord(
            outputs=load_step_outputs(
                self.config,
                iteration_dir=iteration_dir,
                step_sequence=steps_of_interest,
            ),
            metrics=val_output.metrics if val_output is not None else {},
            duration=f'{load_iteration_duration(iteration_dir) / 3600:.2f} hours' if load_iteration_duration(iteration_dir) else "",
            split_version=DataSplitStep.get_split_version(self.config, iteration_dir=iteration_dir),
            split_changed=bool(
                load_step_output(self.config, DataSplitStep.step_id, iteration_dir).split_changed
            ),
            is_new_best=bool(val_output.is_new_best) if val_output is not None else False,
        )

    def _build_splitting_info(self) -> str:
        iteration = load_current_iteration_index(self.config)
        best_metric_iteration = load_best_run_iteration(self.config)
        latest_split_version = self._get_latest_split_version()

        if not load_iteration_metadata(self.config.current_iteration_dir)["split_allowed_at_start"]:
            return "Instruct to skip the splitting step, as the current iteration agent cannot split the data."

        if latest_split_version is None:
            split_status = "No reusable train/validation split exists yet. Instruct the agent to create the initial train/validation split for this run."
        else:
            split_status = f"If the split should stay the same (current split version is {latest_split_version}), instruct to 'Re-use the current split'."

        if self.config.split_time_deadline is not None:
            hours_left = (self.config.split_time_deadline - time.time()) / 3600
            split_budget = f"Splitting is allowed for the next {hours_left:.2f} hours."
        else:
            remaining = max(self.config.split_allowed_iterations - (iteration + 1), 0)
            split_budget = f"Splitting is allowed for the next {remaining} iteration/s."

        best_iteration_note = (
            f" If you do instruct to change the split, also instruct to re-use the best iteration "
            f"(currently iteration {best_metric_iteration}) representation, architecture, training scripts and inference scripts."
            if best_metric_iteration is not None else ""
        )

        return (
            f"{split_status} "
            "If you choose data splitting needs change, never suggest cross-validations split or any other split "
            "that would result in more than two files (train.csv and validation.csv). "
            "Keep in mind that using a more representative validation split will result in a better selected "
            "'best iteration model' and therefore a better final hidden test set metrics. "
            "Based on the iteration history, if you suspect the current split is not representative "
            "and/or produces overfitted models, instruct to change the splitting strategy to a more representative and robust one."
            f"{best_iteration_note}\n"
            f"{split_budget} After that, the latest split version will be used for all future iterations."
        )

    def _build_time_info(self) -> str:
        iteration = load_current_iteration_index(self.config)
        time_deadline = self.config.time_deadline
        time_left_in_seconds = time_deadline - time.time() if time_deadline is not None else None
        if time_left_in_seconds is not None:
            return f"{time_left_in_seconds / 3600:.2f} hours"
        iterations_left = self.config.iterations - iteration - 1
        return f"{iterations_left} iterations"

    def _build_foundation_models_info(self) -> str:
        catalog = build_foundation_model_catalog(self.config.foundation_models_yaml, self.config.foundation_models_type)
        if not catalog:
            return "No foundation models available"

        foundation_models_info = format_foundation_model_catalog(catalog)
        foundation_models_info += (
            "\n<foundation_model_guidelines>\n"
            "When loading foundation model weights in certain ways (for example "
            "transformers.AutoModelForSequenceClassification) a randomly initialized layer gets added "
            "on top of the loaded model. If not trained, this layer will harm prediction accuracy.\n"
            "If you instruct the agent to use a foundation model, you must:\n"
            "1) Instruct the train script to train any layers that were automatically created and "
            "randomly initialized by the transformers library, and save them to the training "
            "artifacts directory\n"
            "2) Instruct the inference step to load these saved layers properly and avoid using "
            "randomly initialized layers in the inference.py script\n"
            "</foundation_model_guidelines>"
        )
        return foundation_models_info

    def _build_exploration_info(self) -> str:
        iteration = load_current_iteration_index(self.config)
        exploration_iterations = self.config.exploration_iterations
        if iteration >= exploration_iterations:
            return ""

        return (
            f"You are still in the exploration phase (current iteration: {iteration}; exploration phase covers iterations 0 to {int(exploration_iterations) - 1}). "
            "Your task is to instruct the agent to implement a baseline model. "
            "Follow these instructions for these specific steps: "
            "Data Representation: implement the most basic commonly used representation for the data type. "
            "Model Architecture: use a classical machine learning model or a simple low-parameter neural network. "
            "Never suggest a representation-model combination that has already been explored."
        )

    def _build_iteration_extra_info(self, record: IterationPlanStep.IterationHistoryRecord) -> str:
        extra_info_parts: list[str] = []
        if record["split_changed"]:
            extra_info_parts.append(
                "The train/validation split changed in this iteration, so metrics from older split versions are not directly comparable to it."
            )
        return "\n".join(extra_info_parts)

    def _build_iteration_standing_info(self, iteration: int, was_new_best: bool) -> str:
        best_metric_iteration = load_best_run_iteration(self.config)
        if best_metric_iteration is None:
            #TODO can this happen?
            return ""
        if iteration == best_metric_iteration:
            return "This iteration is currently selected as the best iteration."
        if was_new_best:
            return "This iteration became the best iteration when it finished, but it is no longer the current best iteration."
        return "This iteration is not the best iteration"

    def _build_best_iteration_model_info(self) -> str:
        best_metric_iteration = load_best_run_iteration(self.config)
        latest_split_version = self._get_latest_split_version()
        if best_metric_iteration is None:
            return "No best iteration model exists yet. Only models using the latest split are candidates for the future best iteration model."
        if latest_split_version is None:
            return (
                f"The current best iteration model is from iteration {best_metric_iteration}. "
                "Only models using the latest split are candidates for this best iteration model."
            )
        return (
            f"The current best iteration model is from iteration {best_metric_iteration}. "
            f"Only models using the latest split (version {latest_split_version}) are candidates for this best iteration model."
        )

    def _build_iteration_split_info(self, iteration: int, iteration_to_records: dict[int, IterationPlanStep.IterationHistoryRecord]) -> str:
        latest_split_version = self._get_latest_split_version()
        if latest_split_version is None:
            raise ValueError("Latest split version must exist when archived iteration records are available.")

        iteration_split_version = iteration_to_records[iteration]["split_version"]
        iterations_with_same_split = [
            other_iteration
            for other_iteration, other_record in iteration_to_records.items()
            if other_iteration != iteration and other_record["split_version"] == iteration_split_version
        ]
        has_multiple_split_versions = (
            len({record["split_version"] for record in iteration_to_records.values()}) > 1
        )

        split_info = (
            "This iteration used train/validation split strategy version "
            f"{iteration_split_version}. "
        )
        if iterations_with_same_split:
            split_info += f"This is the same as iterations {iterations_with_same_split}. "
        if iteration_split_version != latest_split_version:
            split_info += (
                f"This iteration's split (version {iteration_split_version}) is different from the "
                f"latest iteration's split ({latest_split_version}), therefore this iteration "
                "metrics can no longer be considered for a 'best iteration model' candidate. "
            )
        else:
            split_info += (
                f"This iteration's split (version {iteration_split_version}) is the latest split "
                "version, therefore its metrics are considered for a 'best iteration model' candidate. "
            )
        if has_multiple_split_versions:
            split_info += "Note that metrics calculated from different split versions are not directly comparable. "
        return split_info

    def _get_latest_split_version(self) -> int | None:
        latest_successful_iteration = get_last_successful_iteration(self.config)
        if latest_successful_iteration is None:
            return None
        return DataSplitStep.get_split_version(
            self.config,
            iteration_dir=self.config.iteration_dir(latest_successful_iteration),
        )
