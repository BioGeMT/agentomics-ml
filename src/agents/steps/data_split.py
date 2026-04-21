from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import cast

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, ModelRetry, RunContext
import pandas as pd

from agents.steps.base import AgenticStep, AgenticStepOutput
from runtime.filesystem import chown_tree_to_root
from runtime.read_write_utils import get_last_successful_iteration, load_current_iteration_index, load_iteration_state, update_current_iteration_state
from runtime.step_outputs import load_step_output
from run_logging.logging_helpers import log_split_is_allowed
from datasets.dataset_utils import get_numeric_label_col_from_prepared_dataset

class DataSplitOutput(AgenticStepOutput):
    train_path: str = Field(description="Path to generated train.csv file")
    val_path: str = Field(description="Path to generated validation.csv file")
    splitting_strategy: str = Field(description="Detailed description of the splitting strategy used")
    split_changed: bool = Field(
        default=False,
        description="Whether the train/validation split changed during this step. Populated programmatically.",
    )
    split_version: SkipJsonSchema[int] = Field(
        default=0,
        description="Version number of the split used in this step. Populated programmatically.",
    )

class DataSplitStep(AgenticStep):
    step_id = "data_split"
    display_name = "SPLITTING"
    output_type = DataSplitOutput

    def _get_latest_split_strategy(self) -> str:
        iteration = get_last_successful_iteration(self.config)
        assert iteration is not None
        return load_step_output(self.config, self.step_id, self.config.iteration_dir(iteration)).splitting_strategy

    def _get_latest_split_dir(self) -> Path | None:
        dirs = [d for d in self.config.splits_dir.iterdir() if d.is_dir()]
        return max(dirs, key=lambda d: int(d.name.split("_")[1])) if dirs else None

    def _get_next_split_dir(self) -> Path:
        latest = self._get_latest_split_dir()
        if latest is None:
            return self.config.splits_dir / "split_0"
        n = int(latest.name.split("_")[1])
        return self.config.splits_dir / f"split_{n + 1}"

    def _move_split_to_versioned_dir(self, result: DataSplitOutput) -> DataSplitOutput:
        train_path = Path(result.train_path)
        val_path = Path(result.val_path)
        next_split_dir = self._get_next_split_dir()
        next_split_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(train_path), next_split_dir / "train.csv")
        shutil.move(str(val_path), next_split_dir / "validation.csv")
        result.train_path = str(next_split_dir / "train.csv")
        result.val_path = str(next_split_dir / "validation.csv")
        result.split_changed = True
        result.split_version = int(next_split_dir.name.removeprefix("split_"))
        return result

    def attach_output_validator(self, agent: Agent[dict, AgenticStepOutput]) -> None:
        @agent.output_validator
        async def validate_split_dataset(ctx: RunContext[dict], result: AgenticStepOutput) -> AgenticStepOutput:
            result = cast(DataSplitOutput, result)
            if not os.path.exists(result.train_path) or not os.path.exists(result.val_path):
                raise ModelRetry("Split dataset files do not exist.")

            train_path = Path(result.train_path)
            val_path = Path(result.val_path)
            if train_path.name != 'train.csv' or val_path.name != 'validation.csv':
                for p in (train_path, val_path):
                    if p.parent == self.config.current_step_dir:
                        p.unlink(missing_ok=True)
                raise ModelRetry(f"The files must be called exactly 'train.csv' and 'validation.csv'. Files ({train_path.name} and {val_path.name}) have been deleted.")

            target_col = get_numeric_label_col_from_prepared_dataset(self.config.prepared_dataset_dir)
            required_cols = [target_col, 'id']
            train_df, val_df = pd.read_csv(result.train_path), pd.read_csv(result.val_path)
            for df, path in [(train_df, result.train_path), (val_df, result.val_path)]:
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    raise ModelRetry(f"Required columns {missing} missing from {path}.")

            train_ids = set(train_df['id'].dropna().tolist())
            val_ids = set(val_df['id'].dropna().tolist())
            if train_ids.intersection(val_ids):
                raise ModelRetry("Train and validation datasets have overlapping IDs. IDs must be unique across train and validation splits.")

            #The moved files will be absent from created_files field of the output object, however will be in the structured output field
            if not Path(result.train_path).is_relative_to(self.config.splits_dir):
                #TODO what if its the explicit split files -> should be moved or copied?
                result = self._move_split_to_versioned_dir(result)
            else:
                result.splitting_strategy = self._get_latest_split_strategy()
                result.split_version = int(Path(result.train_path).parent.name.removeprefix("split_"))
            return result

    def step_prompt(self) -> str:
        iteration = load_current_iteration_index(self.config)
        latest_split_dir = self._get_latest_split_dir()
        current_step_dir = self.config.current_step_dir

        if self.config.task_type == 'classification':
            extra_instructions = "Ensure that the validation split contains representative samples from ALL classes."
        elif self.config.task_type == 'regression':
            extra_instructions = ""
        else:
            raise ValueError(f"Unknown task type: {self.config.task_type}. Supported types are 'classification' and 'regression'.")

        if iteration != 0 and latest_split_dir is not None:
            extra_info = f"""
            Note: An existing split is already available at {latest_split_dir} (train.csv and validation.csv).
            If you don't have a reason to change the splitting strategy, return those existing paths immediately and return an empty string for the splitting strategy.
            If you do create a new split, save 'train.csv' and 'validation.csv' in {current_step_dir}.
            """
        else:
            extra_info = f"Save 'train.csv' and 'validation.csv' in {current_step_dir}."

        train_csv_path = self.config.agent_dataset_dir / "train.csv"
        return f"""
            Your next task: Split the training dataset ({train_csv_path}) into training and validation sets.
            Ensure the validation split is representative of new unseen data, since it will be used for optimizing choices like architecture, hyperparameters, and training strategies.
            {extra_instructions}
            Return the absolute paths to the train and validation files.

            {extra_info}
            """

    def on_iteration_start(self, iteration: int) -> None:
        if (self.config.agent_dataset_dir / "validation.csv").exists():
            split_allowed = False
        else:
            has_reusable = any(d.is_dir() for d in self.config.splits_dir.iterdir())
            if not has_reusable:
                split_allowed = True
            elif self.config.split_time_deadline is None:
                split_allowed = iteration < self.config.split_allowed_iterations
            else:
                split_allowed = time.time() < self.config.split_time_deadline
        update_current_iteration_state(self.config, split_allowed_at_start=split_allowed)

    def should_run(self) -> bool:
        return bool(load_iteration_state(self.config.current_iteration_dir)["split_allowed_at_start"])

    def build_skipped_output(self) -> DataSplitOutput:
        latest_split_dir = self._get_latest_split_dir()
        if latest_split_dir is None:
            if not (self.config.agent_dataset_dir / "validation.csv").exists(): #check for explicit validation files
                raise AssertionError(
                    "Agent did not have a chance to split data. "
                    "Provide a non-zero split budget or ensure split files are available on disk."
                )
            latest_split_dir = self._get_next_split_dir()
            latest_split_dir.mkdir(parents=True, exist_ok=True)
            for f in ("train.csv", "validation.csv"):
                shutil.copy2(self.config.agent_dataset_dir / f, latest_split_dir / f)
            splitting_strategy = ""
        else:
            splitting_strategy = self._get_latest_split_strategy()
        return DataSplitOutput(
            train_path=str(latest_split_dir / "train.csv"),
            val_path=str(latest_split_dir / "validation.csv"),
            splitting_strategy=splitting_strategy,
            split_changed=False,
            split_version=int(latest_split_dir.name.removeprefix("split_")),
        )

    def on_step_success(self, output: DataSplitOutput) -> None:
        if self.config.agent_user:
            chown_tree_to_root(Path(output.train_path).parent)
        super().on_step_success(output)

    def on_iteration_fail(self, iteration: int) -> None:
        output = load_step_output(self.config, self.step_id, self.config.current_iteration_dir)
        if output is not None and output.split_changed:
            split_dir = Path(output.train_path).parent
            if split_dir.exists():
                shutil.rmtree(split_dir)

    def on_iteration_end(self, iteration: int) -> None:
        iteration_state = load_iteration_state(self.config.current_iteration_dir)
        log_split_is_allowed(
            iteration=iteration,
            is_allowed=bool(iteration_state["split_allowed_at_start"]),
        )
        #TODO log if split has changed?

