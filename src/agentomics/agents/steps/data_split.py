from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import cast

import pandas as pd
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, ModelRetry, RunContext

from agentomics.agents.steps.base import AgenticStep, AgenticStepOutput
from agentomics.runtime.filesystem import (
    chown_tree_to_root,
    create_absolute_symlink,
    rewrite_symlinks_to_absolute,
    validate_symlinks_targets_in,
)
from agentomics.utils.task_types import TaskTypes
from agentomics.runtime.read_write_utils import (
    get_last_successful_iteration,
    load_current_iteration_index,
    load_iteration_state,
    update_current_iteration_state,
)
from agentomics.runtime.step_outputs import load_step_output
from agentomics.run_logging.logging_helpers import log_split_is_allowed
from agentomics.datasets.data_contract import (
    ID_COLUMN_NAME,
    INPUT_DIR_NAME,
    LABELS_FILE_NAME,
    METADATA_FILE_NAME,
    MINI_TRAIN_SPLIT,
    NUMERIC_LABEL_COLUMN_NAME,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    validate_splits,
)

class DataSplitOutput(AgenticStepOutput):
    train_path: str = Field(description="Path to generated train split folder")
    val_path: str = Field(description="Path to generated validation split folder")
    mini_train_path: str = Field(description="Path to generated mini_train split folder")
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

    CLASSIFICATION_MINI_TRAIN_SAMPLES_PER_CLASS = 100
    REGRESSION_MINI_TRAIN_SAMPLE_COUNT = 100

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

    def _load_expected_input_structure(self) -> list[str]:
        metadata_path = self.config.dataset_dir / METADATA_FILE_NAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "input_structure" not in metadata:
            raise ValueError(f"{metadata_path} is missing input_structure")
        return metadata["input_structure"]

    def _validate_mini_train_classification(self, train_df: pd.DataFrame, mini_train_df: pd.DataFrame) -> None:
        train_class_counts = train_df[NUMERIC_LABEL_COLUMN_NAME].value_counts()
        mini_train_class_counts = mini_train_df[NUMERIC_LABEL_COLUMN_NAME].value_counts()
        missing_classes = set(train_class_counts.index) - set(mini_train_class_counts.index)
        if missing_classes:
            raise ModelRetry(
                f"mini_train must contain samples from all classes. "
                f"Missing classes: {sorted(missing_classes)}"
            )
        for cls, mini_count in mini_train_class_counts.items():
            expected_count = min(self.CLASSIFICATION_MINI_TRAIN_SAMPLES_PER_CLASS, train_class_counts[cls])
            if mini_count != expected_count:
                raise ModelRetry(
                    f"mini_train must have exactly {expected_count} samples for class {cls} "
                    f"(min({self.CLASSIFICATION_MINI_TRAIN_SAMPLES_PER_CLASS}, {train_class_counts[cls]}) available in train), "
                    f"got {mini_count}."
                )

    def _validate_mini_train_regression(self, mini_train_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
        expected_count = min(self.REGRESSION_MINI_TRAIN_SAMPLE_COUNT, len(train_df))
        if len(mini_train_df) != expected_count:
            raise ModelRetry(
                f"mini_train must have exactly {expected_count} samples "
                f"(min({self.REGRESSION_MINI_TRAIN_SAMPLE_COUNT}, {len(train_df)}) available in train), "
                f"got {len(mini_train_df)}."
            )

    def _raise_on_unnecessary_copies(self, split_path: Path) -> None:
        dataset_train_input = self.config.dataset_dir / TRAIN_SPLIT / INPUT_DIR_NAME
        split_input = split_path / INPUT_DIR_NAME
        copied_files = []
        for path in split_input.rglob("*"):
            if path.is_symlink() or path.is_dir():
                continue
            # Assumes split mirrors the original structure.
            # If the agent reorganized files, original.is_file() is False and we skip — no false positives but could have some false negatives.
            original = dataset_train_input / path.relative_to(split_input)
            if original.is_file() and path.stat().st_size == original.stat().st_size:
                copied_files.append(path)
        if copied_files:
            names = [str(c.relative_to(split_path)) for c in copied_files[:10]]
            raise ModelRetry(
                f"Files in {split_path.name}/input/ that exist unchanged in the source dataset "
                f"must be symbolic links, not copies. Up to first 10: {names}"
            )

    def _move_split_to_versioned_dir(self, result: DataSplitOutput, flag_as_changed: bool) -> DataSplitOutput:
        train_path = Path(result.train_path)
        val_path = Path(result.val_path)
        mini_train_path = Path(result.mini_train_path)
        next_split_dir = self._get_next_split_dir()
        next_split_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(train_path), next_split_dir / TRAIN_SPLIT)
        shutil.move(str(val_path), next_split_dir / VALIDATION_SPLIT)
        shutil.move(str(mini_train_path), next_split_dir / MINI_TRAIN_SPLIT)
        result.train_path = str(next_split_dir / TRAIN_SPLIT)
        result.val_path = str(next_split_dir / VALIDATION_SPLIT)
        result.mini_train_path = str(next_split_dir / MINI_TRAIN_SPLIT)
        result.split_changed = flag_as_changed
        result.split_version = int(next_split_dir.name.removeprefix("split_"))
        return result

    def attach_output_validator(self, agent: Agent[dict, AgenticStepOutput]) -> None:
        @agent.output_validator
        async def validate_split_dataset(ctx: RunContext[dict], result: AgenticStepOutput) -> AgenticStepOutput:
            result = cast(DataSplitOutput, result)
            train_path = Path(result.train_path)
            val_path = Path(result.val_path)
            mini_train_path = Path(result.mini_train_path)
            if train_path.name != TRAIN_SPLIT or val_path.name != VALIDATION_SPLIT or mini_train_path.name != MINI_TRAIN_SPLIT:
                raise ModelRetry(
                    f"Split folders must be named exactly: "
                    f"train_path='{TRAIN_SPLIT}', val_path='{VALIDATION_SPLIT}', mini_train_path='{MINI_TRAIN_SPLIT}'. "
                    f"Received: train_path='{train_path.name}', val_path='{val_path.name}', mini_train_path='{mini_train_path.name}'."
                )
            expected_structure = self._load_expected_input_structure()
            try:
                validate_splits(
                    {TRAIN_SPLIT: train_path, VALIDATION_SPLIT: val_path, MINI_TRAIN_SPLIT: mini_train_path},
                    expected_structure,
                )
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc

            train_df = pd.read_csv(train_path / LABELS_FILE_NAME, dtype={ID_COLUMN_NAME: str}, keep_default_na=False)
            val_ids = set(
                pd.read_csv(val_path / LABELS_FILE_NAME, dtype={ID_COLUMN_NAME: str}, keep_default_na=False)[ID_COLUMN_NAME].tolist()
            )
            train_ids = set(train_df[ID_COLUMN_NAME].tolist())
            overlapping_ids = train_ids & val_ids
            if overlapping_ids:
                raise ModelRetry(
                    "Train and validation labels.csv files have overlapping ids. "
                    f"First overlaps: {list(overlapping_ids)[:20]}"
                )

            mini_train_df = pd.read_csv(mini_train_path / LABELS_FILE_NAME, dtype={ID_COLUMN_NAME: str}, keep_default_na=False)
            mini_train_ids = set(mini_train_df[ID_COLUMN_NAME].tolist())
            if not mini_train_ids.issubset(train_ids):
                extra_ids = mini_train_ids - train_ids
                raise ModelRetry(
                    f"mini_train IDs must be a subset of training IDs. "
                    f"Found {len(extra_ids)} IDs not in train: {list(extra_ids)[:10]}"
                )
            if self.config.task_type == TaskTypes.CLASSIFICATION:
                self._validate_mini_train_classification(train_df, mini_train_df)
            else:
                self._validate_mini_train_regression(mini_train_df, train_df)

            is_mini_train_only = load_iteration_state(self.config.current_iteration_dir)["only_mini_split_allowed_at_start"]
            if is_mini_train_only:
                step_dir = self.config.current_step_dir
                for split_name in [TRAIN_SPLIT, VALIDATION_SPLIT]:
                    split_link = step_dir / split_name
                    create_absolute_symlink(self.config.dataset_dir / split_name, split_link)
                result.train_path = str(step_dir / TRAIN_SPLIT)
                result.val_path = str(step_dir / VALIDATION_SPLIT)
                train_path = Path(result.train_path)
                val_path = Path(result.val_path)

            if train_path.parent != val_path.parent or train_path.parent != mini_train_path.parent:
                raise ModelRetry("Train, validation, and mini_train split folders must be in the same directory.")
            if not train_path.is_relative_to(self.config.splits_dir) or not val_path.is_relative_to(self.config.splits_dir) or not mini_train_path.is_relative_to(self.config.splits_dir):
                split_paths_to_validate = [mini_train_path] if is_mini_train_only else [train_path, val_path, mini_train_path]
                for split_path in split_paths_to_validate:
                    try:
                        validate_symlinks_targets_in(split_path, self.config.dataset_dir)
                    except ValueError as e:
                        raise ModelRetry(str(e))
                    self._raise_on_unnecessary_copies(split_path)
                    rewrite_symlinks_to_absolute(split_path)
                result = self._move_split_to_versioned_dir(result, flag_as_changed=not is_mini_train_only)
            else:
                result.splitting_strategy = self._get_latest_split_strategy()
                result.split_version = int(train_path.parent.name.removeprefix("split_"))
            return result

    def step_prompt(self) -> str:
        iteration = load_current_iteration_index(self.config)
        latest_split_dir = self._get_latest_split_dir()
        current_step_dir = self.config.current_step_dir

        if self.config.task_type == TaskTypes.CLASSIFICATION:
            validation_instructions = "Ensure that the validation split contains representative samples from ALL classes."
            mini_train_size_instructions = (
                f"exactly {self.CLASSIFICATION_MINI_TRAIN_SAMPLES_PER_CLASS} samples per class "
                f"(or all available if the training split has fewer than {self.CLASSIFICATION_MINI_TRAIN_SAMPLES_PER_CLASS} for a class)"
            )
        elif self.config.task_type == TaskTypes.REGRESSION:
            validation_instructions = ""
            mini_train_size_instructions = (
                f"exactly {self.REGRESSION_MINI_TRAIN_SAMPLE_COUNT} samples "
                f"(or all available if the training split has fewer than {self.REGRESSION_MINI_TRAIN_SAMPLE_COUNT})"
            )
        else:
            raise ValueError(f"Unknown task type: {self.config.task_type}. Supported types are {TaskTypes}.")

        if load_iteration_state(self.config.current_iteration_dir)["only_mini_split_allowed_at_start"]:
            train_split_path = self.config.dataset_dir / TRAIN_SPLIT
            validation_split_path = self.config.dataset_dir / VALIDATION_SPLIT
            return f"""
            Your next task: Create a '{MINI_TRAIN_SPLIT}' folder as a subset of the training data.

            The '{TRAIN_SPLIT}' and '{VALIDATION_SPLIT}' splits are already provided:
            - {TRAIN_SPLIT}: {train_split_path}
            - {VALIDATION_SPLIT}: {validation_split_path}
            Do NOT modify or recreate them. Only create the '{MINI_TRAIN_SPLIT}' folder in {current_step_dir}.

            '{MINI_TRAIN_SPLIT}': a subset of the training split ({train_split_path}) with {mini_train_size_instructions}, used solely for automatic system validation.
            The '{MINI_TRAIN_SPLIT}' folder must contain an input/ folder with the data files and a labels.csv file. The input/ must contain only the data that corresponds to the ids in the newly created labels.csv file.
            When populating input/, use symbolic links to the original files in {train_split_path}/input/ instead of copying them. Only create new files when the data must be modified (e.g. splitting a single file that contains multiple samples into per-split subsets). Always create labels.csv as a new file.
            Preserve the original ID scheme: each id in labels.csv must refer to the same data in input/ as it did in the original data.
            The input/ interface must match the original train/input/ interface.

            Return the absolute paths to all three split folders ('{TRAIN_SPLIT}', '{VALIDATION_SPLIT}', and '{MINI_TRAIN_SPLIT}').
            """

        if iteration != 0 and latest_split_dir is not None:
            extra_info = f"""
            Note: An existing split is already available at {latest_split_dir} ({TRAIN_SPLIT}/, {VALIDATION_SPLIT}/, and {MINI_TRAIN_SPLIT}/ folders).
            If you don't have a reason to change the splitting strategy, return those existing paths immediately and return an empty string for the splitting strategy.
            If you do create a new split, save '{TRAIN_SPLIT}', '{VALIDATION_SPLIT}', and '{MINI_TRAIN_SPLIT}' folders in {current_step_dir}.
            """
        else:
            extra_info = f"Save '{TRAIN_SPLIT}', '{VALIDATION_SPLIT}', and '{MINI_TRAIN_SPLIT}' folders in {current_step_dir}."

        train_split_path = self.config.dataset_dir / TRAIN_SPLIT
        return f"""
            Your next task: Split the training dataset ({train_split_path}) into '{TRAIN_SPLIT}', '{VALIDATION_SPLIT}', and '{MINI_TRAIN_SPLIT}' folders.
            Each split folder must contain an {INPUT_DIR_NAME}/ folder with the data files and a {LABELS_FILE_NAME} file.
            Each split's {INPUT_DIR_NAME}/ must contain only data for ids in that split's {LABELS_FILE_NAME}.
            When populating {INPUT_DIR_NAME}/, use symbolic links to the original files in {train_split_path}/{INPUT_DIR_NAME}/ instead of copying them. Only create new files when the data must be modified (e.g. splitting a single file that contains multiple samples into per-split subsets); do not reuse an unfiltered shared file across splits. Always create {LABELS_FILE_NAME} as a new file.
            Train and validation {LABELS_FILE_NAME} files must contain disjoint ids. mini_train {LABELS_FILE_NAME} must contain only ids that are present in train {LABELS_FILE_NAME}.
            Preserve the original ID scheme: each id in labels.csv must refer to the same data in input/ as it did in the original data.
            The input/ interface must match the original train/input/ interface.

            '{VALIDATION_SPLIT}': representative of new unseen data, used for optimizing architecture, hyperparameters, and training strategies.
            {validation_instructions}

            '{MINI_TRAIN_SPLIT}': a subset of the training split with {mini_train_size_instructions}, used solely for automatic system validation.

            Return the absolute paths to all three split folders.

            {extra_info}
            """

    def on_iteration_start(self, iteration: int) -> None:
        has_reusable = self._get_latest_split_dir() is not None
        if (self.config.dataset_dir / VALIDATION_SPLIT).exists():
            update_current_iteration_state(
                self.config,
                full_split_allowed_at_start=False,
                only_mini_split_allowed_at_start=not has_reusable,
            )
        else:
            if not has_reusable:
                full_split_allowed = True
            elif self.config.split_time_deadline is None:
                full_split_allowed = iteration < self.config.split_allowed_iterations
            else:
                full_split_allowed = time.time() < self.config.split_time_deadline
            update_current_iteration_state(
                self.config,
                full_split_allowed_at_start=full_split_allowed,
                only_mini_split_allowed_at_start=False,
            )

    def should_be_simulated(self) -> bool:
        state = load_iteration_state(self.config.current_iteration_dir)
        return not state["full_split_allowed_at_start"] and not state["only_mini_split_allowed_at_start"]

    def build_simulated_output(self) -> DataSplitOutput:
        latest_split_dir = self._get_latest_split_dir()
        if latest_split_dir is None:
            validation_split = self.config.dataset_dir / VALIDATION_SPLIT
            mini_train_split = self.config.dataset_dir / MINI_TRAIN_SPLIT
            if not validation_split.exists() or not mini_train_split.exists():
                raise AssertionError(
                    "Agent did not have a chance to split data. "
                    "Provide a non-zero split budget or ensure split folders are available on disk."
                )
            latest_split_dir = self._get_next_split_dir()
            latest_split_dir.mkdir(parents=True, exist_ok=True)
            for split_name in [TRAIN_SPLIT, VALIDATION_SPLIT, MINI_TRAIN_SPLIT]:
                create_absolute_symlink(self.config.dataset_dir / split_name, latest_split_dir / split_name)
            splitting_strategy = ""
        else:
            splitting_strategy = self._get_latest_split_strategy()
        return DataSplitOutput(
            train_path=str(latest_split_dir / TRAIN_SPLIT),
            val_path=str(latest_split_dir / VALIDATION_SPLIT),
            mini_train_path=str(latest_split_dir / MINI_TRAIN_SPLIT),
            splitting_strategy=splitting_strategy,
            split_changed=False,
            split_version=int(latest_split_dir.name.removeprefix("split_")),
        )

    def on_step_success(self, output: DataSplitOutput) -> None:
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
            is_allowed=bool(iteration_state["full_split_allowed_at_start"]),
        )
        #TODO log if split has changed?
