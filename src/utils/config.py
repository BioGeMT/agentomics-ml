from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from runtime.system_resources import get_resources_summary


@dataclass(kw_only=True)
class Config:
    CONFIG_FILENAME: ClassVar[str] = "config.json"
    SHARED_DIRNAME: ClassVar[str] = "shared"
    RUNTIME_INFO_DIRNAME: ClassVar[str] = "runtime_info"
    ITERATION_DIR_PREFIX: ClassVar[str] = "iteration_"
    BASE_PROMPT_FILENAME: ClassVar[str] = "base_prompt.txt"
    SYSTEM_PROMPT_FILENAME: ClassVar[str] = "system_prompt.txt"
    ITERATION_METADATA_FILENAME: ClassVar[str] = "iteration_metadata.json"
    ITERATION_STATE_FILENAME: ClassVar[str] = "iteration_state.json"
    STEP_OUTPUT_FILENAME: ClassVar[str] = "output.json"
    DEFAULT_ITERATIONS: ClassVar[int] = 5
    DEFAULT_SPLIT_ALLOWED_ITERATIONS: ClassVar[int] = 1
    DEFAULT_EXPLORATION_ITERATIONS: ClassVar[int] = 4
    DEFAULT_STEP_SEQUENCE: ClassVar[list[str]] = [
        "iteration_plan",
        "data_exploration",
        "data_split",
        "data_representation",
        "model_architecture",
        "model_training",
        "model_inference",
        "prediction_exploration",
        "validation_evaluation",
    ]
    DEFAULT_TOOL_IDS: ClassVar[list[str]] = [
        "bash",
        "write_python",
        "run_python",
        "foundation_models_info",
        "replace",
    ]
    DEFAULT_MAX_STEPS: ClassVar[int] = 100
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.7
    DEFAULT_MAX_VALIDATION_RETRIES: ClassVar[int] = 5
    DEFAULT_USE_PROXY: ClassVar[bool] = True
    DEFAULT_LLM_RESPONSE_TIMEOUT: ClassVar[int] = 60 * 10
    DEFAULT_BASH_TOOL_TIMEOUT: ClassVar[int] = 60 * 3
    DEFAULT_MAX_TOOL_RETRIES: ClassVar[int] = 5
    DEFAULT_RUN_PYTHON_TOOL_TIMEOUT: ClassVar[int] = 60 * 60 * 6

    agent_id: str
    model_name: str
    iteration_plan_model_name: str
    dataset: str
    tags: list[str]
    val_metric: str
    workspace_dir: str
    prepared_datasets_dir: str
    user_prompt: str
    task_type: str
    iterations: int = DEFAULT_ITERATIONS
    provider_name: str | None = None
    max_steps: int = DEFAULT_MAX_STEPS
    split_allowed_iterations: int = DEFAULT_SPLIT_ALLOWED_ITERATIONS
    exploration_iterations: int = DEFAULT_EXPLORATION_ITERATIONS
    step_sequence: list[str] = field(default_factory=lambda: Config.DEFAULT_STEP_SEQUENCE.copy())
    tool_ids: list[str] = field(default_factory=lambda: Config.DEFAULT_TOOL_IDS.copy())
    time_deadline: int | None = None
    split_time_deadline: int | None = None
    run_python_tool_timeout: int = DEFAULT_RUN_PYTHON_TOOL_TIMEOUT
    temperature: float = DEFAULT_TEMPERATURE
    max_validation_retries: int = DEFAULT_MAX_VALIDATION_RETRIES
    use_proxy: bool = DEFAULT_USE_PROXY
    llm_response_timeout: int = DEFAULT_LLM_RESPONSE_TIMEOUT
    bash_tool_timeout: int = DEFAULT_BASH_TOOL_TIMEOUT
    max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES
    wandb_run_id: str | None = None
    foundation_models_type: str | None = None
    foundation_models_yaml: str | None = None
    agent_user: str | None = None

    @property
    def runs_dir(self) -> Path:
        return Path(self.workspace_dir) / "runs"

    @property
    def snapshots_dir(self) -> Path:
        return Path(self.workspace_dir) / "snapshots"

    @property
    def fallbacks_dir(self) -> Path:
        return Path(self.workspace_dir) / "fallbacks"

    @property
    def reports_dir(self) -> Path:
        return Path(self.workspace_dir) / "reports"

    @property
    def agent_reports_dir(self) -> Path:
        return self.reports_dir / self.agent_id

    @property
    def extras_dir(self) -> Path:
        return Path(self.workspace_dir) / "extras"

    @property
    def prepared_dataset_dir(self) -> Path:
        return Path(self.prepared_datasets_dir) / self.dataset

    @property
    def agent_dataset_dir(self) -> Path:
        return self.shared_dir / "datasets" / self.dataset

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.agent_id

    @property
    def shared_dir(self) -> Path:
        return self.run_dir / self.SHARED_DIRNAME

    @property
    def config_path(self) -> Path:
        return self.shared_dir / self.CONFIG_FILENAME

    @property
    def current_iteration_dir(self) -> Path:
        return self.run_dir / "current_iteration"

    @property
    def current_step_dir(self) -> Path:
        return self.current_iteration_dir / "current_step"

    @property
    def current_iteration_runtime_info_dir(self) -> Path:
        return self.current_iteration_dir / self.RUNTIME_INFO_DIRNAME

    @property
    def splits_dir(self) -> Path:
        return self.shared_dir / "splits"

    @property
    def snapshot_dir(self) -> Path:
        return self.snapshots_dir / self.agent_id

    def archived_step_dir(self, step_id: str) -> Path:
        return self.current_iteration_dir / step_id

    def iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / f"{self.ITERATION_DIR_PREFIX}{iteration}"

    def print_summary(self) -> None:
        print("=== AGENTOMICS CONFIGURATION ===")
        print("MAIN MODEL:", self.model_name)
        print("ITERATION PLAN MODEL:", self.iteration_plan_model_name)
        print("DATASET:", self.dataset)
        print("TASK TYPE:", self.task_type)
        print("VAL METRIC:", self.val_metric)
        print("AGENT ID:", self.agent_id)
        print("ITERATIONS:", self.iterations)
        print("EXPLORATION ITERATIONS:", self.exploration_iterations)
        print("SPLIT ALLOWED ITERATIONS:", self.split_allowed_iterations)
        print(
            "TIMEOUT IN HOURS:",
            (self.time_deadline - time.time()) / 3600 if self.time_deadline is not None else "No timeout",
        )
        print(
            "SPLIT TIMEOUT IN HOURS:",
            (self.split_time_deadline - time.time()) / 3600
            if self.split_time_deadline is not None
            else "No timeout",
        )
        print("USER PROMPT:", self.user_prompt)
        print("RESOURCES SUMMARY:", get_resources_summary())
        print("===============================")
