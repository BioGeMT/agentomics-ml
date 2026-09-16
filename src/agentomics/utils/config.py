from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from agentomics.runtime.system_resources import get_resources_summary
from agentomics.utils.versioning import get_version


@dataclass(kw_only=True)
class Config:
    CONFIG_FILENAME: ClassVar[str] = "config.json"
    RUN_DIRNAME: ClassVar[str] = "run"
    AGENT_USER: ClassVar[str] = "agentomics-agent"
    SHARED_DIRNAME: ClassVar[str] = "shared"
    RUNTIME_INFO_DIRNAME: ClassVar[str] = "runtime_info"
    BEST_ITERATION_SNAPSHOT_DIRNAME: ClassVar[str] = "best_iteration_snapshot"
    ITERATION_DIR_PREFIX: ClassVar[str] = "iteration_"
    BASE_PROMPT_FILENAME: ClassVar[str] = "base_prompt.txt"
    SYSTEM_PROMPT_FILENAME: ClassVar[str] = "system_prompt.txt"
    ITERATION_METADATA_FILENAME: ClassVar[str] = "iteration_metadata.json"
    ITERATION_STATE_FILENAME: ClassVar[str] = "iteration_state.json"
    ENVIRONMENT_DESCRIPTOR_FILENAME: ClassVar[str] = "environment.yml"
    STEP_OUTPUT_FILENAME: ClassVar[str] = "output.json"
    FETCHED_PAPERS_DIRNAME: ClassVar[str] = "fetched_papers"

    DEFAULT_ITERATIONS: ClassVar[int] = 5
    DEFAULT_SPLIT_ALLOWED_ITERATIONS: ClassVar[int] = 1
    DEFAULT_EXPLORATION_ITERATIONS: ClassVar[int] = 2
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
        "paper_fetch",
        "bash",
        "write_python",
        "run_python",
        "replace",
    ]
    DEFAULT_MAX_STEPS: ClassVar[int] = 100
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.7
    DEFAULT_MAX_VALIDATION_RETRIES: ClassVar[int] = 5
    DEFAULT_PAPER_FETCH_REQUEST_LIMIT: ClassVar[int] = 10
    DEFAULT_PAPER_FETCH_MAX_RESULTS: ClassVar[int] = 10
    DEFAULT_USE_PROXY: ClassVar[bool] = True
    DEFAULT_LLM_RESPONSE_TIMEOUT: ClassVar[int] = 60 * 10
    DEFAULT_BASH_TOOL_TIMEOUT: ClassVar[int] = 60 * 3
    DEFAULT_WEB_FETCH_TIMEOUT: ClassVar[int] = 60
    DEFAULT_MAX_TOOL_RETRIES: ClassVar[int] = 5
    DEFAULT_RUN_PYTHON_TOOL_TIMEOUT: ClassVar[int] = 60 * 60 * 6
    DEFAULT_USER_PROMPT: ClassVar[str] = "Develop a machine learning model that generalizes well to new unseen data."

    # Required, settable through the CLI (run_agent_interactive.py)
    model_name: str
    iteration_plan_model_name: str
    dataset: str
    tags: list[str]
    val_metric: str
    workspace_dir: str
    datasets_dir: str

    # Required, not settable through the CLI
    agent_id: str
    task_type: str
    input_structure: list[str]
    agentomics_version: str = get_version()

    # Optional, default values can be overwritten through the CLI
    user_prompt: str = DEFAULT_USER_PROMPT
    iterations: int = DEFAULT_ITERATIONS
    provider_name: str | None = None
    split_allowed_iterations: int = DEFAULT_SPLIT_ALLOWED_ITERATIONS
    exploration_iterations: int = DEFAULT_EXPLORATION_ITERATIONS
    time_deadline: int | None = None
    split_time_deadline: int | None = None
    run_python_tool_timeout: int = DEFAULT_RUN_PYTHON_TOOL_TIMEOUT
    disable_training_reporting: bool = False
    conda_export_mode: str = "full"

    # Optional, default values can not be overwritten through the CLI
    max_steps: int = DEFAULT_MAX_STEPS
    step_sequence: list[str] = field(default_factory=lambda: Config.DEFAULT_STEP_SEQUENCE.copy())
    tool_ids: list[str] = field(default_factory=lambda: Config.DEFAULT_TOOL_IDS.copy())
    temperature: float = DEFAULT_TEMPERATURE
    max_validation_retries: int = DEFAULT_MAX_VALIDATION_RETRIES
    use_proxy: bool = DEFAULT_USE_PROXY
    llm_response_timeout: int = DEFAULT_LLM_RESPONSE_TIMEOUT
    bash_tool_timeout: int = DEFAULT_BASH_TOOL_TIMEOUT
    web_fetch_timeout: int = DEFAULT_WEB_FETCH_TIMEOUT
    paper_fetch_request_limit: int = DEFAULT_PAPER_FETCH_REQUEST_LIMIT
    paper_fetch_max_results: int = DEFAULT_PAPER_FETCH_MAX_RESULTS
    max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES
    label_to_scalar: dict[str, int] | None = None
    wandb_run_id: str | None = None

    @property
    def run_dir(self) -> Path:
        return Path(self.workspace_dir) / self.RUN_DIRNAME

    @property
    def reports_dir(self) -> Path:
        return Path(self.workspace_dir) / "reports"

    @property
    def markdown_reports_dir(self) -> Path:
        return self.reports_dir / "markdown"

    @property
    def pdf_reports_dir(self) -> Path:
        return self.reports_dir / "pdf"

    @property
    def logs_dir(self) -> Path:
        return Path(self.workspace_dir) / "logs"

    @property
    def dataset_dir(self) -> Path:
        return Path(self.datasets_dir) / self.dataset

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
    def fetched_papers_dir(self) -> Path:
        return self.shared_dir / self.FETCHED_PAPERS_DIRNAME

    @property
    def current_iteration_runtime_info_dir(self) -> Path:
        return self.current_iteration_dir / self.RUNTIME_INFO_DIRNAME

    @property
    def splits_dir(self) -> Path:
        return self.shared_dir / "splits"

    @property
    def best_iteration_snapshot_dir(self) -> Path:
        return Path(self.workspace_dir) / self.BEST_ITERATION_SNAPSHOT_DIRNAME

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
