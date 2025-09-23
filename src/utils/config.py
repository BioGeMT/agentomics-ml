from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from .dataset_utils import get_task_type_from_prepared_dataset

@dataclass
class Config:
    # defined at runtime
    model_name: str
    feedback_model_name: str
    dataset: str
    tags: List[str]
    val_metric: str
    prepared_dataset_dir: Path
    agent_dataset_dir: Path
    workspace_dir: Path
    snapshots_dir: Path
    runs_dir: Path
    reports_dir: Path
    root_privileges: bool
    iterations: int
    task_type: str
    user_prompt: str

    agent_id: Optional[str] = None # assigned after user creation
    # static defaults
    temperature: float = 1.0
    max_steps: int = 100 #TODO rename, this is per-step limit
    max_run_retries: int = 1
    max_validation_retries: int = 5
    use_proxy: bool = True
    llm_response_timeout: int = 60 * 15
    bash_tool_timeout: int = 60 * 5
    write_python_tool_timeout: int = 60 * 1
    run_python_tool_timeout: int = 60 * 60 * 6 #This affects max training time
    credit_budget: int = 30 # Only applies when using a provisioning openrouter key #TODO
    max_tool_retries: int = 5
    agent_id: str = None

    def __init__(
        self,
        model_name: str,
        feedback_model_name: str,
        dataset: str,
        tags: List[str],
        val_metric: str,
        root_privileges: bool,
        workspace_dir: Path,
        prepared_datasets_dir: Path,
        agent_dataset_dir: Path,
        user_prompt: str,
        max_steps: Optional[int] = None,
        iterations: Optional[int] = 5,
    ):
        self.model_name = model_name
        self.feedback_model_name = feedback_model_name
        self.dataset = dataset
        self.tags = tags
        self.val_metric = val_metric
        self.root_privileges = root_privileges
        self.prepared_dataset_dir = prepared_datasets_dir / dataset
        self.agent_dataset_dir = agent_dataset_dir / dataset
        self.workspace_dir = workspace_dir
        self.runs_dir = workspace_dir / "runs"
        self.snapshots_dir = workspace_dir / "snapshots"
        self.reports_dir = workspace_dir / "reports"
        self.iterations = iterations
        self.task_type = get_task_type_from_prepared_dataset(prepared_datasets_dir / dataset)
        self.user_prompt = user_prompt

        if max_steps is not None:
            self.max_steps = max_steps

    def print_summary(self):
        print('=== AGENTOMICS CONFIGURATION ===')
        print('MAIN MODEL:', self.model_name)
        print('FEEDBACK MODEL:', self.feedback_model_name)
        print('DATASET:', self.dataset)
        print('TASK TYPE:', self.task_type)
        print('VAL METRIC:', self.val_metric)
        print('AGENT ID:', self.agent_id)
        print('ITERATIONS:', self.iterations)
        print('USER PROMPT:', self.user_prompt)
        print('===============================')
