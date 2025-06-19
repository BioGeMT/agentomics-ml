from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class Config:
    # defined at runtime
    model: str
    feedback_model: str
    dataset: str
    tags: List[str]
    best_metric: str #TODO rename into validation_metric
    dataset_dir: Path
    workspace_dir: Path
    snapshot_dir: Path
    root_privileges: bool
    agent_id: Optional[str] = None # assigned after user creation

    # static defaults
    temperature: float = 1.0
    max_steps: int = 100 #TODO rename, this is per-step limit
    max_run_retries: int = 1
    max_validation_retries: int = 5
    use_proxy: bool = True
    iterations: int = 5
    llm_response_timeout: int = 60 * 15
    bash_tool_timeout: int = 60 * 5
    write_python_tool_timeout: int = 60 * 1
    run_python_tool_timeout: int = 60 * 60 * 6 #This affects max training time
    credit_budget: int = 30
    max_tool_retries: int = 5

def make_config(
    model: str,
    feedback_model: str,
    dataset: str,
    tags: List[str],
    best_metric: str,
    root_privileges: bool,
    workspace_dir: Path,
    dataset_dir: Path
) -> Config:
    return Config(
        model=model,
        feedback_model=feedback_model,
        dataset=dataset,
        tags=tags,
        best_metric=best_metric,
        root_privileges=root_privileges,
        workspace_dir=workspace_dir,
        dataset_dir = dataset_dir / dataset,
        snapshot_dir= workspace_dir / "snapshots"
    )
