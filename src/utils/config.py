from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from utils.dataset_utils import get_task_type_from_prepared_dataset

@dataclass
class Config:
    # defined at runtime
    model: str
    feedback_model: str
    dataset: str
    tags: List[str]
    val_metric: str
    prepared_dataset_dir: Path
    agent_dataset_dir: Path
    workspace_dir: Path
    snapshot_dir: Path
    root_privileges: bool
    iterations: int
    task_type: str

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
    
    # # Logging and tracing configuration
    # # These can be overridden via command line arguments or environment variables
    # wandb_entity: str = "ceitec-ai"                # --wandb-entity
    # wandb_project: str = "Agentomics-ML"           # --wandb-project  
    # weave_project: Optional[str] = None            # --weave-project (defaults to {wandb_entity}/{wandb_project})
    
    # #TODO reintroduce Weave
    # def get_weave_project(self) -> str:
    #     """Get the Weave project name. Defaults to wandb_entity/wandb_project if weave_project is None."""
    #     return self.weave_project or f"{self.wandb_entity}/{self.wandb_project}"

def make_config(
    model: str,
    feedback_model: str,
    dataset: str,
    tags: List[str],
    val_metric: str,
    root_privileges: bool,
    workspace_dir: Path,
    prepared_datasets_dir: Path,
    agent_dataset_dir: Path,
    # wandb_entity: Optional[str] = None,
    # wandb_project: Optional[str] = None,
    # weave_project: Optional[str] = None,
    max_steps: Optional[int] = None,
    iterations: Optional[int] = 5,
) -> Config:
    config = Config(
        model=model,
        feedback_model=feedback_model,
        dataset=dataset,
        tags=tags,
        val_metric=val_metric,
        root_privileges=root_privileges,
        workspace_dir=workspace_dir,
        prepared_dataset_dir=prepared_datasets_dir / dataset,
        agent_dataset_dir=agent_dataset_dir / dataset,
        snapshot_dir= workspace_dir / "snapshots",
        iterations=iterations,
        task_type = get_task_type_from_prepared_dataset(prepared_datasets_dir / dataset),
        max_steps=max_steps,
    )
    
    # Override defaults with provided values
    # if wandb_entity is not None:
    #     config.wandb_entity = wandb_entity
    # if wandb_project is not None:
    #     config.wandb_project = wandb_project
    # if weave_project is not None:
    #     config.weave_project = weave_project
    # if max_steps is not None:
    #     config.max_steps = max_steps
        
    return config
