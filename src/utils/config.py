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
    agent_dataset_dir: Path
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
    credit_budget: int = 30 # Only applies when using a provisioning openrouter key
    max_tool_retries: int = 5
    
    # Logging and tracing configuration
    # These can be overridden via command line arguments or environment variables
    wandb_entity: str = "ceitec-ai"                # --wandb-entity
    wandb_project: str = "Agentomics-ML"           # --wandb-project  
    weave_project: Optional[str] = None            # --weave-project (defaults to {wandb_entity}/{wandb_project})
    
    def get_weave_project(self) -> str:
        """Get the Weave project name. Defaults to wandb_entity/wandb_project if weave_project is None."""
        return self.weave_project or f"{self.wandb_entity}/{self.wandb_project}"

def make_config(
    model: str,
    feedback_model: str,
    dataset: str,
    tags: List[str],
    best_metric: str,
    root_privileges: bool,
    workspace_dir: Path,
    dataset_dir: Path,
    agent_dataset_dir: Path,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    weave_project: Optional[str] = None,
    max_steps: Optional[int] = None
) -> Config:
    config = Config(
        model=model,
        feedback_model=feedback_model,
        dataset=dataset,
        tags=tags,
        best_metric=best_metric,
        root_privileges=root_privileges,
        workspace_dir=workspace_dir,
        dataset_dir=dataset_dir / dataset,
        agent_dataset_dir=agent_dataset_dir / dataset,
        snapshot_dir= workspace_dir / "snapshots"
    )
    
    # Override defaults with provided values
    if wandb_entity is not None:
        config.wandb_entity = wandb_entity
    if wandb_project is not None:
        config.wandb_project = wandb_project
    if weave_project is not None:
        config.weave_project = weave_project
    if max_steps is not None:
        config.max_steps = max_steps
        
    return config
