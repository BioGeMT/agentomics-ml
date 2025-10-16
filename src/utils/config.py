from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import List, Optional
from .dataset_utils import get_task_type_from_prepared_dataset

@dataclass
class Config:
    # defined at runtime
    agent_id: str
    model_name: str
    feedback_model_name: str
    dataset: str
    tags: List[str]
    val_metric: str
    prepared_dataset_dir: Path
    prepared_test_set_dir: Path
    agent_dataset_dir: Path
    workspace_dir: Path
    snapshots_dir: Path
    runs_dir: Path
    reports_dir: Path
    iterations: int
    task_type: str
    user_prompt: str

    wandb_run_id: Optional[str] = None
    # static defaults
    temperature: float = 1.0
    max_steps: int = 100 #TODO rename, this is per-step limit
    max_run_retries: int = 1
    max_validation_retries: int = 5
    use_proxy: bool = True
    llm_response_timeout: int = 60 * 15
    bash_tool_timeout: int = 60 * 15
    write_python_tool_timeout: int = 60 * 1
    run_python_tool_timeout: int = 60 * 60 * 15 #This affects max training time
    credit_budget: int = 30 # Only applies when using a provisioning openrouter key #TODO
    max_tool_retries: int = 5

    def __init__(
        self,
        agent_id: str,
        model_name: str,
        feedback_model_name: str,
        dataset: str,
        tags: List[str],
        val_metric: str,
        workspace_dir: Path,
        prepared_datasets_dir: Path,
        prepared_test_sets_dir: Path,
        agent_datasets_dir: Path,
        user_prompt: str,
        max_steps: Optional[int] = None,
        iterations: Optional[int] = 5,
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.feedback_model_name = feedback_model_name
        self.dataset = dataset
        self.tags = tags
        self.val_metric = val_metric
        self.prepared_dataset_dir = prepared_datasets_dir / dataset
        self.prepared_test_set_dir = prepared_test_sets_dir / dataset
        self.agent_dataset_dir = agent_datasets_dir / dataset
        self.workspace_dir = workspace_dir
        self.runs_dir = workspace_dir / "runs"
        self.snapshots_dir = workspace_dir / "snapshots"
        self.reports_dir = workspace_dir / "reports"
        self.iterations = iterations
        self.task_type = get_task_type_from_prepared_dataset(prepared_datasets_dir / dataset)
        self.user_prompt = user_prompt
        self.explicit_valid_set_provided = (agent_datasets_dir / dataset / "validation.csv").exists()

        if max_steps is not None:
            self.max_steps = max_steps

    def _check_gpu_availability(self) -> Optional[str]:
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    line = line.split('(UUID:')[0].strip() # remove UUID part
                    line = line.split(':', 1)[1].strip() # get only device name
                    gpus.append(line)
                return ', '.join(gpus)
            return None
        except:
            return None

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

        gpu_info = self._check_gpu_availability()
        if gpu_info:
            print(f'GPU: Available ({gpu_info})')
        else:
            print('GPU: Not available')

        print('===============================')
