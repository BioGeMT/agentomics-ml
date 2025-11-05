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
    split_allowed_iterations: int
    time_deadline: Optional[int] = None

    wandb_run_id: Optional[str] = None
    # static defaults
    temperature: float = 1.0
    max_steps: int = 100 #TODO rename, this is per-step limit
    max_validation_retries: int = 5
    use_proxy: bool = True
    llm_response_timeout: int = 60 * 15
    bash_tool_timeout: int = 60 * 5
    write_python_tool_timeout: int = 60 * 1
    run_python_tool_timeout: int = 60 * 60 * 6 #This affects max training time
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
        split_allowed_iterations: int = 1,
        time_deadline: Optional[int] = None,
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
        self.fallbacks_dir = workspace_dir / "fallbacks"
        self.reports_dir = workspace_dir / "reports"
        self.iterations = iterations
        self.task_type = get_task_type_from_prepared_dataset(prepared_datasets_dir / dataset)
        self.user_prompt = user_prompt
        self.explicit_valid_set_provided = (agent_datasets_dir / dataset / "validation.csv").exists()
        self.split_allowed_iterations = split_allowed_iterations if not self.explicit_valid_set_provided else 0
        self.time_deadline = time_deadline
        self.foundational_model_to_desc = {} #TODO implement
        
        if max_steps is not None:
            self.max_steps = max_steps

    def can_iteration_split_data(self, iteration):
        return not self.explicit_valid_set_provided and iteration < self.split_allowed_iterations

    def check_gpu_availability(self) -> Optional[str]:
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    line = line.split('(UUID:')[0].strip() # remove UUID part
                    line = line.split(':', 1)[1].strip() # get only device name
                    gpu_max_memory = subprocess.run(
                        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True
                    ).stdout.strip().split('\n')[len(gpus)-1]
                    gpus.append(f'{line} (Memory: {gpu_max_memory} MB)')
                return ', '.join(gpus)
            return None
        except:
            return None
    
    def get_cpu_info(self):
        cpu_info = subprocess.run(['lscpu'], capture_output=True, text=True).stdout
        cpu_count = 'CPU count not available'
        for line in cpu_info.split('\n'):
            if 'CPU(s):' in line and 'NUMA' not in line:
                cpu_count = line.split(':')[1].strip() + ' cores'
                break
        return cpu_count
    
    def get_ram_info(self):
        mem_info = subprocess.run(['free', '-m'], capture_output=True, text=True).stdout
        for line in mem_info.splitlines():
            if line.strip().startswith('Mem:'):
                return line.split()[1] + ' MB'  # Return total memory
        return 'RAM info not available'
    
    def get_resources_summary(self):
        gpu_info = self.check_gpu_availability()
        resources = []
        if gpu_info:
            resources.append(f'GPU Resource Available ({gpu_info})')
        else:
            resources.append('CPU Resources Only')
        resources.append(f'RAM: {self.get_ram_info()}')
        resources.append(f'CPU: {self.get_cpu_info()}')
        return ', '.join(resources)


    def print_summary(self):
        print('=== AGENTOMICS CONFIGURATION ===')
        print('MAIN MODEL:', self.model_name)
        print('FEEDBACK MODEL:', self.feedback_model_name)
        print('DATASET:', self.dataset)
        print('TASK TYPE:', self.task_type)
        print('VAL METRIC:', self.val_metric)
        print('AGENT ID:', self.agent_id)
        print('ITERATIONS:', self.iterations)
        print('SPLIT ALLOWED ITERATIONS:', self.split_allowed_iterations)
        print('USER PROMPT:', self.user_prompt)
        print('RESOURCES SUMMARY:', self.get_resources_summary())
        print('===============================')
