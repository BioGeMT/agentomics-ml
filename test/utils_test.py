import unittest
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.tools.setup_tools import create_tools
from src.utils.create_user import create_run_and_snapshot_dirs
from src.utils.config import Config
from src.utils.workspace_setup import ensure_workspace_folders
from src.utils.dataset_utils import setup_nonsensitive_dataset_files_for_agent

_shared_test_resources = None

def check_foundation_model_gpu_usage(run_result: str, model: str) -> bool:
    if "using device: cuda" in run_result.lower() and "cpu" not in run_result.lower():
        print(f"\n{model} test used GPU")
    else:
        print(f"\nWARNING: {model} test ran on CPU (no GPU detected)")

def get_shared_test_resources():
    global _shared_test_resources

    agent_id = os.getenv('AGENT_ID')

    if _shared_test_resources is None:
        config = Config(
              agent_id=agent_id,
              model_name="openai/gpt-3.5-turbo",
              feedback_model_name="openai/gpt-3.5-turbo",
              dataset="AGO2_CLASH_Hejret",
              tags=[],
              val_metric="ACC",
              workspace_dir=Path("/workspace").resolve(),
              prepared_datasets_dir=Path('../repository/prepared_datasets').resolve(),
              prepared_test_sets_dir=Path('../repository/prepared_test_sets').resolve(),
              agent_datasets_dir=Path('../workspace/datasets').resolve(),
              iterations=5,
              user_prompt="Create the best possible machine learning model that will generalize to new unseen data."
          )
        
        setup_nonsensitive_dataset_files_for_agent(
            prepared_datasets_dir=config.prepared_dataset_dir.parent,
            agent_datasets_dir=config.agent_dataset_dir.parent,
            dataset_name=config.dataset,
        )
        ensure_workspace_folders(config)
        create_run_and_snapshot_dirs(config)
        print(f"Created shared test agent: {agent_id}")

        print("Setting up tools for testing (including conda env creation, might take a moment)\n")

        bash_tool, write_python_tool, run_python_tool, foundation_models_info, replace_tool = create_tools(config)

        _shared_test_resources = {
            'config': config,
            'agent_id': agent_id,
            'bash_tool': bash_tool,
            'write_python_tool': write_python_tool,
            'run_python_tool': run_python_tool,
            'foundation_models_info_tool': foundation_models_info,
            'replace_tool': replace_tool
        }

    return _shared_test_resources


class BaseAgentTest(unittest.TestCase):
    """Base test class with common setup for agent tests"""

    @classmethod
    def setUpClass(cls):
        resources = get_shared_test_resources()

        cls.config = resources['config']
        cls.agent_id = resources['agent_id']
        cls.bash_tool = resources['bash_tool']
        cls.write_python_tool = resources['write_python_tool']
        cls.run_python_tool = resources['run_python_tool']
        cls.foundation_models_info_tool = resources['foundation_models_info_tool']
        cls.replace_tool = resources['replace_tool']
