import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

WORKSPACE_DIR = "/workspace"
PREPARED_DATASETS_DIR = str(REPO_ROOT / "prepared_datasets")

from tools.tool_registry import create_tools
from utils.config import Config
from runtime.read_write_utils import (
    initialize_current_iteration_workspace,
    initialize_run_directories,
    save_config,
)

_shared_test_resources = None

def get_shared_test_resources():
    global _shared_test_resources

    agent_id = os.getenv('AGENT_ID')
    agent_user = os.getenv('AGENT_USER')

    if _shared_test_resources is None:
        config = Config(
            agent_id=agent_id,
            model_name="openai/gpt-3.5-turbo",
            iteration_plan_model_name="openai/gpt-3.5-turbo",
            dataset="AGO2_CLASH_Hejret2023",
            tags=[],
            val_metric="ACC",
            task_type="classification",
            input_structure=["data.csv"],
            workspace_dir=str(Path("/workspace").resolve()),
            datasets_dir=str(Path('../repository/datasets').resolve()),
            iterations=5,
            user_prompt="Create the best possible machine learning model that will generalize to new unseen data.",
            agent_user=agent_user,
        )

        initialize_run_directories(config)
        save_config(config)
        # Clean up leftover workspace from a previous failed setup attempt.
        if config.current_iteration_dir.exists():
            import shutil
            shutil.rmtree(config.current_iteration_dir)
        initialize_current_iteration_workspace(config)

        # Mirror what on_step_start does: create current_step_dir and assign it to
        # agent_user so the agent has a writable working directory during tests.
        config.current_step_dir.mkdir(exist_ok=True)
        if agent_user:
            subprocess.run(["chown", agent_user, str(config.current_step_dir)], check=True)

        print(f"Created shared test agent: {agent_id}")
        print("Setting up tools for testing (including conda env creation, might take a moment)\n")

        tools = create_tools(config, config.tool_ids)
        tools_by_name = {tool.name: tool for tool in tools}

        _shared_test_resources = {
            'config': config,
            'agent_id': agent_id,
            'bash_tool': tools_by_name['bash'],
            'write_python_tool': tools_by_name['write_python'],
            'run_python_tool': tools_by_name['run_python'],
            'replace_tool': tools_by_name['replace'],
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
        cls.replace_tool = resources['replace_tool']
