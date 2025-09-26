import unittest
from pathlib import Path
import dotenv

from src.tools.bash_tool import create_bash_tool
from src.tools.write_python_tool import create_write_python_tool
from src.tools.run_python_tool import create_run_python_tool
from src.utils.create_user import create_new_user_and_rundir
from src.utils.config import Config
from src.utils.workspace_setup import ensure_workspace_folders
from src.utils.dataset_utils import setup_nonsensitive_dataset_files_for_agent

_shared_test_resources = None

def get_shared_test_resources():
    global _shared_test_resources

    if _shared_test_resources is None:
        config = Config(
              model_name="openai/gpt-3.5-turbo",
              feedback_model_name="openai/gpt-3.5-turbo",
              dataset="AGO2_CLASH_Hejret",
              tags=[],
              val_metric="ACC",
              root_privileges=True,
              workspace_dir=Path("/workspace").resolve(),
              prepared_datasets_dir=Path('../repository/prepared_datasets').resolve(),
              agent_dataset_dir=Path('../workspace/datasets').resolve(),
              iterations=5,
              user_prompt="Create the best possible machine learning model that will generalize to new unseen data."
          )
        
        dotenv.load_dotenv()

        setup_nonsensitive_dataset_files_for_agent(
            prepared_datasets_dir=config.prepared_dataset_dir.parent,
            agent_datasets_dir=config.agent_dataset_dir.parent,
            dataset_name=config.dataset,
        )
        ensure_workspace_folders(config)
        agent_id = create_new_user_and_rundir(config)
        config.agent_id = agent_id
        print(f"Created shared test agent: {agent_id}")

        print("Setting up tools for testing (including conda env creation, might take a moment)\n")

        bash_tool = create_bash_tool(
            agent_id=agent_id,
            runs_dir=config.runs_dir,
            timeout=10*60,
            max_retries=config.max_tool_retries,
            autoconda=True,
            proxy=config.use_proxy
        )

        write_python_tool = create_write_python_tool(
            agent_id=agent_id,
            runs_dir=config.runs_dir,
            max_retries=config.max_tool_retries
        )

        run_python_tool = create_run_python_tool(
            agent_id=agent_id,
            runs_dir=config.runs_dir,
            timeout=config.run_python_tool_timeout,
            proxy=config.use_proxy,
            max_retries=config.max_tool_retries
        )

        _shared_test_resources = {
            'config': config,
            'agent_id': agent_id,
            'bash_tool': bash_tool,
            'write_python_tool': write_python_tool,
            'run_python_tool': run_python_tool
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