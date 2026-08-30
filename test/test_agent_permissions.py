import os
import subprocess

from test.test_utils import BaseAgentTest
from agentomics.runtime.conda_utils import get_shared_environment_path
from agentomics.utils.config import Config

class TestAgentPermissions(BaseAgentTest):
    """Test suite for agent isolation and security."""

    def test_current_working_directory(self):
        """Test that the agent's current working directory is its own workspace."""
        result = self.bash_tool.function("pwd")
        expected_dir = str(self.config.current_step_dir)
        self.assertIn(expected_dir, result.strip(), f"Agent's working directory should be {expected_dir}, got: {result.strip()}")

    def test_agent_directory_access(self):
        """Test that agent can read its run directory and write inside current_step_dir."""

        result = self.bash_tool.function(f"ls -la {self.config.run_dir}/")
        self.assertNotIn("Permission denied", result, "Agent should be able to list its run directory")
        self.assertNotIn("Command failed", result, "ls command should succeed")

        result = self.bash_tool.function(f"touch {self.config.current_step_dir}/test_file.txt")
        self.assertNotIn("Permission denied", result, "Agent should create files in current_step_dir")
        self.assertNotIn("Command failed", result, "touch command should succeed")
    
    def test_cross_agent_isolation(self):
        """Test that workspace contains the refactored shared layout, not per-agent run directories."""
        
        result = self.bash_tool.function(f"ls -1 {self.config.workspace_dir}/")
        self.assertNotIn("Command failed", result, "Should be able to list workspace root")
        directories = [d.strip() for d in result.strip().split('\n') if d.strip() and not d.strip().startswith('[Tool call')]

        self.assertIn(Config.RUN_DIRNAME, directories, f"Expected workspace to contain {Config.RUN_DIRNAME}/, found: {directories}")
        self.assertNotIn(self.agent_id, directories, f"Did not expect legacy per-agent run directory in workspace root: {directories}")

    def test_protection_test_dataset(self):
        """Test that agent cannot access test datasets."""
        
        result = self.bash_tool.function(f"touch {self.config.dataset_dir}/test_write.txt 2>&1")
        self.assertTrue(
            "Permission denied" in result or "Command failed" in result,
            "Agent should not write to datasets directory",
        )
        
        result = self.bash_tool.function(f"ls {self.config.dataset_dir}/test/input 2>&1")
        self.assertTrue(
            "Permission denied" in result or "Command failed" in result or "No such file" in result,
            "Agent should not read test data content"
        )

    def test_agent_dataset_access_permissions(self):
        """Test agent's dataset access from workspace."""
        
        result = self.bash_tool.function(f"ls -la {self.config.dataset_dir}/")
        self.assertNotIn("Permission denied", result, "Agent should access its dataset directory")
        self.assertNotIn("Command failed", result, "ls command should succeed")

        result = self.bash_tool.function(f"ls {self.config.dataset_dir}/train/input 2>&1")
        self.assertNotIn("Permission denied", result, "Agent should read its dataset content")
        self.assertNotIn("Command failed", result, "ls command should succeed")
    
    def test_agent_access_to_datasets(self):
        """Agent reads every entry exposed in its dataset directory, and the
        co-located hidden ``test`` split is never handed to it.

        Production keeps the ``test`` split away from the agent by not mounting
        it (see DatasetMountContractTest); the agent-visible dataset directory
        must therefore never contain a ``test`` entry.
        """
        for entry in self.config.dataset_dir.iterdir():
            if entry.is_dir():
                result = self.bash_tool.function(f"ls {self.config.dataset_dir}/{entry.name} 2>&1")
            else:
                result = self.bash_tool.function(f"head -5 {self.config.dataset_dir}/{entry.name} 2>&1")
            self.assertNotIn("Permission denied", result, f"Agent should access the {entry.name} entry in its dataset")

        hidden_test_split = self.config.dataset_dir / "test"
        result = self.bash_tool.function(f"ls {hidden_test_split} 2>&1")
        self.assertIn(
            "No such file or directory",
            result,
            "The co-located test split must not be exposed to the agent",
        )

    def test_api_key_protection(self):
        """Test that agent cannot access API keys."""
        
        # Check if agent can see environment variables with API keys
        env_result = self.bash_tool.function("env | grep -i 'api\\|key' 2>&1")
        self.assertIn("Command failed", env_result, f"Agent should not see any API/key environment variables. Found: {env_result}")

        self.assertNotIn("Command failed", subprocess.run(["env", "|", "grep", "-i", "'api\\|key'", "2>&1"], shell=True, capture_output=True, text=True).stdout, "root should see environment variables")

        # Check .env file access
        env_file_result = self.bash_tool.function("cat /repository/.env 2>&1")
        self.assertTrue(
            "Permission denied" in env_file_result or "No such file or directory" in env_file_result,
            f"Should not access .env file. Got: {env_file_result}"
        )
    
    def test_conda_environment_isolation(self):
        """Test that conda environment is set up and isolated."""
        which_python = self.bash_tool.function("which python")
        self.assertNotIn("Command failed", which_python, "'which python' should succeed")
        expected_python = get_shared_environment_path(self.config) / "bin" / "python"
        self.assertIn(str(expected_python), which_python.strip(),
                      "Python should be from the agent's conda environment")
        
        install_result = self.bash_tool.function("conda install matplotlib -y 2>&1")
        self.assertNotIn("Command failed", install_result, "Conda install should succeed")
        self.assertNotIn("Permission denied", install_result, "Conda install should not have permission issues")
        
        python_result = self.bash_tool.function("python -c 'import matplotlib; print(f\"Matplotlib version: {matplotlib.__version__}\")' 2>&1")
        self.assertNotIn("Command failed", python_result, "Python command should succeed")
        self.assertIn("Matplotlib version:", python_result, "Should successfully import matplotlib")

    def test_agent_runs_as_restricted_user(self):
        """Test that agent tool subprocesses run as the restricted agent user, not root."""
        result = self.bash_tool.function("whoami")
        expected_user = os.getenv("AGENT_USER")
        self.assertIn(expected_user, result, f"Agent should run as {expected_user}, got: {result}")

    def test_cannot_write_to_current_iteration_dir(self):
        """Test that the agent cannot create files directly in current_iteration_dir.
        Only current_step_dir (chowned to the agent user) should be writable."""
        evil_path = self.config.current_iteration_dir / "evil.txt"
        result = self.bash_tool.function(f"touch {evil_path} 2>&1")
        self.assertIn("Permission denied", result,
                      f"Agent must not write directly to current_iteration_dir. Got: {result}")

    def test_can_write_to_current_step_dir(self):
        """Test that the agent can write inside current_step_dir (its designated workspace)."""
        test_path = self.config.current_step_dir / "sandbox_write_test.txt"
        result = self.bash_tool.function(f"touch {test_path} 2>&1")
        self.assertNotIn("Permission denied", result,
                         f"Agent should be able to write to current_step_dir. Got: {result}")
        self.assertNotIn("Command failed", result,
                         f"touch in current_step_dir should succeed. Got: {result}")

    def test_agent_python_tools_security(self):
        """Test that Python tools see proper workspace isolation."""
        disruptive_code = f"""
import os
try:
    workspace_entries = sorted(os.listdir("{self.config.workspace_dir}"))
    print(f"Found workspace entries: {{workspace_entries}}")

    if "{Config.RUN_DIRNAME}" in workspace_entries and "{self.agent_id}" not in workspace_entries:
        print("Good: Workspace uses shared run layout without legacy per-agent run directories")
    else:
        print(f"SECURITY_ISSUE: Unexpected workspace layout: {{workspace_entries}}")

    run_dir = "{self.config.run_dir}"
    if os.path.exists(run_dir) and os.access(run_dir, os.R_OK):
        print("Good: Agent can access the shared run directory")
    else:
        print("ISSUE: Agent cannot access the shared run directory")
except Exception as e:
    print(f"Error during security test: {{e}}")
    """

        test_file_path = self.config.current_step_dir / "security_test.py"
        write_result = self.write_python_tool.function(
            file_path=test_file_path,
            code=disruptive_code
        )
        self.assertNotIn("Command failed", write_result, "Should be able to write test file")

        run_result = self.run_python_tool.function(python_file_path=test_file_path)
        self.assertNotIn("SECURITY_BREACH", run_result, f"Agent performed disruptive action: {run_result}")
        self.assertIn("Good:", run_result, "Should show security restrictions are working")
