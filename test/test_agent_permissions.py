from test.utils_test import BaseAgentTest
from pathlib import Path
import subprocess

class TestAgentPermissions(BaseAgentTest):
    """Test suite for agent isolation and security."""

    def test_current_working_directory(self):
        """Test that the agent's current working directory is its own workspace."""
        result = self.bash_tool.function("pwd")
        expected_dir = f"{self.config.runs_dir}/{self.agent_id}"
        self.assertIn(expected_dir, result.strip(), f"Agent's working directory should be {expected_dir}, got: {result.strip()}")

    def test_agent_directory_access(self):
        """Test that agent can access only its own work directory."""

        result = self.bash_tool.function(f"ls -la {self.config.runs_dir}/{self.agent_id}/")
        self.assertNotIn("Permission denied", result, "Agent should access its own directory")
        self.assertNotIn("Command failed", result, "ls command should succeed")

        result = self.bash_tool.function(f"touch {self.config.runs_dir}/{self.agent_id}/test_file.txt")
        self.assertNotIn("Permission denied", result, "Agent should create files in its directory")
        self.assertNotIn("Command failed", result, "touch command should succeed")
    
    def test_cross_agent_isolation(self):
        """Test that workspace contains only the current agent's directory."""
        
        result = self.bash_tool.function(f"ls -1 {self.config.runs_dir}/")
        self.assertNotIn("Command failed", result, "Should be able to list runs directory")
        directories = [d.strip() for d in result.strip().split('\n') if d.strip() and not d.strip().startswith('[Tool call')]

        self.assertEqual(len(directories), 1, f"Expected exactly 1 directory, found {len(directories)}: {directories}")
        self.assertEqual(directories[0], self.agent_id, f"Expected only {self.agent_id} directory, found: {directories}")

    def test_protection_test_dataset(self):
        """Test that agent cannot access test datasets."""
        
        result = self.bash_tool.function(f"touch {self.config.agent_dataset_dir}/test_write.txt 2>&1")
        self.assertTrue("Permission denied" or "Command failed" in result, "Agent should not write to datasets directory")
        
        result = self.bash_tool.function(f"head -5 {self.config.agent_dataset_dir}/{self.config.dataset}/test.csv 2>&1")
        self.assertTrue(
            "Permission denied" in result or "Command failed" in result or "No such file" in result,
            "Agent should not read test data content"
        )

    def test_agent_dataset_access_permissions(self):
        """Test agent's dataset access from workspace."""
        
        result = self.bash_tool.function(f"ls -la {self.config.agent_dataset_dir}/")
        self.assertNotIn("Permission denied", result, "Agent should access its dataset directory")
        self.assertNotIn("Command failed", result, "ls command should succeed")

        result = self.bash_tool.function(f"head -5 {self.config.agent_dataset_dir}/train.csv 2>&1")
        self.assertNotIn("Permission denied", result, "Agent should read its dataset content")
        self.assertNotIn("Command failed", result, "head command should succeed")
    
    def test_agent_access_to_datasets(self):
        """Test that agent can access all files in prepared_datasets and not the ones in prepared_test_sets."""
        for file in self.config.prepared_dataset_dir.iterdir():
            result = self.bash_tool.function(f"head -5 {self.config.prepared_dataset_dir}/{file.name} 2>&1")
            self.assertNotIn("Permission denied", result, f"Agent should access the {file.name} file in prepared_datasets")

        test_set_dir = self.config.prepared_test_set_dir
        result = self.bash_tool.function(f"ls {test_set_dir} 2>&1")
        self.assertIn("No such file or directory", result, "Agent should not have access to prepared_test_sets directory")

    def test_agent_access_to_repository_datasets(self):
        self.assertFalse(Path("/repository/datasets/").is_dir())

    def test_api_key_protection(self):
        """Test that agent cannot access API keys."""
        
        # Check if agent can see environment variables with API keys
        env_result = self.bash_tool.function("env | grep -i 'api\\|key' 2>&1")
        self.assertIn("Command failed", env_result, f"Agent should not see any API/key environment variables. Found: {env_result}")

        self.assertNotIn("Command failed", subprocess.run(["env", "|", "grep", "-i", "'api\\|key'", "2>&1"], shell=True, capture_output=True, text=True).stdout, "root should see environment variables")

        # Check .env file access
        env_file_result = self.bash_tool.function("cat /repository/.env 2>&1")
        self.assertTrue(
            "Permission denied" or "No such file or directory" in env_file_result,
            f"Should not access .env file. Got: {env_file_result}"
        )
    
    def test_conda_environment_isolation(self):
        """Test that conda environment is set up and isolated."""
        which_python = self.bash_tool.function("which python")
        self.assertNotIn("Command failed", which_python, "'which python' should succeed")
        self.assertIn(f"{self.config.runs_dir}/{self.agent_id}/.conda/envs/{self.agent_id}_env/bin/python", which_python.strip(),
                      "Python should be from the agent's conda environment")
        
        install_result = self.bash_tool.function("conda install matplotlib -y 2>&1")
        self.assertNotIn("Command failed", install_result, "Conda install should succeed")
        self.assertNotIn("Permission denied", install_result, "Conda install should not have permission issues")
        
        python_result = self.bash_tool.function("python -c 'import matplotlib; print(f\"Matplotlib version: {matplotlib.__version__}\")' 2>&1")
        self.assertNotIn("Command failed", python_result, "Python command should succeed")
        self.assertIn("Matplotlib version:", python_result, "Should successfully import matplotlib")

    def test_agent_python_tools_security(self):
        """Test that Python tools see proper workspace isolation."""
        disruptive_code = f"""
import os
try:
    # Check that workspace contains only current agent directory
    run_dirs = os.listdir("{self.config.runs_dir}")
    print(f"Found directories: {{run_dirs}}")
    
    if len(run_dirs) == 1 and run_dirs[0] == "{self.agent_id}":
        print("Good: Workspace properly isolated - only current agent directory exists")
    else:
        print(f"SECURITY_ISSUE: Expected only [{self.agent_id}], found: {{run_dirs}}")
    
    # Test that agent can access its own directory
    agent_dir = "{self.config.runs_dir}/{self.agent_id}"
    if os.path.exists(agent_dir) and os.access(agent_dir, os.R_OK | os.W_OK):
        print("Good: Agent can access its own directory")
    else:
        print("ISSUE: Agent cannot access its own directory")
except Exception as e:
    print(f"Error during security test: {{e}}")
    """

        test_file_path = self.config.runs_dir / self.config.agent_id / "security_test.py"
        write_result = self.write_python_tool.function(
            file_path=test_file_path,
            code=disruptive_code
        )
        self.assertNotIn("Command failed", write_result, "Should be able to write test file")

        run_result = self.run_python_tool.function(python_file_path=test_file_path)
        self.assertNotIn("SECURITY_BREACH", run_result, f"Agent performed disruptive action: {run_result}")
        self.assertIn("Good:", run_result, "Should show security restrictions are working")