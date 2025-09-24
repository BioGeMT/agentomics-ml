import unittest
from pathlib import Path

from src.tools.bash_tool import create_bash_tool
from src.tools.write_python_tool import create_write_python_tool
from src.tools.run_python_tool import create_run_python_tool
from src.utils.create_user import create_new_user_and_rundir
from src.utils.config import Config
from src.utils.workspace_setup import ensure_workspace_folders

class TestGpuAccess(unittest.TestCase):
    """Test suite for GPU agent access"""

    @classmethod
    def setUpClass(cls):
        cls.config = Config(
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
        
        ensure_workspace_folders(cls.config)
        cls.agent_id = create_new_user_and_rundir(cls.config)
        cls.config.agent_id = cls.agent_id
        print(f"Created test agent: {cls.agent_id}")
        
        print("Setting up tools for testing (including conda env creation, might take a moment)\n")
        
        cls.bash_tool = create_bash_tool(
            agent_id=cls.agent_id,
            runs_dir=cls.config.runs_dir,
            timeout=10*60,
            max_retries=cls.config.max_tool_retries,
            autoconda=True,
            proxy=cls.config.use_proxy
        )
        
        cls.write_python_tool = create_write_python_tool(
            agent_id=cls.agent_id,
            runs_dir=cls.config.runs_dir,
            max_retries=cls.config.max_tool_retries
        )
        
        cls.run_python_tool = create_run_python_tool(
            agent_id=cls.agent_id,
            runs_dir=cls.config.runs_dir,
            timeout=cls.config.run_python_tool_timeout,
            proxy=cls.config.use_proxy,
            max_retries=cls.config.max_tool_retries
        )

    def test_gpu_access_bash(self):
        """Test if the agent can access the GPU using bash tool."""

        result = self.bash_tool.function("nvidia-smi")
        self.assertIn("NVIDIA-SMI", result, "GPU access failed in bash tool")

        gpu_list_result = self.bash_tool.function("nvidia-smi -L") #should show at least one GPU
        self.assertIn("GPU 0:", gpu_list_result, "No GPU devices found")

    def test_gpu_pytorch_python(self):
        """Test if the agent can access the GPU using python tool (PyTorch)."""

        print("Installing PyTorch with CUDA support, might take a while...")
        install_result = self.bash_tool.function("pip install torch")
        self.assertNotIn("ERROR", install_result, "Failed to install PyTorch through pip")
        
        code = (
            "import torch\n"
            "if torch.cuda.is_available():\n"
            "    print('CUDA is available')\n"
            "    print(f'GPU count: {torch.cuda.device_count()}')\n"
            "    print(f'Current GPU: {torch.cuda.current_device()}')\n"
            "    print(f'GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}')\n"
            "else:\n"
            "    print('CUDA is not available')\n"
        )
        
        file_path = self.config.runs_dir / self.config.agent_id / "test_pytorch.py"
        write_result = self.write_python_tool.function(file_path=file_path, code=code)
        self.assertNotIn("Command failed", write_result, "Should be able to write test file")

        run_result = self.run_python_tool.function(python_file_path=file_path)
        self.assertIn("CUDA is available", run_result, "GPU access failed in python tool")
        self.assertIn("GPU Name:", run_result, "Failed to retrieve GPU name in python tool")

    def test_gpu_tensorflow_python(self):
        """Test if the agent can access the GPU using python tool (Tensorflow)."""

        print("Installing tensorflow with CUDA support, might take a while...")
        install_result = self.bash_tool.function("pip install tensorflow[and-cuda]") #without [and-cuda] would not detect GPU
        self.assertNotIn("ERROR", install_result, "Failed to install tensorflow through pip")

        code = (
          "import tensorflow as tf\n"
          "gpus = tf.config.list_physical_devices('GPU')\n"
          "print(f'GPU devices found: {len(gpus)}')\n"
          "if not gpus:\n"
          "    print('No GPU devices found for TensorFlow')\n"
      )
        
        file_path = self.config.runs_dir / self.config.agent_id / "test_tensorflow.py"
        write_result = self.write_python_tool.function(file_path=file_path, code=code)
        self.assertNotIn("Command failed", write_result, "Should be able to write TensorFlow test file")

        run_result = self.run_python_tool.function(python_file_path=file_path)
        print(run_result)
        self.assertNotIn("GPU devices found: 0", run_result, "No GPU devices found for TensorFlow")

def run_gpu_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGpuAccess)

    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == "__main__":
    run_gpu_tests()
