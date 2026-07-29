import os
import unittest

from test.test_utils import BaseAgentTest


@unittest.skipIf(
    os.getenv("CUDA_VISIBLE_DEVICES") == "",
    "GPU access tests do not apply in CPU-only mode",
)
class TestGpuAccess(BaseAgentTest):
    """Test suite for GPU agent access"""

    def _tool_file_path(self, filename: str):
        return self.config.current_step_dir / filename

    def test_gpu_access_bash(self):
        """Test if the agent can access the GPU using bash tool."""

        result = self.bash_tool.function("nvidia-smi")
        self.assertIn("NVIDIA-SMI", result, "GPU access failed in bash tool")

        gpu_list_result = self.bash_tool.function("nvidia-smi -L") #should show at least one GPU
        self.assertIn("GPU 0:", gpu_list_result, "No GPU devices found")

    def test_gpu_pytorch_python(self):
        """Test if the agent can access the GPU using python tool (PyTorch)."""

        print("Installing PyTorch, might take a while...")
        install_result = self.bash_tool.function(
            "pip install torch && echo TORCH_INSTALL_SUCCEEDED"
        )
        self.assertIn(
            "TORCH_INSTALL_SUCCEEDED",
            install_result,
            "Failed to install PyTorch through pip",
        )
        
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
        
        file_path = self._tool_file_path("test_pytorch.py")
        write_result = self.write_python_tool.function(file_path=file_path, code=code)
        self.assertNotIn("Command failed", write_result, "Should be able to write test file")
        self.assertNotIn("Error:", write_result, "Should be able to write test file")

        run_result = self.run_python_tool.function(python_file_path=file_path)
        self.assertIn("CUDA is available", run_result, "GPU access failed in python tool")
        self.assertIn("GPU Name:", run_result, "Failed to retrieve GPU name in python tool")
