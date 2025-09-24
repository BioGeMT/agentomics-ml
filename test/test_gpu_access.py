from test.utils_test import BaseAgentTest

class TestGpuAccess(BaseAgentTest):
    """Test suite for GPU agent access"""

    def test_gpu_access_bash(self):
        """Test if the agent can access the GPU using bash tool."""

        result = self.bash_tool.function("nvidia-smi")
        self.assertIn("NVIDIA-SMI", result, "GPU access failed in bash tool")

        gpu_list_result = self.bash_tool.function("nvidia-smi -L") #should show at least one GPU
        self.assertIn("GPU 0:", gpu_list_result, "No GPU devices found")

    def test_gpu_pytorch_python(self):
        """Test if the agent can access the GPU using python tool (PyTorch)."""

        print("Installing PyTorch, might take a while...")
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
        self.assertNotIn("GPU devices found: 0", run_result, "No GPU devices found for TensorFlow")