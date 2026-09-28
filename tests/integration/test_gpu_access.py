import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    reason="CPU-only execution was requested",
)


def test_agent_bash_tool_can_access_gpu(bash_tool):
    summary = bash_tool.function("nvidia-smi")
    devices = bash_tool.function("nvidia-smi -L")

    assert "NVIDIA-SMI" in summary
    assert "GPU 0:" in devices


def test_agent_python_tool_can_access_gpu(
    initialized_run_config,
    run_python_tool,
):
    script_path = Path(initialized_run_config.current_step_dir) / "inspect_gpu.py"
    script_path.write_text(
        "import torch\n"
        "print(f'available={torch.cuda.is_available()}')\n"
        "print(f'devices={torch.cuda.device_count()}')\n"
        "print(f'name={torch.cuda.get_device_name(0)}')\n",
        encoding="utf-8",
    )

    result = run_python_tool.function(python_file_path=script_path)

    assert "available=True" in result
    assert "devices=0" not in result
    assert "name=" in result
