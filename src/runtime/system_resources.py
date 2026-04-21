from __future__ import annotations

import subprocess


def check_gpu_availability() -> str | None:
    try:
        result = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            return None

        gpu_names = []
        memory_lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()

        for index, line in enumerate(result.stdout.strip().splitlines()):
            gpu_name = line.split("(UUID:")[0].strip().split(":", 1)[1].strip()
            gpu_memory = memory_lines[index] if index < len(memory_lines) else "unknown"
            gpu_names.append(f"{gpu_name} (Memory: {gpu_memory} MB)")
        return ", ".join(gpu_names)
    except Exception:
        return None

def get_cpu_info() -> str:
    cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    for line in cpu_info.splitlines():
        if "CPU(s):" in line and "NUMA" not in line:
            return f"{line.split(':')[1].strip()} cores"
    return "CPU count not available"

def get_ram_info() -> str:
    mem_info = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    for line in mem_info.splitlines():
        if line.strip().startswith("Mem:"):
            return f"{line.split()[1]} MB"
    return "RAM info not available"

def get_resources_summary() -> str:
    gpu_info = check_gpu_availability()
    resources = []
    if gpu_info:
        resources.append(f"GPU Resource Available ({gpu_info})")
    else:
        resources.append("CPU Resources Only")
    resources.append(f"RAM: {get_ram_info()}")
    resources.append(f"CPU: {get_cpu_info()}")
    return ", ".join(resources)
