#!/usr/bin/env bash
set -euo pipefail

COMPETITORS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
CONFIG="$COMPETITORS_DIR/config.yaml"
ENV_NAME="biomlbench-agents"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "[setup] Conda environment '$ENV_NAME' already exists"
else
    echo "[setup] Creating conda environment '$ENV_NAME' with Python 3.11"
    conda create -n "$ENV_NAME" python=3.11 -y
fi

echo "[setup] Activating environment"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[setup] Ensuring Python packages are installed"
conda install -c conda-forge "numpy<2" pyyaml pandas scikit-learn pyarrow wandb -y

echo "[setup] Cloning and installing biomlbench"
python "$COMPETITORS_DIR/scripts/setup_repo.py" --config "$CONFIG"

echo "[setup] Setting up Agentomics tasks"
python "$COMPETITORS_DIR/scripts/setup_tasks.py"

echo "[setup] Building Docker images (optional)..."
cd "$COMPETITORS_DIR/biomlbench"

echo "[setup] Building base environment..."
bash scripts/build_base_env.sh

echo "[setup] Building AIDE agent image..."
bash scripts/build_agent.sh aide

echo "[setup] Building BioMNI agent image..."
bash scripts/build_agent.sh biomni

echo "[setup] Building STELLA agent image..."
bash scripts/build_agent.sh stella

echo "[setup] Building 1-shot agent image..."
bash scripts/build_agent.sh oneshot

echo "[setup] Done! Activate the environment with: conda activate $ENV_NAME"
