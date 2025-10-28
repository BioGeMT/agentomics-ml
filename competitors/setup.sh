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

echo "[setup] Installing basic dependencies"
conda install -c conda-forge pyyaml pandas scikit-learn pyarrow -y

echo "[setup] Cloning and installing biomlbench"
python "$COMPETITORS_DIR/scripts/setup_repo.py" --config "$CONFIG"

echo "[setup] Setting up Agentomics tasks"
python "$COMPETITORS_DIR/scripts/setup_tasks.py" --config "$CONFIG"

echo "[setup] Done! Activate the environment with: conda activate $ENV_NAME"
