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

echo "[setup] Checking dependencies"
if python -c "import yaml, pandas, sklearn, pyarrow" 2>/dev/null; then
    echo "[setup] All dependencies already installed"
else
    echo "[setup] Installing basic dependencies"
    conda install -c conda-forge pyyaml pandas scikit-learn pyarrow -y
fi

echo "[setup] Cloning and installing biomlbench"
python "$COMPETITORS_DIR/scripts/setup_repo.py" --config "$CONFIG"

echo "[setup] Setting up Agentomics tasks"
python "$COMPETITORS_DIR/scripts/setup_tasks.py"

echo "[setup] Building Docker images (this will take a while)..."
cd "$COMPETITORS_DIR/biomlbench"

# Build base environment first if it doesn't exist
if ! docker images biomlbench-env | grep -q biomlbench-env; then
    echo "[setup] Building base environment..."
    bash scripts/build_base_env.sh
fi

# Build AIDE agent image
echo "[setup] Building AIDE agent image..."
bash scripts/build_agent.sh aide

# Build BioMNI agent image
echo "[setup] Building BioMNI agent image..."
bash scripts/build_agent.sh biomni

# Build MLAgentBench image 
echo "[setup] Building MLAgentBench agent image..."
bash scripts/build_agent.sh mlagentbench

echo "[setup] Done! Activate the environment with: conda activate $ENV_NAME"
