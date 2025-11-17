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

echo "[setup] Building Docker images (fresh builds)..."
cd "$COMPETITORS_DIR/biomlbench"

echo "[setup] Removing old images to ensure fresh builds..."
# Remove old local tags if they exist (ignore errors if they don't)
docker rmi biomlbench-env:latest 2>/dev/null || true
docker rmi aide:latest 2>/dev/null || true
docker rmi biomni:latest 2>/dev/null || true
docker rmi stella:latest 2>/dev/null || true
docker rmi oneshot:latest 2>/dev/null || true

# Remove millerh1 tags to prevent Docker from reusing pulled images
# These tags share the same image ID as local tags, so removing local tags
# doesn't delete the images - we need to remove millerh1 tags too
docker rmi millerh1/biomlbench-env:v0.1a 2>/dev/null || true
docker rmi millerh1/aide:v0.1a 2>/dev/null || true
docker rmi millerh1/biomni:v0.1a 2>/dev/null || true
docker rmi millerh1/stella:v0.1a 2>/dev/null || true
docker rmi millerh1/oneshot:v0.1a 2>/dev/null || true
docker rmi millerh1/dummy:v0.1a 2>/dev/null || true
docker rmi millerh1/mlagentbench:v0.1a 2>/dev/null || true

echo "[setup] Ensuring base image (ubuntu:22.04) exists..."
# Ubuntu is a standard base image - pull it once if it doesn't exist
# This is the ONLY image we allow to be pulled (it's not a custom image)
if ! docker images ubuntu:22.04 | grep -q ubuntu; then
    echo "[setup] Pulling ubuntu:22.04 (standard base image)..."
    docker pull ubuntu:22.04
fi

echo "[setup] Building base environment (fresh build)..."
bash scripts/build_base_env.sh --force

echo "[setup] Building AIDE agent image (fresh build)..."
bash scripts/build_agent.sh --force aide

echo "[setup] Building BioMNI agent image (fresh build)..."
bash scripts/build_agent.sh --force biomni

echo "[setup] Building STELLA agent image (fresh build)..."
bash scripts/build_agent.sh --force stella

echo "[setup] Building 1-shot agent image (fresh build)..."
bash scripts/build_agent.sh --force oneshot

echo "[setup] Done! Activate the environment with: conda activate $ENV_NAME"
