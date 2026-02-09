#!/usr/bin/env bash
set -euo pipefail

COMPETITORS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
CONFIG="$COMPETITORS_DIR/config.yaml"
ENV_NAME="biomlbench-agents"
TARGET_AGENT=""
ALL_AGENTS=(aide biomni stella zeroshot)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            TARGET_AGENT="$2"
            shift 2
            ;;
        *)
            echo "[setup] Unknown argument: $1 (only --name <agent> is supported)"
            exit 1
            ;;
    esac
done

if [[ -n "$TARGET_AGENT" ]]; then
    case "$TARGET_AGENT" in
        aide|biomni|stella|zeroshot)
            BUILD_AGENTS=("$TARGET_AGENT")
            ;;
        *)
            echo "[setup] Invalid --name '$TARGET_AGENT'. Allowed: aide, biomni, stella, zeroshot"
            exit 1
            ;;
    esac
else
    BUILD_AGENTS=("${ALL_AGENTS[@]}")
fi

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
# Remove old local tags if they exist (ignore errors if they don't).
# Base env is always required.
docker rmi biomlbench-env:latest 2>/dev/null || true
for agent in "${BUILD_AGENTS[@]}"; do
    docker rmi "${agent}:latest" 2>/dev/null || true
done

# Remove millerh1 tags to prevent Docker from reusing pulled images
# These tags share the same image ID as local tags, so removing local tags
# doesn't delete the images - we need to remove millerh1 tags too
docker rmi millerh1/biomlbench-env:v0.1a 2>/dev/null || true
for agent in "${BUILD_AGENTS[@]}"; do
    docker rmi "millerh1/${agent}:v0.1a" 2>/dev/null || true
done

echo "[setup] Ensuring base image (ubuntu:22.04) exists..."
# Ubuntu is a standard base image - pull it once if it doesn't exist
# This is the ONLY image we allow to be pulled (it's not a custom image)
if ! docker images ubuntu:22.04 | grep -q ubuntu; then
    echo "[setup] Pulling ubuntu:22.04 (standard base image)..."
    docker pull ubuntu:22.04
fi

echo "[setup] Building base environment (fresh build)..."
bash scripts/build_base_env.sh --force

for agent in "${BUILD_AGENTS[@]}"; do
    echo "[setup] Building ${agent} agent image (fresh build)..."
    bash scripts/build_agent.sh --force "$agent"
done

echo "[setup] Done! Built agents: ${BUILD_AGENTS[*]}"
echo "[setup] Activate the environment with: conda activate $ENV_NAME"
