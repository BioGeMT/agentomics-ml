#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${AGENTOMICS_DATASETS_ENV_NAME:-agentomics-datasets}"
cd "$REPO_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda is required to prepare example datasets." >&2
    echo "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

if ! conda env list | grep -Eq "^${ENV_NAME}[[:space:]]"; then
    echo "Creating Conda environment '${ENV_NAME}' for example datasets..."
    conda env create -n "$ENV_NAME" -f envs/environment_datasets.yaml
else
    echo "Synchronizing Conda environment '${ENV_NAME}'..."
    conda env update -n "$ENV_NAME" -f envs/environment_datasets.yaml --prune
fi

PYTHONNOUSERSITE=1 PYTHONPATH="$(pwd)/src" \
    conda run -n "$ENV_NAME" python src/datasets/create_datasets.py "$@"
