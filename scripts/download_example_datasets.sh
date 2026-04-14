#!/usr/bin/env bash
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR" || exit 1
conda env remove -n agentomics-datasets -y 2>/dev/null || true
conda env create -f envs/environment_datasets.yaml
conda run -n agentomics-datasets python src/datasets/create_datasets.py