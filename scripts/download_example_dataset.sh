#!/usr/bin/env bash
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR" || exit 1

for arg in "$@"; do
    if [[ "$arg" == "--list" ]]; then
        PYTHONPATH="$(pwd)/src" python src/datasets/create_datasets.py --list
        exit $?
    fi
done

conda env remove -n agentomics-datasets -y 2>/dev/null || true
conda env create -f envs/environment_datasets.yaml
PYTHONPATH="$(pwd)/src" conda run -n agentomics-datasets python src/datasets/create_datasets.py "$@"
