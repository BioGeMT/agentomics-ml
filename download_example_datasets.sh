#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$REPO_ROOT/datasets"
CACHE_DIR="$REPO_ROOT"
PASSTHROUGH_ARGS=()
ENV_NAME="agentomics-datasets"

env_has_dataset_dependencies() {
    conda run -n "$ENV_NAME" python -c "import pandas, genomic_benchmarks, miRBench" >/dev/null 2>&1
}

show_help() {
    cat <<'EOF'
Usage: agentomics-download-datasets [--datasets-dir PATH] [--cache-dir PATH]

Download the bundled example datasets using a dedicated conda environment with
the optional dataset dependencies installed.
EOF
}

for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        show_help
        exit 0
    fi
done

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --datasets-dir" >&2; exit 1; }
            DATASETS_DIR="$2"
            shift 2
            ;;
        --cache-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --cache-dir" >&2; exit 1; }
            CACHE_DIR="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda env create -f "$REPO_ROOT/environment_datasets.yaml"
elif env_has_dataset_dependencies; then
    echo "Reusing existing conda environment: $ENV_NAME"
    echo "Dataset dependencies already installed; skipping environment update"
else
    echo "Reusing existing conda environment: $ENV_NAME"
    echo "Updating $ENV_NAME to ensure dataset dependencies are installed"
    conda env update -n "$ENV_NAME" -f "$REPO_ROOT/environment_datasets.yaml"
fi

echo "Downloading example datasets"
conda run --no-capture-output -n "$ENV_NAME" env PYTHONPATH="$REPO_ROOT/src" \
    python -m agentomics.utils.create_datasets \
    --datasets-dir "$DATASETS_DIR" \
    --cache-dir "$CACHE_DIR" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
