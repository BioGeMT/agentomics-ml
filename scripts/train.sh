#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bash_helpers.sh"

CPU_ONLY=false
ARGS=()

show_help() {
    echo "Usage: $0 --agent-dir <agent_folder_path> --dataset-dir <dataset_dir> --artifacts-dir <artifacts_dir_path> [--cpu-only]"
    echo "Options:"
    echo "  --agent-dir       Path to agent folder (required)"
    echo "  --dataset-dir     Path to a dataset folder with train/validation splits (split folders with input/+labels.csv, or train.csv/validation.csv + metadata.json). Prepared into the run's format (numeric labels via the run's trained mapping) before training (required)"
    echo "  --artifacts-dir   Path to directory where training artifacts will be saved (required)"
    echo "  --label-col       Label column name for CSV-form --dataset-dir (overrides metadata.json). Not needed for folder splits or when metadata.json declares it (optional)"
    echo "  --code-path       Path to code files, points to best_iteration_snapshot by default, must be relative to --agent-dir and a child of --agent-dir (optional)"
    echo "  --cpu-only        Run without GPU (optional)"
    echo "  --help            Show this help message and exit"
    exit 0
}

AGENT_DIR=""
DATASET_DIR=""
ARTIFACTS_DIR=""
LABEL_COL=""

for arg in "$@"; do
    if [[ "$arg" == "--help" ]]; then
        show_help
        exit 0
    fi
done

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent-dir)
            require_opt_value "$1" "${2:-}"
            AGENT_DIR="$2"
            shift 2
            ;;
        --dataset-dir)
            require_opt_value "$1" "${2:-}"
            DATASET_DIR="$2"
            shift 2
            ;;
        --artifacts-dir)
            require_opt_value "$1" "${2:-}"
            ARTIFACTS_DIR="$2"
            shift 2
            ;;
        --label-col)
            require_opt_value "$1" "${2:-}"
            LABEL_COL="$2"
            shift 2
            ;;
        --code-path)
            require_opt_value "$1" "${2:-}"
            CODE_PATH="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# ensure all required args are provided
if [[ -z "$AGENT_DIR" ]]; then
      die "Missing required argument: --agent-dir. Run '$0 --help' for usage"
fi
if [[ -z "$DATASET_DIR" ]]; then
    die "Missing required argument: --dataset-dir. Run '$0 --help' for usage"
fi
if [[ -z "$ARTIFACTS_DIR" ]]; then
    die "Missing required argument: --artifacts-dir. Run '$0 --help' for usage"
fi

[[ -d "$AGENT_DIR" ]] || die "--agent-dir does not exist: $AGENT_DIR"
[[ -d "$DATASET_DIR" ]] || die "--dataset-dir must be a directory: $DATASET_DIR"
[[ -d "$(dirname "$ARTIFACTS_DIR")" ]] || die "--artifacts-dir parent directory does not exist: $(dirname "$ARTIFACTS_DIR")"

AGENT_DIR="$(cd "$AGENT_DIR" && pwd)"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
ARTIFACTS_DIR="$(cd "$(dirname "$ARTIFACTS_DIR")" && pwd)/$(basename "$ARTIFACTS_DIR")"

CODE_PATH=${CODE_PATH:-"best_iteration_snapshot"}
CODE_ROOT="${AGENT_DIR}/${CODE_PATH}"
echo "Using code path: $CODE_PATH"
ENV_DIR="${CODE_ROOT}/.conda/envs"
TRAIN_PATH="${CODE_ROOT}/model_training/train.py"
TRAIN_WORKDIR="$(dirname "$TRAIN_PATH")"
DESCRIPTOR_PATH="${CODE_ROOT}/environment.yml"

[[ -f "$TRAIN_PATH" ]] || die "train.py not found at: $TRAIN_PATH"
[[ -f "$DESCRIPTOR_PATH" ]] || die "environment.yml not found at: $DESCRIPTOR_PATH"

ensure_conda_env "agentomics-env" "$SCRIPT_DIR/../envs/environment.yaml"
PREP_ROOT="$(mktemp -d)"
trap 'rm -rf "$PREP_ROOT"' EXIT
PREPARED_DIR="$PREP_ROOT/prepared"
PREP_ARGS=(--dataset-dir "$DATASET_DIR" --output-dir "$PREPARED_DIR" --agent-dir "$AGENT_DIR")
[[ -n "$LABEL_COL" ]] && PREP_ARGS+=(--label-col "$LABEL_COL")
PYTHONPATH="$SCRIPT_DIR/../src" conda run -n agentomics-env \
    python -m runtime.prepare_training_data "${PREP_ARGS[@]}"

TRAIN_DATA_PATH="$PREPARED_DIR/train"
VALIDATION_DATA_PATH="$PREPARED_DIR/validation"
[[ -d "$TRAIN_DATA_PATH" ]] || die "Prepared train split not found (does --dataset-dir contain a train split?): $TRAIN_DATA_PATH"
[[ -d "$VALIDATION_DATA_PATH" ]] || die "Prepared validation split not found (does --dataset-dir contain a validation split?): $VALIDATION_DATA_PATH"

print_summary() {
    local artifacts_path="$1"
    echo ""
    echo "========== Training Summary =========="
    if [[ -d "$artifacts_path" ]]; then
        echo "Artifacts created in: $artifacts_path"
        echo ""
        ls -lh "$artifacts_path"
        echo ""
        if [[ -f "$artifacts_path/metadata.json" ]]; then
            echo "Metadata:"
            cat "$artifacts_path/metadata.json"
            echo ""
        fi
    else
        echo "Warning: Artifacts directory not found at $artifacts_path"
    fi
    echo "======================================"
}

if [ "$CPU_ONLY" = true ]; then
    info "Training in CPU-only mode"
    export CUDA_VISIBLE_DEVICES=""
fi

need_cmd conda

ENV_PATH="$(find "$ENV_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n1)"
if [[ -z "$ENV_PATH" ]]; then
    echo "No model conda env found under $ENV_DIR; creating one from $DESCRIPTOR_PATH"
    ENV_PATH="$ENV_DIR/model_env"
    conda env create -f "$DESCRIPTOR_PATH" -p "$ENV_PATH"
fi
echo "Using model env: $ENV_PATH"
echo "Running training..."
cd "$TRAIN_WORKDIR"
conda run -p "$ENV_PATH" \
    python "$TRAIN_PATH" \
    --train-data "$TRAIN_DATA_PATH" \
    --validation-data "$VALIDATION_DATA_PATH" \
    --artifacts-dir "$ARTIFACTS_DIR" ${ARGS[@]+"${ARGS[@]}"}
echo "Training done"
print_summary "$ARTIFACTS_DIR"

if [[ -n "${HOST_UID:-}" ]]; then
    chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$ARTIFACTS_DIR"
fi
