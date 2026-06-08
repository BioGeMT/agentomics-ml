#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bash_helpers.sh"

CPU_ONLY=false
ARGS=()

show_help() {
    echo "Usage: $0 --agent-dir <agent_folder_path> --train-data <train_split_path> --validation-data <validation_split_path> --artifacts-dir <artifacts_dir_path> [--cpu-only] [--local]"
    echo "Options:"
    echo "  --agent-dir       Path to agent folder (required)"
    echo "  --train-data      Path to training split folder with input/ and labels.csv (required)"
    echo "  --validation-data Path to validation split folder with input/ and labels.csv (required)"
    echo "  --artifacts-dir   Path to directory where training artifacts will be saved (required)"
    echo "  --label-col       Name of the label column in the raw --train-data/--validation-data. The data is converted to the prepared format (id + numeric_label, using the run's label mapping) before training (required)"
    echo "  --code-path   Path to code files, points to best_iteration_snapshot by default, must be relative to --agent-dir and a child of --agent-dir (optional)"
    echo "  --cpu-only        Run without GPU (optional)"
    echo "  --help            Show this help message and exit"
    exit 0
}

AGENT_DIR=""
TRAIN_DATA_PATH=""
VALIDATION_DATA_PATH=""
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
        --train-data)
            require_opt_value "$1" "${2:-}"
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --validation-data)
            require_opt_value "$1" "${2:-}"
            VALIDATION_DATA_PATH="$2"
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
if [[ -z "$TRAIN_DATA_PATH" ]]; then
    die "Missing required argument: --train-data. Run '$0 --help' for usage"
fi
if [[ -z "$VALIDATION_DATA_PATH" ]]; then
    die "Missing required argument: --validation-data. Run '$0 --help' for usage"
fi
if [[ -z "$ARTIFACTS_DIR" ]]; then
    die "Missing required argument: --artifacts-dir. Run '$0 --help' for usage"
fi
if [[ -z "$LABEL_COL" ]]; then
    die "Missing required argument: --label-col. Run '$0 --help' for usage"
fi

[[ -d "$AGENT_DIR" ]] || die "--agent-dir does not exist: $AGENT_DIR"

AGENT_DIR="$(cd "$AGENT_DIR" && pwd)"

CODE_PATH=${CODE_PATH:-"best_iteration_snapshot"}
CODE_ROOT="${AGENT_DIR}/${CODE_PATH}"
echo "Using code path: $CODE_PATH"
ENV_DIR="${CODE_ROOT}/.conda/envs"
TRAIN_PATH="${CODE_ROOT}/model_training/train.py"
TRAIN_WORKDIR="$(dirname "$TRAIN_PATH")"
DESCRIPTOR_PATH="${CODE_ROOT}/environment.yml"

[[ -d "$TRAIN_DATA_PATH" ]] || die "--train-data must be a split folder: $TRAIN_DATA_PATH"
[[ -d "$VALIDATION_DATA_PATH" ]] || die "--validation-data must be a split folder: $VALIDATION_DATA_PATH"
[[ -d "$TRAIN_DATA_PATH/input" ]] || die "--train-data must contain an input/ folder: $TRAIN_DATA_PATH"
[[ -f "$TRAIN_DATA_PATH/labels.csv" ]] || die "--train-data must contain labels.csv: $TRAIN_DATA_PATH"
[[ -d "$VALIDATION_DATA_PATH/input" ]] || die "--validation-data must contain an input/ folder: $VALIDATION_DATA_PATH"
[[ -f "$VALIDATION_DATA_PATH/labels.csv" ]] || die "--validation-data must contain labels.csv: $VALIDATION_DATA_PATH"
[[ -f "$TRAIN_PATH" ]] || die "train.py not found at: $TRAIN_PATH"
[[ -f "$DESCRIPTOR_PATH" ]] || die "environment.yml not found at: $DESCRIPTOR_PATH"
[[ -d "$(dirname "$ARTIFACTS_DIR")" ]] || die "--artifacts-dir parent directory does not exist: $(dirname "$ARTIFACTS_DIR")"

TRAIN_DATA_PATH="$(cd "$(dirname "$TRAIN_DATA_PATH")" && pwd)/$(basename "$TRAIN_DATA_PATH")"
VALIDATION_DATA_PATH="$(cd "$(dirname "$VALIDATION_DATA_PATH")" && pwd)/$(basename "$VALIDATION_DATA_PATH")"
ARTIFACTS_DIR="$(cd "$(dirname "$ARTIFACTS_DIR")" && pwd)/$(basename "$ARTIFACTS_DIR")"

CONFIG_PATH="$AGENT_DIR/run/shared/config.json"
[[ -f "$CONFIG_PATH" ]] || die "Run config not found at $CONFIG_PATH (needed to map --label-col)"
NORMALIZE_SCRIPT="$SCRIPT_DIR/../src/datasets/normalize_dataset.py"

train_norm=$(python "$NORMALIZE_SCRIPT" --input "$TRAIN_DATA_PATH" --label-col "$LABEL_COL" --config-path "$CONFIG_PATH")
val_norm=$(python "$NORMALIZE_SCRIPT" --input "$VALIDATION_DATA_PATH" --label-col "$LABEL_COL" --config-path "$CONFIG_PATH")
TRAIN_DATA_PATH="$(dirname "$TRAIN_DATA_PATH")/$train_norm"
VALIDATION_DATA_PATH="$(dirname "$VALIDATION_DATA_PATH")/$val_norm"
trap "rm -f \"$TRAIN_DATA_PATH\" \"$VALIDATION_DATA_PATH\"" EXIT

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
