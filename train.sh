#!/usr/bin/env bash

DOCKER_MODE=true
CPU_ONLY=false
ARGS=()
show_help() {
    echo "Usage: $0 --agent-dir <agent_folder_path> --train-data <train_data_path> --validation-data <validation_data_path> --artifacts-dir <artifacts_dir_path> [--cpu-only] [--local]"
    echo "Options:"
    echo "  --agent-dir       Path to agent folder (required)"
    echo "  --train-data      Path to training data CSV file (required)"
    echo "  --validation-data Path to validation data CSV file (required)"
    echo "  --artifacts-dir   Path to directory where training artifacts will be saved (required)"
    echo "  --cpu-only        Run without GPU (optional)"
    echo "  --local           Run locally without Docker (optional)"
    echo "  --help            Show this help message and exit"
    exit 0
}

for arg in "$@"; do
    if [[ "$arg" == "--help" ]]; then
        show_help
    fi
done

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent-dir)
            AGENT_DIR="$2"
            shift 2
            ;;
        --train-data)
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --validation-data)
            VALIDATION_DATA_PATH="$2"
            shift 2
            ;;
        --artifacts-dir)
            ARTIFACTS_DIR="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --local)
            DOCKER_MODE=false
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
if [[ -z "$AGENT_DIR" || -z "$TRAIN_DATA_PATH" || -z "$VALIDATION_DATA_PATH" || -z "$ARTIFACTS_DIR" ]]; then
    show_help
fi

AGENT_NAME=$(basename "$AGENT_DIR")
ENV_PATH="${AGENT_DIR}/best_run_files/.conda/envs/${AGENT_NAME}_env"
TRAIN_PATH="${AGENT_DIR}/best_run_files/train.py"

if [[ ! -d "$ENV_PATH" ]]; then
    echo "Conda environment not found at: $ENV_PATH"
    exit 1
fi

if [[ ! -f "$TRAIN_PATH" ]]; then
    echo "train.py not found at: $TRAIN_PATH"
    exit 1
fi

GPU_FLAGS=()
if [ "$CPU_ONLY" = false ]; then
    GPU_FLAGS+=(--gpus all)
    GPU_FLAGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
fi

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

if [[ "$DOCKER_MODE" == true ]]; then
    echo "Running training in Docker..."
    AGENT_DIR_ABS="$(cd "$(dirname "$AGENT_DIR")" && pwd)/$(basename "$AGENT_DIR")"
    TRAIN_DATA_PATH_ABS="$(cd "$(dirname "$TRAIN_DATA_PATH")" && pwd)/$(basename "$TRAIN_DATA_PATH")"
    VALIDATION_DATA_PATH_ABS="$(cd "$(dirname "$VALIDATION_DATA_PATH")" && pwd)/$(basename "$VALIDATION_DATA_PATH")"
    ARTIFACTS_DIR_ABS="$(cd "$(dirname "$ARTIFACTS_DIR")" && pwd)/$(basename "$ARTIFACTS_DIR")"
    docker run --rm \
        -v "${AGENT_DIR_ABS}/best_run_files:/workspace" \
        -v "$(dirname "$TRAIN_DATA_PATH_ABS"):/train_data_dir" \
        -v "$(dirname "$VALIDATION_DATA_PATH_ABS"):/validation_data_dir" \
        -v "$(dirname "$ARTIFACTS_DIR_ABS"):/artifacts_parent_dir" \
        ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
        --entrypoint "" \
        -w /workspace \
        -e PATH="/workspace/.conda/envs/${AGENT_NAME}_env/bin:$PATH" \
        agentomics_img \
        python train.py \
        --train-data "/train_data_dir/$(basename "$TRAIN_DATA_PATH_ABS")" \
        --validation-data "/validation_data_dir/$(basename "$VALIDATION_DATA_PATH_ABS")" \
        --artifacts-dir "/artifacts_parent_dir/$(basename "$ARTIFACTS_DIR_ABS")" "${ARGS[@]}"
    echo "Training done"
    print_summary "$ARTIFACTS_DIR"
else
    echo "Running training locally..."
    cd "$(dirname "$TRAIN_PATH")"
    conda run -p "$ENV_PATH" \
        python "$TRAIN_PATH" \
        --train-data "$TRAIN_DATA_PATH" \
        --validation-data "$VALIDATION_DATA_PATH" \
        --artifacts-dir "$ARTIFACTS_DIR" "${ARGS[@]}"
    echo "Training done"
    print_summary "$ARTIFACTS_DIR"
fi

