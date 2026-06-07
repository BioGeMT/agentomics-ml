#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bash_helpers.sh"

DOCKER_MODE=true
CPU_ONLY=false
DOCKERHUB_USERNAME="biogemt"
ARGS=()

show_help() {
    echo "Usage: $0 --agent-dir <agent_folder_path> --train-data <train_split_path> --validation-data <validation_split_path> --artifacts-dir <artifacts_dir_path> [--cpu-only] [--local]"
    echo "Options:"
    echo "  --agent-dir       Path to agent folder (required)"
    echo "  --train-data      Path to training split folder with input/ and labels.csv (required)"
    echo "  --validation-data Path to validation split folder with input/ and labels.csv (required)"
    echo "  --artifacts-dir   Path to directory where training artifacts will be saved (required)"
    echo "  --cpu-only        Run without GPU (optional)"
    echo "  --local           Run locally without Docker (optional)"
    echo "  --help            Show this help message and exit"
    exit 0
}

AGENT_DIR=""
TRAIN_DATA_PATH=""
VALIDATION_DATA_PATH=""
ARTIFACTS_DIR=""

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

[[ -d "$AGENT_DIR" ]] || die "--agent-dir does not exist: $AGENT_DIR"

AGENT_NAME=$(basename "$AGENT_DIR")
CODE_ROOT="${AGENT_DIR}/best_iteration_snapshot"
ENV_PATH="${CODE_ROOT}/.conda/envs/${AGENT_NAME}_env"
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

if [[ "$DOCKER_MODE" == true ]]; then
    if [ "$CPU_ONLY" = false ]; then
        if ! docker_has_gpu; then
            warn "GPU not available (nvidia-smi not found or Docker lacks GPU support)"
            warn "Automatically switching to CPU-only mode"
            warn "To suppress this warning, use --cpu-only flag"
            CPU_ONLY=true
        fi
    fi
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
    need_cmd docker
    if ! docker info >/dev/null 2>&1; then
        die "Docker is not running or not accessible (start Docker and retry)"
    fi

    FOUNDATION_MODEL_TYPE="$(load_string_field_from_run_config "$AGENT_DIR" "foundation_models_type")"
    ensure_agentomics_docker_image "$FOUNDATION_MODEL_TYPE" "$DOCKERHUB_USERNAME"

    echo "Running training in Docker..."
    AGENT_DIR_ABS="$(cd "$(dirname "$AGENT_DIR")" && pwd)/$(basename "$AGENT_DIR")"
    TRAIN_DATA_PATH_ABS="$(cd "$(dirname "$TRAIN_DATA_PATH")" && pwd)/$(basename "$TRAIN_DATA_PATH")"
    VALIDATION_DATA_PATH_ABS="$(cd "$(dirname "$VALIDATION_DATA_PATH")" && pwd)/$(basename "$VALIDATION_DATA_PATH")"
    ARTIFACTS_DIR_ABS="$(cd "$(dirname "$ARTIFACTS_DIR")" && pwd)/$(basename "$ARTIFACTS_DIR")"
    CODE_ROOT_IN_CONTAINER="/workspace"
    TRAIN_WORKDIR_IN_CONTAINER="${CODE_ROOT_IN_CONTAINER}/model_training"
    DESCRIPTOR_PATH_IN_CONTAINER="${CODE_ROOT_IN_CONTAINER}/environment.yml"
    if [[ ! -d "$ENV_PATH" ]]; then
        echo "Conda environment not found at: $ENV_PATH"
        echo "Creating conda environment inside Docker..."
        docker run --rm \
            -v "${AGENT_DIR_ABS}/best_iteration_snapshot:${CODE_ROOT_IN_CONTAINER}" \
            --entrypoint "" \
            -w "${CODE_ROOT_IN_CONTAINER}" \
            "${AGENTOMICS_IMAGE}" \
            bash -c "conda install -n base mamba -c conda-forge -y && mamba env create -f \"${DESCRIPTOR_PATH_IN_CONTAINER}\" -p \"${CODE_ROOT_IN_CONTAINER}/.conda/envs/${AGENT_NAME}_env\""
    fi
    docker run --rm \
        -v "${AGENT_DIR_ABS}/best_iteration_snapshot:${CODE_ROOT_IN_CONTAINER}" \
        -v "$(dirname "$TRAIN_DATA_PATH_ABS"):/train_data_dir" \
        -v "$(dirname "$VALIDATION_DATA_PATH_ABS"):/validation_data_dir" \
        -v "$(dirname "$ARTIFACTS_DIR_ABS"):/artifacts_parent_dir" \
        ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
        --entrypoint "" \
        -w "${TRAIN_WORKDIR_IN_CONTAINER}" \
        -e PATH="${CODE_ROOT_IN_CONTAINER}/.conda/envs/${AGENT_NAME}_env/bin:$PATH" \
        "${AGENTOMICS_IMAGE}" \
        python train.py \
        --train-data "/train_data_dir/$(basename "$TRAIN_DATA_PATH_ABS")" \
        --validation-data "/validation_data_dir/$(basename "$VALIDATION_DATA_PATH_ABS")" \
        --artifacts-dir "/artifacts_parent_dir/$(basename "$ARTIFACTS_DIR_ABS")" ${ARGS[@]+"${ARGS[@]}"}
    echo "Training done"
    print_summary "$ARTIFACTS_DIR"
else
    need_cmd conda
    if [[ ! -d "$ENV_PATH" ]]; then
        echo "Conda environment not found at: $ENV_PATH"
        conda env create -f "$DESCRIPTOR_PATH" -p "$ENV_PATH"
    fi
    echo "Running training locally..."
    cd "$TRAIN_WORKDIR"
    conda run -p "$ENV_PATH" \
        python "$TRAIN_PATH" \
        --train-data "$TRAIN_DATA_PATH" \
        --validation-data "$VALIDATION_DATA_PATH" \
        --artifacts-dir "$ARTIFACTS_DIR" ${ARGS[@]+"${ARGS[@]}"}
    echo "Training done"
    print_summary "$ARTIFACTS_DIR"
fi
