#!/usr/bin/env bash
set -e

DOCKER_MODE=true
CPU_ONLY=false
REMOVE_CONDA_ENV=false
ARGS=()
show_help() {
    echo "Usage: $0 --agent-dir <agent_folder_path> --input <input_path> --output <output_path> [--cpu-only] [--local]"
    echo "Options:"
    echo "  --agent-dir   Path to agent folder (required)"
    echo "  --input       Path to input file (required)"
    echo "  --output      Path to output file (required)"
    echo "  --code-path   Path to code files, points to best iteration files by default, must be relative to --agent-dir and a child of --agent-dir (optional)"
    echo "  --remove-conda-env   Remove the conda environment after inference (optional)"
    echo "  --cpu-only    Run without GPU (optional)"
    echo "  --local       Run locally without Docker (optional)"
    echo "  --help        Show this help message and exit"
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
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --code-path)
            CODE_PATH="$2"
            shift 2
            ;;
        --remove-conda-env)
            REMOVE_CONDA_ENV=true
            shift
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
if [[ -z "$AGENT_DIR" || -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    show_help
fi

AGENT_NAME=$(basename "$AGENT_DIR")
CODE_PATH=${CODE_PATH:-"best_run_files"}
echo "Using code path: $CODE_PATH"
ENV_PATH="${AGENT_DIR}/${CODE_PATH}/.conda/envs/${AGENT_NAME}_env"
INFERENCE_PATH="${AGENT_DIR}/${CODE_PATH}/inference.py"

if [[ ! -f "$INFERENCE_PATH" ]]; then
    echo "inference.py not found at: $INFERENCE_PATH"
    exit 1
fi

if [[ ! -d "$ENV_PATH" ]]; then
    echo "Conda environment not found at: $ENV_PATH"
    if [[ ! -f "$AGENT_DIR/${CODE_PATH}/conda_environment.yml" ]]; then
        echo "conda_environment.yml not found at: $AGENT_DIR/${CODE_PATH}/conda_environment.yml"
        exit 1
    fi
    conda env create -f "$AGENT_DIR/${CODE_PATH}/conda_environment.yml" -p "$ENV_PATH"
fi

GPU_FLAGS=()
if [ "$CPU_ONLY" = false ]; then
    GPU_FLAGS+=(--gpus all)
    GPU_FLAGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
fi

if [[ "$DOCKER_MODE" == true ]]; then
    echo "Running inference in Docker..."
    AGENT_DIR_ABS="$(cd "$(dirname "$AGENT_DIR")" && pwd)/$(basename "$AGENT_DIR")"
    INPUT_PATH_ABS="$(cd "$(dirname "$INPUT_PATH")" && pwd)/$(basename "$INPUT_PATH")"
    OUTPUT_PATH_ABS="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"
    docker run --rm \
        -v "${AGENT_DIR_ABS}/${CODE_PATH}:/workspace" \
        -v "$(dirname "$INPUT_PATH_ABS"):/input_dir" \
        -v "$(dirname "$OUTPUT_PATH_ABS"):/output_dir" \
        ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
        --entrypoint "" \
        -w /workspace \
        -e PATH="/workspace/.conda/envs/${AGENT_NAME}_env/bin:$PATH" \
        agentomics_img \
        python inference.py \
        --input "/input_dir/$(basename "$INPUT_PATH_ABS")" \
        --output "/output_dir/$(basename "$OUTPUT_PATH_ABS")" \
        --artifacts-dir "/workspace/training_artifacts"
    echo "Inference done"
else
    echo "Running inference locally..."
    cd "$(dirname "$INFERENCE_PATH")"
    conda run -p "$ENV_PATH" \
        python "$INFERENCE_PATH" \
        --input "$INPUT_PATH" \
        --output "$OUTPUT_PATH" "${ARGS[@]}"
    echo "Inference done"
fi

if [[ "$REMOVE_CONDA_ENV" == true ]]; then
    echo "Removing conda environment at: $ENV_PATH"
    rm -rf "$ENV_PATH"
fi