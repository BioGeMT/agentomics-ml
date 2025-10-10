#!/usr/bin/env bash

DOCKER_MODE=true
CPU_ONLY=false
ARGS=()
show_help() {
    echo "Usage: $0 --agent-dir <agent_folder_path> --input <input_path> --output <output_path> [--cpu-only] [--local]"
    echo "Options:"
    echo "  --agent-dir   Path to agent folder (required)"
    echo "  --input       Path to input file (required)"
    echo "  --output      Path to output file (required)"
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

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agent-dir) AGENT_DIR="$2"; shift ;;
        --input) INPUT_PATH="$2"; shift ;;
        --output) OUTPUT_PATH="$2"; shift ;;
        --cpu-only) CPU_ONLY=true; shift ;;
        --local) DOCKER_MODE=false ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

# ensure all required args are provided
if [[ -z "$AGENT_DIR" || -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    show_help
fi

AGENT_NAME=$(basename "$AGENT_DIR")
ENV_PATH="${AGENT_DIR}/best_run_files/.conda/envs/${AGENT_NAME}_env"
INFERENCE_PATH="${AGENT_DIR}/best_run_files/inference.py"

if [[ ! -d "$ENV_PATH" ]]; then
    echo "Conda environment not found at: $ENV_PATH"
    exit 1
fi

if [[ ! -f "$INFERENCE_PATH" ]]; then
    echo "inference.py not found at: $INFERENCE_PATH"
    exit 1
fi

GPU_FLAGS=()
if [ "$CPU_ONLY" = false ]; then
    GPU_FLAGS+=(--gpus all)
    GPU_FLAGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
fi

if [[ "$DOCKER_MODE" == true ]]; then
    AGENT_DIR_ABS="$(cd "$(dirname "$AGENT_DIR")" && pwd)/$(basename "$AGENT_DIR")"
    INPUT_PATH_ABS="$(cd "$(dirname "$INPUT_PATH")" && pwd)/$(basename "$INPUT_PATH")"
    OUTPUT_PATH_ABS="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"
    docker run --rm \
        -v "${AGENT_DIR_ABS}/best_run_files:/workspace" \
        -v "$(dirname "$INPUT_PATH_ABS"):/input_dir" \
        -v "$(dirname "$OUTPUT_PATH_ABS"):/output_dir" \
        ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
        -w /workspace \
        -e PATH="/workspace/.conda/envs/${AGENT_NAME}_env/bin:$PATH" \
        condaforge/mambaforge:23.3.1-0 \
        python inference.py \
        --input "/input_dir/$(basename "$INPUT_PATH_ABS")" \
        --output "/output_dir/$(basename "$OUTPUT_PATH_ABS")" "${ARGS[@]}"
else
    conda run -p "$ENV_PATH" \
        python "$INFERENCE_PATH" \
        --input "$INPUT_PATH" \
        --output "$OUTPUT_PATH" "${ARGS[@]}"
fi