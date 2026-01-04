#!/usr/bin/env bash

source "./bash_helpers.sh"

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
}

usage_error() {
    show_help
    exit 1
}

AGENT_DIR=""
INPUT_PATH=""
OUTPUT_PATH=""

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
        --input)
            require_opt_value "$1" "${2:-}"
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            require_opt_value "$1" "${2:-}"
            OUTPUT_PATH="$2"
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
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# ensure all required args are provided
if [[ -z "$AGENT_DIR" || -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    usage_error
fi

[[ -d "$AGENT_DIR" ]] || die "--agent-dir does not exist: $AGENT_DIR"
[[ -f "$INPUT_PATH" ]] || die "--input does not exist: $INPUT_PATH"
[[ -d "$(dirname "$OUTPUT_PATH")" ]] || die "--output directory does not exist: $(dirname "$OUTPUT_PATH")"

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

if [[ "$DOCKER_MODE" == true ]]; then
    # Auto-detect GPU availability if not explicitly set to CPU-only
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
    info "GPU mode enabled"
else
    info "Running in CPU-only mode"
fi

if [[ "$DOCKER_MODE" == true ]]; then
    need_cmd docker
    if ! docker info >/dev/null 2>&1; then
        die "Docker is not running or not accessible (start Docker and retry)"
    fi
    if ! docker image inspect agentomics_img >/dev/null 2>&1; then
        die "Docker image 'agentomics_img' not found. Run ./run.sh once to build it (or build it manually) and retry."
    fi
    echo "Running inference in Docker..."
    AGENT_DIR_ABS="$(cd "$(dirname "$AGENT_DIR")" && pwd)/$(basename "$AGENT_DIR")"
    INPUT_PATH_ABS="$(cd "$(dirname "$INPUT_PATH")" && pwd)/$(basename "$INPUT_PATH")"
    OUTPUT_PATH_ABS="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"
    docker run --rm \
        -v "${AGENT_DIR_ABS}/best_run_files:/workspace" \
        -v "$(dirname "$INPUT_PATH_ABS"):/input_dir" \
        -v "$(dirname "$OUTPUT_PATH_ABS"):/output_dir" \
        ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
        --entrypoint "" \
        -w /workspace \
        -e PATH="/workspace/.conda/envs/${AGENT_NAME}_env/bin:$PATH" \
        agentomics_img \
        python inference.py \
        --input "/input_dir/$(basename "$INPUT_PATH_ABS")" \
        --output "/output_dir/$(basename "$OUTPUT_PATH_ABS")" "${ARGS[@]}" 
    echo "Inference done"
else
    need_cmd conda
    echo "Running inference locally..."
    cd "$(dirname "$INFERENCE_PATH")"
    conda run -p "$ENV_PATH" \
        python "$INFERENCE_PATH" \
        --input "$INPUT_PATH" \
        --output "$OUTPUT_PATH" "${ARGS[@]}"
    echo "Inference done"
    
fi