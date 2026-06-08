#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bash_helpers.sh"

CPU_ONLY=false
REMOVE_CONDA_ENV=false
ARGS=()

show_help() {
    echo "Usage: $0 --agent-dir <agent_folder_path> --input <input_path> --output <output_path> [--cpu-only]"
    echo "Options:"
    echo "  --agent-dir   Path to agent folder (required)"
    echo "  --input       Path to input folder (required)"
    echo "  --output      Path to output file (required)"
    echo "  --label-col   Name of the true label column in --input. When set, metrics are computed against it and written to metrics.json next to --output (optional)"
    echo "  --code-path   Path to code files, points to best_iteration_snapshot by default, must be relative to --agent-dir and a child of --agent-dir (optional)"
    echo "  --remove-conda-env   Remove the conda environment after inference (optional)"
    echo "  --cpu-only    Run without GPU (optional)"
    echo "  --help        Show this help message and exit"
}

AGENT_DIR=""
INPUT_PATH=""
OUTPUT_PATH=""
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
        --remove-conda-env)
            REMOVE_CONDA_ENV=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
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
if [[ -z "$AGENT_DIR" ]]; then
      die "Missing required argument: --agent-dir. Run '$0 --help' for usage"
fi
if [[ -z "$INPUT_PATH" ]]; then
    die "Missing required argument: --input. Run '$0 --help' for usage"
fi
if [[ -z "$OUTPUT_PATH" ]]; then
    die "Missing required argument: --output. Run '$0 --help' for usage"
fi

[[ -d "$AGENT_DIR" ]] || die "--agent-dir does not exist: $AGENT_DIR"
[[ -d "$INPUT_PATH" ]] || die "--input must be an input folder: $INPUT_PATH"
[[ -d "$(dirname "$OUTPUT_PATH")" ]] || die "--output directory does not exist: $(dirname "$OUTPUT_PATH")"

# Resolve to absolute paths so they survive the cd into the inference workdir.
AGENT_DIR="$(cd "$AGENT_DIR" && pwd)"
INPUT_PATH="$(cd "$(dirname "$INPUT_PATH")" && pwd)/$(basename "$INPUT_PATH")"
OUTPUT_PATH="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"

CODE_PATH=${CODE_PATH:-"best_iteration_snapshot"}
CODE_ROOT="${AGENT_DIR}/${CODE_PATH}"
echo "Using code path: $CODE_PATH"
ENV_DIR="${CODE_ROOT}/.conda/envs"
INFERENCE_PATH="${CODE_ROOT}/model_inference/inference.py"
INFERENCE_WORKDIR="$(dirname "$INFERENCE_PATH")"
ARTIFACTS_PATH="${CODE_ROOT}/model_training/training_artifacts"
DESCRIPTOR_PATH="${CODE_ROOT}/environment.yml"

if [[ ! -f "$DESCRIPTOR_PATH" ]]; then
    DESCRIPTOR_PATH="${CODE_ROOT}/runtime_info/environment.yml"
fi

if [[ ! -f "$INFERENCE_PATH" ]]; then
    echo "inference.py not found at: $INFERENCE_PATH"
    exit 1
fi

if [[ ! -f "$DESCRIPTOR_PATH" ]]; then
    echo "environment.yml not found at: $DESCRIPTOR_PATH"
    exit 1
fi

if [ "$CPU_ONLY" = true ]; then
    info "Running in CPU-only mode"
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
NORMALIZE_SCRIPT_ABS="$SCRIPT_DIR/../src/datasets/normalize_dataset.py"
NORMALIZED_FILENAME=$(python "$NORMALIZE_SCRIPT_ABS" --input "$INPUT_PATH")
if [[ -n "$NORMALIZED_FILENAME" ]]; then
    trap "rm -f \"$(dirname "$INPUT_PATH")/$NORMALIZED_FILENAME\"" EXIT
    INPUT_PATH="$(dirname "$INPUT_PATH")/$NORMALIZED_FILENAME"
fi

echo "Running inference"
cd "$INFERENCE_WORKDIR"
conda run -p "$ENV_PATH" \
    python "$INFERENCE_PATH" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --artifacts-dir "$ARTIFACTS_PATH" ${ARGS[@]+"${ARGS[@]}"}
echo "Inference done"

METRICS_PATH="$(dirname "$OUTPUT_PATH")/metrics.json"
if [[ -n "$LABEL_COL" ]]; then
    ensure_conda_env "agentomics-env" "$SCRIPT_DIR/../envs/environment.yaml"
    echo "Computing metrics..."
    PYTHONPATH="$SCRIPT_DIR/../src" conda run -n agentomics-env python "$SCRIPT_DIR/../src/runtime/evaluate.py" \
        --agent-dir "$AGENT_DIR" \
        --predictions "$OUTPUT_PATH" \
        --labeled-input "$INPUT_PATH" \
        --label-col "$LABEL_COL" \
        --output "$METRICS_PATH"
fi

if [[ -n "${HOST_UID:-}" ]]; then
    chown "$HOST_UID:${HOST_GID:-$HOST_UID}" "$OUTPUT_PATH"
    [[ -f "$METRICS_PATH" ]] && chown "$HOST_UID:${HOST_GID:-$HOST_UID}" "$METRICS_PATH"
fi

if [[ "$REMOVE_CONDA_ENV" == true ]]; then
        rm -rf "$ENV_PATH"
fi
