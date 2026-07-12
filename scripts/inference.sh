#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bash_helpers.sh"

CPU_ONLY=false
REMOVE_CONDA_ENV=false
ALL_ITERATIONS=false
ARGS=()

show_help() {
    echo "Usage: $0 --agent-dir <agent_folder_path> --input <input_path> --output <output_path> [options]"
    echo "Options:"
    echo "  --agent-dir   Path to agent folder (required)"
    echo "  --input       Path to a split folder (input/ + optional labels.csv). If a labels.csv is present, also produces metrics.json. Alternatively, a csv file can be passed for csv datasets. (required)"
    echo "  --output      Path to output file (required)"
    echo "  --label-col   Label column for a CSV dataset. When set, labels are extracted and metrics are computed, written to <output>.metrics.json next to --output. Ignored when --input is a split folder (optional)"
    echo "  --code-path   Path to code files, points to best_iteration_snapshot by default, must be relative to --agent-dir and a child of --agent-dir (optional)"
    echo "  --all-iterations   Run inference for every run/iteration_N in --agent-dir against --input."
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
        --all-iterations)
            ALL_ITERATIONS=true
            shift
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
[[ -f "$INPUT_PATH" || -d "$INPUT_PATH" ]] || die "--input not found: $INPUT_PATH"
[[ -d "$(dirname "$OUTPUT_PATH")" ]] || die "--output directory does not exist: $(dirname "$OUTPUT_PATH")"

# Resolve to absolute paths so they survive the cd into the inference workdir.
AGENT_DIR="$(cd "$AGENT_DIR" && pwd)"
INPUT_PATH="$(cd "$(dirname "$INPUT_PATH")" && pwd)/$(basename "$INPUT_PATH")"
OUTPUT_PATH="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"

if [[ "$ALL_ITERATIONS" == true ]]; then
    RUN_DIR="${AGENT_DIR}/run"
    [[ -d "$RUN_DIR" ]] || die "--all-iterations requires ${RUN_DIR}, not found"
    OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"

    mapfile -t ITERATION_DIRS < <(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'iteration_*' | sort -V)
    [[ ${#ITERATION_DIRS[@]} -gt 0 ]] || die "No iteration_* directories found under ${RUN_DIR}"

    echo "Evaluating ${#ITERATION_DIRS[@]} archived iteration(s) on $INPUT_PATH"
    for iter_dir in "${ITERATION_DIRS[@]}"; do
        iter_name="$(basename "$iter_dir")"
        echo "=== ${iter_name} ==="
        iter_args=(
            --agent-dir "$AGENT_DIR"
            --input "$INPUT_PATH"
            --output "${OUTPUT_DIR}/${iter_name}_predictions.csv"
            --code-path "run/${iter_name}"
        )
        [[ -n "$LABEL_COL" ]] && iter_args+=(--label-col "$LABEL_COL")
        [[ "$CPU_ONLY" == true ]] && iter_args+=(--cpu-only)
        iter_args+=(--remove-conda-env)

        #This recursively calls the inference.sh with iteration-specific arguments
        "$0" "${iter_args[@]}" ${ARGS[@]+"${ARGS[@]}"} \
            || warn "Inference failed for ${iter_name}; continuing"
    done
    echo "All iterations done. Predictions written to ${OUTPUT_DIR}/<iteration>_predictions.csv"
    exit 0
fi

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

ENV_PATH=""
if [[ -d "$ENV_DIR" ]]; then
    ENV_PATH="$(find "$ENV_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n1)"
fi
if [[ -z "$ENV_PATH" ]]; then
    echo "No model conda env found under $ENV_DIR; creating one from $DESCRIPTOR_PATH"
    ENV_PATH="$ENV_DIR/model_env"
    conda env create -f "$DESCRIPTOR_PATH" -p "$ENV_PATH"
fi
echo "Using model env: $ENV_PATH"

if [[ -d "$INPUT_PATH" ]]; then
    [[ -d "$INPUT_PATH/input" ]] || die "--input folder must contain input/: $INPUT_PATH"
    INPUT_SPLIT_DIR="$INPUT_PATH"
else
    ensure_conda_env "agentomics-env" "$SCRIPT_DIR/../envs/environment.yaml"
    INPUT_SPLIT_DIR="$(mktemp -d)"
    trap 'rm -rf "$INPUT_SPLIT_DIR"' EXIT
    CONVERT_ARGS=(--input "$INPUT_PATH" --output-split "$INPUT_SPLIT_DIR")
    [[ -n "$LABEL_COL" ]] && CONVERT_ARGS+=(--label-col "$LABEL_COL")
    PYTHONPATH="$SCRIPT_DIR/../src" conda run -n agentomics-env \
        python -m datasets.prepare_inference_input "${CONVERT_ARGS[@]}"
fi

echo "Running inference"
cd "$INFERENCE_WORKDIR"
conda run -p "$ENV_PATH" \
    python "$INFERENCE_PATH" \
    --input "$INPUT_SPLIT_DIR/input" \
    --output "$OUTPUT_PATH" \
    --artifacts-dir "$ARTIFACTS_PATH" ${ARGS[@]+"${ARGS[@]}"}
echo "Inference done"

METRICS_PATH="$(dirname "$OUTPUT_PATH")/$(basename "$OUTPUT_PATH" .csv).metrics.json"
if [[ -f "$INPUT_SPLIT_DIR/labels.csv" ]]; then
    ensure_conda_env "agentomics-env" "$SCRIPT_DIR/../envs/environment.yaml"
    echo "Computing metrics..."
    PYTHONPATH="$SCRIPT_DIR/../src" conda run -n agentomics-env python "$SCRIPT_DIR/../src/runtime/evaluate.py" \
        --agent-dir "$AGENT_DIR" \
        --predictions "$OUTPUT_PATH" \
        --labels "$INPUT_SPLIT_DIR/labels.csv" \
        --output "$METRICS_PATH"
fi

if [[ -n "${HOST_UID:-}" ]]; then
    chown "$HOST_UID:${HOST_GID:-$HOST_UID}" "$OUTPUT_PATH"
    [[ -f "$METRICS_PATH" ]] && chown "$HOST_UID:${HOST_GID:-$HOST_UID}" "$METRICS_PATH"
fi

if [[ "$REMOVE_CONDA_ENV" == true ]]; then
        rm -rf "$ENV_PATH"
fi
