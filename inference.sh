#!/usr/bin/env bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agent-dir) AGENT_DIR="$2"; shift ;;
        --input) INPUT_PATH="$2"; shift ;;
        --output) OUTPUT_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ensure all required args are provided
if [[ -z "$AGENT_DIR" || -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    echo "Usage: $0 --agent-dir <agent_folder_path> --input <input_path> --output <output_path>"
    exit 1
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

conda run -p "$ENV_PATH" \
    python "$INFERENCE_PATH" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH"