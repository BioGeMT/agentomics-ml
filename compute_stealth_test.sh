#!/usr/bin/env bash
set -e

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-folder)
            EXPERIMENT_FOLDER="$2"
            shift 2
            ;;
        --agentomics-dir)
            AGENTOMICS_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

CONFIG_FILE="$EXPERIMENT_FOLDER/extras/config.json"
AGENT_ID=$(jq -r '.agent_id' "$CONFIG_FILE")
DATASET=$(jq -r '.dataset' "$CONFIG_FILE")
WANDB_RUN_ID=$(jq -r '.wandb_run_id' "$CONFIG_FILE")

echo "Agent ID: $AGENT_ID"
echo "Dataset: $DATASET"
echo "WandB Run ID: $WANDB_RUN_ID"

ITERATIONS=()
for dir in "$EXPERIMENT_FOLDER/run_files"/iteration_[0-9]*; do
    ITERATIONS+=("$(basename "$dir")")
done
echo "Found ${#ITERATIONS[@]} iterations"

conda activate agentomics-env && python src/utils/biomlbench_custom_prepare.py --agentomics-dir "$AGENTOMICS_DIR" --dataset-name "$DATASET"

TEST_OUTPUT_DIR=$(mktemp -d)
trap "rm -rf $TEST_OUTPUT_DIR" EXIT
for iteration in "${ITERATIONS[@]}"; do
    OUTPUT_FILE="$TEST_OUTPUT_DIR/${iteration}_test_predictions.csv"
    echo "Processing $iteration..."
    ./inference.sh \
        --agent-dir "$EXPERIMENT_FOLDER" \
        --code-path "run_files/$iteration" \
        --remove-conda-env \
        --input "prepared_test_sets/$DATASET/test.no_label.csv" \
        --output "$OUTPUT_FILE" || echo "Warning: Failed to run inference for $iteration"
done

EXPERIMENT_FOLDER_ABS="$(cd "$(dirname "$EXPERIMENT_FOLDER")" && pwd)/$(basename "$EXPERIMENT_FOLDER")"
docker run --rm \
    --env-file $(pwd)/.env \
    -e PYTHONPATH=/repository/src \
    -v "$(pwd)/src":/repository/src:ro \
    -v "$(pwd)/prepared_test_sets":/repository/prepared_test_sets:ro \
    -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
    -v "$EXPERIMENT_FOLDER_ABS":/experiment:ro \
    -v "$TEST_OUTPUT_DIR":/test_outputs:ro \
    --entrypoint /opt/conda/envs/agentomics-env/bin/python \
    agentomics_img src/run_logging/evaluate_stealth_test.py \
    --dataset "$DATASET" \
    --test-output-dir /test_outputs \
    --experiment-folder /experiment

echo "Stealth test evaluation complete"