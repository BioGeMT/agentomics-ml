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

POLARIS_DATASETS_WITH_POLARIS_PREFIX=(
    "pkis2-egfr-wt-c-1"
    "adme-fang-hclint-1"
    "adme-fang-hppb-1"
    "adme-fang-solu-1"
)
POLARIS_DATASETS_WITH_TDCOMMONS_PREFIX=(
    "cyp2d6-substrate-carbonmangels"
    "lipophilicity-astrazeneca"
    "herg"
    "bbb-martins"
    "caco2-wang"
)
PROTEINGYM_DATASETS=(
    "SPIKE_SARS2_Starr_2020_binding"
    "SPA_STAAU_Tsuboyama_2023_1LP1"
    "PSAE_PICP2_Tsuboyama_2023_1PSE_indels"
    "CBX4_HUMAN_Tsuboyama_2023_2K28"
    "Q8EG35_SHEON_Campbell_2022_indels"
    "CSN4_MOUSE_Tsuboyama_2023_1UFM_indels"
)

if [[ "$DATASET" != *"/"* ]]; then
    for dset in "${POLARIS_DATASETS_WITH_POLARIS_PREFIX[@]}"; do
        if [[ "$DATASET" == "$dset" ]]; then
            DATASET="polarishub/polaris-$DATASET"
            break
        fi
    done
    for dset in "${POLARIS_DATASETS_WITH_TDCOMMONS_PREFIX[@]}"; do
        if [[ "$DATASET" == "$dset" ]]; then
            DATASET="polarishub/tdcommons-$DATASET"
            break
        fi
    done

    for dset in "${PROTEINGYM_DATASETS[@]}"; do
        if [[ "$DATASET" == "$dset" ]]; then
            DATASET="proteingym-dms/$DATASET"
            break
        fi
    done
fi

echo "Agent ID: $AGENT_ID"
echo "Dataset: $DATASET"
echo "WandB Run ID: $WANDB_RUN_ID"

declare -A ITERATION_PATHS
while IFS= read -r dir; do
    iteration_name="$(basename "$dir")"
    parent_path="${dir%/*}"
    relative_parent="${parent_path#$EXPERIMENT_FOLDER/}"
    relative_path="$relative_parent/$iteration_name"
    ITERATION_PATHS["$iteration_name"]="$relative_path"
done < <(find "$EXPERIMENT_FOLDER" -maxdepth 3 -type d -name "iteration_*" | sort -V)
echo "Found ${#ITERATION_PATHS[@]} iterations"

source activate agentomics-env && PYTHONPATH="${AGENTOMICS_DIR}/src" python src/utils/biomlbench_custom_prepare.py --agentomics-dir "$AGENTOMICS_DIR" --dataset-name "$DATASET"

TEST_OUTPUT_DIR=$(mktemp -d)
trap "rm -rf $TEST_OUTPUT_DIR" EXIT
for iteration in "${!ITERATION_PATHS[@]}"; do
    CODE_PATH="${ITERATION_PATHS[$iteration]}"
    OUTPUT_FILE="$TEST_OUTPUT_DIR/${iteration}_test_predictions.csv"
    echo "Processing $iteration (code path: $CODE_PATH)..."
    ./inference.sh \
        --agent-dir "$EXPERIMENT_FOLDER" \
        --code-path "$CODE_PATH" \
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