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

IS_BIOMLBENCH=false
if [[ "$DATASET" != *"/"* ]]; then
    for dset in "${POLARIS_DATASETS_WITH_POLARIS_PREFIX[@]}"; do
        if [[ "$DATASET" == "$dset" ]]; then
            DATASET="polarishub/polaris-$DATASET"
            IS_BIOMLBENCH=true
            break
        fi
    done
    for dset in "${POLARIS_DATASETS_WITH_TDCOMMONS_PREFIX[@]}"; do
        if [[ "$DATASET" == "$dset" ]]; then
            DATASET="polarishub/tdcommons-$DATASET"
            IS_BIOMLBENCH=true
            break
        fi
    done

    for dset in "${PROTEINGYM_DATASETS[@]}"; do
        if [[ "$DATASET" == *"$dset"* ]]; then
            DATASET="proteingym-dms/$dset"
            IS_BIOMLBENCH=true
            break
        fi
    done
fi

echo "Agent ID: $AGENT_ID"
echo "Dataset: $DATASET"
echo "WandB Run ID: $WANDB_RUN_ID"

declare -A ITERATION_PATHS
EXPERIMENT_FOLDER_NORMALIZED="${EXPERIMENT_FOLDER%/}"
while IFS= read -r dir; do
    iteration_name="$(basename "$dir")"
    parent_path="${dir%/*}"
    relative_parent="${parent_path#$EXPERIMENT_FOLDER_NORMALIZED/}"
    relative_path="$relative_parent/$iteration_name"
    ITERATION_PATHS["$iteration_name"]="$relative_path"
done < <(find "$EXPERIMENT_FOLDER" -maxdepth 3 -type d -name "iteration_*" | sort -V)
echo "Found ${#ITERATION_PATHS[@]} iterations"

if [[ "$IS_BIOMLBENCH" == true ]]; then
    docker run --rm \
        --env-file $(pwd)/.env \
        -e PYTHONPATH=/repository/src \
        -v "$(pwd)":/repository \
        -v "$HOME/.cache/bioml-bench":/root/.cache/bioml-bench \
        -w /repository \
        --entrypoint /opt/conda/envs/agentomics-env/bin/python \
        agentomics_img \
        src/utils/biomlbench_custom_prepare.py \
        --agentomics-dir /repository \
        --dataset-name "$DATASET"
fi

# Check if dataset is a proteingym dataset
IS_PROTEINGYM=false
for dset in "${PROTEINGYM_DATASETS[@]}"; do
    if [[ "$DATASET" == *"$dset"* ]]; then
        IS_PROTEINGYM=true
        break
    fi
done

TEST_OUTPUT_DIR="$EXPERIMENT_FOLDER/temp_stealth_test_predictions"
mkdir -p "$TEST_OUTPUT_DIR"
trap "rm -rf $TEST_OUTPUT_DIR" EXIT

if [[ "$IS_PROTEINGYM" == true ]]; then
    echo "Detected proteingym dataset - using cross-validation retraining approach"

    # For protein datasets, we need to retrain with CV and evaluate
    TRAIN_DATA="prepared_datasets/$DATASET/train.csv"

    for iteration in "${!ITERATION_PATHS[@]}"; do
        CODE_PATH="${ITERATION_PATHS[$iteration]}"
        ITERATION_DIR="$EXPERIMENT_FOLDER/$CODE_PATH"
        OUTPUT_FILE="$TEST_OUTPUT_DIR/${iteration}_test_predictions.csv"

        echo "Processing $iteration (code path: $CODE_PATH) with CV retraining..."

        # Setup conda environment at agent level (shorter path to avoid conda padding issues)
        AGENT_DIR="$(dirname "$ITERATION_DIR")"
        ENV_PATH="$AGENT_DIR/.conda/envs/${AGENT_ID}_env"
        if [[ ! -d "$ENV_PATH" ]]; then
            echo "Conda environment not found at: $ENV_PATH"
            if [[ ! -f "$ITERATION_DIR/conda_environment.yml" ]]; then
                echo "conda_environment.yml not found at: $ITERATION_DIR/conda_environment.yml"
                continue
            fi
            conda env create -f "$ITERATION_DIR/conda_environment.yml" -p "$ENV_PATH"
        else
            # Check if it's a valid conda environment
            if ! conda list -p "$ENV_PATH" >/dev/null 2>&1; then
                echo "Invalid conda environment at $ENV_PATH, recreating..."
                rm -rf "$ENV_PATH"
                conda env create -f "$ITERATION_DIR/conda_environment.yml" -p "$ENV_PATH"
            fi
        fi

        # Run protein CV evaluation inside Docker container
        EXPERIMENT_FOLDER_ABS="$(cd "$(dirname "$EXPERIMENT_FOLDER")" && pwd)/$(basename "$EXPERIMENT_FOLDER")"
        TRAIN_DATA_ABS="$(cd "$(dirname "$TRAIN_DATA")" && pwd)/$(basename "$TRAIN_DATA")"
        OUTPUT_FILE_ABS="$(cd "$(dirname "$OUTPUT_FILE")" && pwd)/$(basename "$OUTPUT_FILE")"
        ITERATION_DIR_REL="${ITERATION_DIR#$EXPERIMENT_FOLDER/}"
        AGENT_DIR_REL="$(dirname "$ITERATION_DIR_REL")"

        docker run --rm \
            --gpus all \
            --env NVIDIA_VISIBLE_DEVICES=all \
            -e PYTHONPATH=/repository/src \
            -e PATH="/experiment/$AGENT_DIR_REL/.conda/envs/${AGENT_ID}_env/bin:$PATH" \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(dirname "$TRAIN_DATA_ABS"):/train_data_dir:ro" \
            -v "$EXPERIMENT_FOLDER_ABS:/experiment" \
            -v "$(dirname "$OUTPUT_FILE_ABS"):/output_dir" \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            agentomics_img src/utils/generate_protein_cv_preds.py \
            --iteration-dir "/experiment/$ITERATION_DIR_REL" \
            --train-csv "/train_data_dir/$(basename "$TRAIN_DATA_ABS")" \
            --output-csv "/output_dir/$(basename "$OUTPUT_FILE_ABS")" \
            --agent-name "$AGENT_ID" || echo "Warning: Failed CV retraining for $iteration"
    done
else
    echo "Using standard inference approach"

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
fi

EXPERIMENT_FOLDER_ABS="$(cd "$(dirname "$EXPERIMENT_FOLDER")" && pwd)/$(basename "$EXPERIMENT_FOLDER")"
TEST_OUTPUT_DIR_ABS="$(cd "$(dirname "$TEST_OUTPUT_DIR")" && pwd)/$(basename "$TEST_OUTPUT_DIR")"
docker run --rm \
    --env-file $(pwd)/.env \
    -e PYTHONPATH=/repository/src \
    -v "$(pwd)/src":/repository/src:ro \
    -v "$(pwd)/prepared_test_sets":/repository/prepared_test_sets:ro \
    -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
    -v "$EXPERIMENT_FOLDER_ABS":/experiment:ro \
    -v "$TEST_OUTPUT_DIR_ABS":/test_outputs:ro \
    --entrypoint /opt/conda/envs/agentomics-env/bin/python \
    agentomics_img src/run_logging/evaluate_stealth_test.py \
    --dataset "$DATASET" \
    --test-output-dir /test_outputs \
    --experiment-folder /experiment

echo "Stealth test evaluation complete"